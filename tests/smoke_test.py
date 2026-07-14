from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import TinySeekConfig, TinySeekForCausalLM
from model.routing import sequence_load_balance_loss
from eval.mini_eval import has_response_content
from trainer.train_pretrain import evaluate
from trainer.train_grpo import rule_reward


def smoke(config: TinySeekConfig) -> None:
    model = TinySeekForCausalLM(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    out = model(x, x)
    assert out["logits"].shape == (2, 16, config.vocab_size)
    assert out["loss"].ndim == 0
    assert out["lm_loss"].ndim == 0
    assert out["mtp_loss"].ndim == 0


if __name__ == "__main__":
    assert not has_response_content("### Instruction\nQuestion\n\n### Response\n")
    assert has_response_content("### Instruction\nQuestion\n\n### Response\nA real answer.")
    smoke(TinySeekConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2))
    smoke(TinySeekConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, use_moe=True, num_experts=4, top_k=2))
    smoke(TinySeekConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, attention_impl="educational_mla", mla_latent_size=24))
    v3 = TinySeekConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        use_moe=True,
        num_experts=4,
        top_k=2,
        router_balance_strategy="bias",
        router_bias_update_rate=0.01,
        mtp_depth=1,
        mtp_loss_weight=0.1,
    )
    smoke(v3)
    v3_model = TinySeekForCausalLM(v3)
    v3_input = torch.randint(0, v3.vocab_size, (2, 16))
    v3_model(v3_input, v3_input)
    before = v3_model.blocks[0].ffn.expert_bias.clone()
    v3_model.blocks[0].ffn.last_expert_counts.copy_(torch.tensor([20, 2, 2, 2]))
    v3_model.update_router_bias()
    assert v3_model.blocks[0].ffn.expert_bias[0] < before[0]

    state = v3_model.state_dict()
    bias_keys = [key for key in state if key.endswith("expert_bias")]
    assert bias_keys
    legacy_state = {key: value for key, value in state.items() if not key.endswith("expert_bias")}
    TinySeekForCausalLM(v3).load_state_dict(legacy_state)

    decoupled_mla = TinySeekConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        attention_impl="educational_mla",
        mla_latent_size=24,
        mla_decoupled_rope=True,
        qk_rope_head_dim=8,
    )
    mla_model = TinySeekForCausalLM(decoupled_mla)
    smoke(decoupled_mla)
    assert mla_model.blocks[0].attn.kv_cache_elements_per_token() == 32

    balanced_scores = torch.full((4, 2), 0.5)
    balanced_top = torch.tensor([[0], [1], [0], [1]])
    specialized_scores = torch.tensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    specialized_top = torch.tensor([[0], [0], [1], [1]])
    balanced_loss = sequence_load_balance_loss(balanced_scores, balanced_top, 2, 2, 2, 1)
    specialized_loss = sequence_load_balance_loss(specialized_scores, specialized_top, 2, 2, 2, 1)
    assert specialized_loss > balanced_loss

    eval_input = torch.randint(0, v3.vocab_size, (4, 16))
    eval_loader = DataLoader(TensorDataset(eval_input, eval_input), batch_size=2)
    metrics = evaluate(TinySeekForCausalLM(v3), eval_loader, "cpu")
    assert set(metrics) == {"objective", "lm_loss", "mtp_loss", "aux_loss"}
    assert metrics["objective"] >= metrics["lm_loss"]
    assert rule_reward("The final answer is 42.", "42") == 1.0
    wrong_but_parseable = rule_reward("The final answer is 41.", "42")
    assert 0.0 < wrong_but_parseable < 1.0
    print("smoke ok")
