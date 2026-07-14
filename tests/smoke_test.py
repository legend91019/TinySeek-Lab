from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import TinySeekConfig, TinySeekForCausalLM
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
    assert rule_reward("The final answer is 42.", "42") == 1.0
    wrong_but_parseable = rule_reward("The final answer is 41.", "42")
    assert 0.0 < wrong_but_parseable < 1.0
    print("smoke ok")
