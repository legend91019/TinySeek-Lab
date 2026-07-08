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


if __name__ == "__main__":
    smoke(TinySeekConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2))
    smoke(TinySeekConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, use_moe=True, num_experts=4, top_k=2))
    smoke(TinySeekConfig(hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, attention_impl="educational_mla", mla_latent_size=24))
    assert rule_reward("The final answer is 42.", "42") == 1.0
    assert rule_reward("The final answer is 41.", "42") == 0.0
    print("smoke ok")
