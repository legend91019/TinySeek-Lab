from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.stages import (
    Stage0Config,
    Stage0DeepSeekLM,
    Stage1Config,
    Stage1DeepSeekMoE,
    Stage2Config,
    Stage2DeepSeekV2,
    Stage3Config,
    Stage3DeepSeekV3,
)


def build_stages() -> list[tuple[str, torch.nn.Module, object]]:
    common = {
        "vocab_size": 64,
        "max_seq_len": 16,
        "hidden_size": 32,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 2,
    }
    stage0_config = Stage0Config(**common)
    stage1_config = Stage1Config(
        **common,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
    )
    stage2_config = Stage2Config(
        **common,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
        kv_lora_rank=12,
        qk_rope_head_dim=4,
    )
    stage3_config = Stage3Config(
        **common,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
        kv_lora_rank=12,
        qk_rope_head_dim=4,
        mtp_depth=1,
    )
    return [
        ("stage0_deepseek_llm", Stage0DeepSeekLM(stage0_config), stage0_config),
        ("stage1_deepseek_moe", Stage1DeepSeekMoE(stage1_config), stage1_config),
        ("stage2_deepseek_v2", Stage2DeepSeekV2(stage2_config), stage2_config),
        ("stage3_deepseek_v3", Stage3DeepSeekV3(stage3_config), stage3_config),
    ]


@torch.no_grad()
def inspect_stage(name: str, model: torch.nn.Module, config: object) -> dict:
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    model.eval()
    out = model(input_ids, input_ids)
    return {
        "stage": name,
        "total_params": model.parameter_count(),
        "activated_params": model.activated_parameter_estimate(),
        "kv_cache_elements_per_token_per_layer": model.kv_cache_elements_per_token(),
        "logits_shape": list(out["logits"].shape),
        "loss_keys": sorted(key for key in out if "loss" in key),
        "loss": float(out["loss"].item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect all four TinySeek teaching stages")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    rows = [inspect_stage(name, model, config) for name, model, config in build_stages()]
    rendered = json.dumps(rows, indent=2, ensure_ascii=False)
    print(rendered)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
