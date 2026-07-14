from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE0_PATH = REPO_ROOT / "model" / "stages" / "stage0_deepseek_llm.py"
STAGE1_PATH = REPO_ROOT / "model" / "stages" / "stage1_deepseek_moe.py"


def require_stage0_file() -> None:
    assert STAGE0_PATH.exists(), "Stage 0 teaching model has not been implemented"


def require_stage1_file() -> None:
    assert STAGE1_PATH.exists(), "Stage 1 teaching model has not been implemented"


def test_stage0() -> None:
    require_stage0_file()
    if importlib.util.find_spec("torch") is None:
        print("stage0 dynamic test skipped: PyTorch is not installed")
        return

    import torch

    from model.stages import Stage0Config, Stage0DeepSeekLM

    cfg = Stage0Config(
        vocab_size=64,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
    )
    model = Stage0DeepSeekLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(input_ids, input_ids)
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    assert out["aux_loss"].ndim == 0
    assert model.kv_cache_elements_per_token() == 2 * cfg.num_kv_heads * (cfg.hidden_size // cfg.num_heads)


def test_stage1() -> None:
    require_stage1_file()
    if importlib.util.find_spec("torch") is None:
        print("stage1 dynamic test skipped: PyTorch is not installed")
        return

    import torch

    from model.stages import Stage1Config, Stage1DeepSeekMoE

    cfg = Stage1Config(
        vocab_size=64,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k=2,
    )
    model = Stage1DeepSeekMoE(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(input_ids, input_ids)
    stats = model.expert_load_summary()
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    assert out["aux_loss"].ndim == 0
    assert sum(stats["total_counts"]) == cfg.num_layers * input_ids.numel() * cfg.top_k
    assert model.activated_parameter_estimate() < model.parameter_count()


if __name__ == "__main__":
    test_stage0()
    test_stage1()
    print("stage model tests ok")
