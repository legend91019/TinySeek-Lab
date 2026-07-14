from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE0_PATH = REPO_ROOT / "model" / "stages" / "stage0_deepseek_llm.py"
STAGE1_PATH = REPO_ROOT / "model" / "stages" / "stage1_deepseek_moe.py"
STAGE2_PATH = REPO_ROOT / "model" / "stages" / "stage2_deepseek_v2.py"
STAGE3_PATH = REPO_ROOT / "model" / "stages" / "stage3_deepseek_v3.py"


def require_stage0_file() -> None:
    assert STAGE0_PATH.exists(), "Stage 0 teaching model has not been implemented"


def require_stage1_file() -> None:
    assert STAGE1_PATH.exists(), "Stage 1 teaching model has not been implemented"


def require_stage2_file() -> None:
    assert STAGE2_PATH.exists(), "Stage 2 teaching model has not been implemented"


def require_stage3_file() -> None:
    assert STAGE3_PATH.exists(), "Stage 3 teaching model has not been implemented"


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


def test_stage2() -> None:
    require_stage2_file()
    if importlib.util.find_spec("torch") is None:
        print("stage2 dynamic test skipped: PyTorch is not installed")
        return

    import torch

    from model.stages import Stage2Config, Stage2DeepSeekV2

    cfg = Stage2Config(
        vocab_size=64,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_routed_experts=4,
        top_k=2,
        kv_lora_rank=12,
        qk_rope_head_dim=4,
    )
    model = Stage2DeepSeekV2(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(input_ids, input_ids)
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    assert model.kv_cache_elements_per_token() == cfg.kv_lora_rank + cfg.qk_rope_head_dim


def test_stage3() -> None:
    require_stage3_file()
    if importlib.util.find_spec("torch") is None:
        print("stage3 dynamic test skipped: PyTorch is not installed")
        return

    import torch

    from model.stages import Stage3Config, Stage3DeepSeekV3
    from model.stages.stage3_deepseek_v3 import BiasBalancedMoE

    cfg = Stage3Config(
        vocab_size=64,
        max_seq_len=16,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_routed_experts=4,
        top_k=2,
        kv_lora_rank=12,
        qk_rope_head_dim=4,
        router_bias_update_rate=0.01,
        mtp_depth=1,
        mtp_loss_weight=0.1,
    )
    router = BiasBalancedMoE(cfg)
    before = router.expert_bias.clone()
    router.last_expert_counts.copy_(torch.tensor([20, 2, 2, 2]))
    router.update_bias()
    assert router.expert_bias[0] < before[0]
    assert torch.all(router.expert_bias[1:] > before[1:])

    model = Stage3DeepSeekV3(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 12))
    out = model(input_ids, input_ids)
    assert out["logits"].shape == (2, 12, cfg.vocab_size)
    assert torch.isfinite(out["lm_loss"])
    assert torch.isfinite(out["mtp_loss"])
    assert torch.allclose(
        out["loss"], out["lm_loss"] + cfg.mtp_loss_weight * out["mtp_loss"]
    )
    model.update_router_bias()


if __name__ == "__main__":
    test_stage0()
    test_stage1()
    test_stage2()
    test_stage3()
    print("stage model tests ok")
