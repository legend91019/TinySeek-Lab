from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "tinyseek.py"


def test_unified_v3_contract() -> None:
    source = MODEL_PATH.read_text(encoding="utf-8")
    required_contract = [
        "router_balance_strategy",
        "router_bias_update_rate",
        "mtp_depth",
        "mtp_loss_weight",
        "expert_ffn_multiplier",
        "def update_router_bias",
        '"mtp_loss"',
        '"lm_loss"',
    ]
    missing = [item for item in required_contract if item not in source]
    assert not missing, f"Unified V3 contract is incomplete: {missing}"
    dense_section = source.split("class DenseFFN", 1)[1].split("class MoEFFN", 1)[0]
    moe_section = source.split("class MoEFFN", 1)[1].split("class TinySeekBlock", 1)[0]
    assert "expert_ffn_multiplier" not in dense_section
    assert "expert_ffn_multiplier" in moe_section


if __name__ == "__main__":
    test_unified_v3_contract()
    print("unified V3 contract ok")
