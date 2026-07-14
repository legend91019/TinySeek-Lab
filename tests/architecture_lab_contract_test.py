from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "architecture_lab"


def load(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.exists(), f"Missing architecture lab config: {name}"
    return json.loads(path.read_text(encoding="utf-8"))


def assert_pair_differs_only(left_name: str, right_name: str, allowed_model_keys: set[str]) -> None:
    left = load(left_name)
    right = load(right_name)
    assert left["train"] == right["train"]
    all_keys = set(left["model"]) | set(right["model"])
    changed = {key for key in all_keys if left["model"].get(key) != right["model"].get(key)}
    assert changed == allowed_model_keys, (left_name, right_name, changed)


def test_architecture_lab_contract() -> None:
    assert_pair_differs_only("dense_mha.json", "dense_gqa.json", {"num_kv_heads"})
    assert_pair_differs_only(
        "moe_aux.json",
        "moe_bias.json",
        {"router_balance_strategy", "moe_aux_loss_weight"},
    )
    assert_pair_differs_only("v3_no_mtp.json", "v3_mtp.json", {"mtp_depth"})
    assert_pair_differs_only(
        "v2_attention_control.json",
        "v2_mla.json",
        {"attention_impl"},
    )
    assert (ROOT / "scripts" / "inspect_stage_models.py").exists()
    assert (ROOT / "experiments" / "06_architecture_evolution_plan_zh.md").exists()
    assert (ROOT / "experiments" / "06_architecture_evolution_plan.md").exists()


if __name__ == "__main__":
    test_architecture_lab_contract()
    print("architecture lab contract ok")
