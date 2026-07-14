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


def expert_capacity(config_name: str) -> tuple[float, float]:
    model = load(config_name)["model"]
    expert_width = model["expert_ffn_multiplier"]
    total = (model["num_experts"] + model["num_shared_experts"]) * expert_width
    activated = (model["top_k"] + model["num_shared_experts"]) * expert_width
    return total, activated


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
    assert_pair_differs_only(
        "v2_low_rank_control.json",
        "v2_low_rank_kv.json",
        {"attention_impl"},
    )
    assert load("v2_low_rank_kv.json")["model"]["mla_decoupled_rope"] is False
    assert load("v2_mla.json")["model"]["mla_decoupled_rope"] is True
    moe_evolution = ["moe_coarse.json", "moe_fine_grained.json", "moe_shared.json"]
    moe_configs = [load(name) for name in moe_evolution]
    assert all(config["train"] == moe_configs[0]["train"] for config in moe_configs[1:])
    assert {expert_capacity(name) for name in moe_evolution} == {(8.0, 4.0)}
    assert moe_configs[0]["model"]["num_shared_experts"] == 0
    assert moe_configs[1]["model"]["num_shared_experts"] == 0
    assert moe_configs[2]["model"]["num_shared_experts"] == 1
    aux_sweep = ["moe_aux_none.json", "moe_aux_weak.json", "moe_aux.json", "moe_aux_strong.json"]
    aux_configs = [load(name) for name in aux_sweep]
    assert all(config["train"] == aux_configs[0]["train"] for config in aux_configs[1:])
    assert [config["model"]["moe_aux_loss_weight"] for config in aux_configs] == [0.0, 0.001, 0.01, 0.1]
    for config in aux_configs:
        model_without_weight = {k: v for k, v in config["model"].items() if k != "moe_aux_loss_weight"}
        baseline_without_weight = {
            k: v for k, v in aux_configs[0]["model"].items() if k != "moe_aux_loss_weight"
        }
        assert model_without_weight == baseline_without_weight
    assert (ROOT / "scripts" / "inspect_stage_models.py").exists()
    assert (ROOT / "experiments" / "06_architecture_evolution_plan_zh.md").exists()
    assert (ROOT / "experiments" / "06_architecture_evolution_plan.md").exists()


if __name__ == "__main__":
    test_architecture_lab_contract()
    print("architecture lab contract ok")
