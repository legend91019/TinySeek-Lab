from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def test_reasoning_data_is_structured_and_holds_out_eval_cases() -> None:
    from scripts.prepare_reasoning_data import HOLDOUT_ADDITION, build_grpo_rows, build_sft_rows

    sft_rows = build_sft_rows(num_samples=500, seed=7, max_operand=50)
    assert len(sft_rows) == 500
    assert all("<think>" in row["response"] and "<answer>" in row["response"] for row in sft_rows)
    prompts = {row["prompt"] for row in sft_rows}
    for left, right in HOLDOUT_ADDITION:
        assert f"Compute {left} + {right}." not in prompts

    grpo_rows = build_grpo_rows(num_samples=200, seed=11, max_operand=50)
    assert len(grpo_rows) == 200
    assert all(row["answer"].lstrip("-").isdigit() for row in grpo_rows)
    assert len({row["prompt"] for row in grpo_rows}) > 150


if __name__ == "__main__":
    test_reasoning_data_is_structured_and_holds_out_eval_cases()
    print("reasoning data tests ok")
