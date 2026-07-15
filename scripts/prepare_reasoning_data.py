from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


HOLDOUT_ADDITION = {(2, 3), (7, 8), (12, 9), (15, 17), (20, 22)}


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def structured_response(left: int, operator: str, right: int, answer: int) -> str:
    operation = {"+": "Add", "-": "Subtract", "*": "Multiply"}[operator]
    return (
        f"<think>{operation} the numbers: {left} {operator} {right} = {answer}.</think>\n"
        f"<answer>{answer}</answer>"
    )


def build_sft_rows(num_samples: int, seed: int, max_operand: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    seen: set[tuple[str, int, int]] = set()
    while len(rows) < num_samples:
        draw = rng.random()
        operator = "+" if draw < 0.7 else "-" if draw < 0.9 else "*"
        limit = min(max_operand, 12) if operator == "*" else max_operand
        left, right = rng.randint(1, limit), rng.randint(1, limit)
        if operator == "-" and right > left:
            left, right = right, left
        if operator == "+" and (left, right) in HOLDOUT_ADDITION:
            continue
        key = (operator, left, right)
        if key in seen:
            continue
        seen.add(key)
        answer = left + right if operator == "+" else left - right if operator == "-" else left * right
        rows.append(
            {
                "prompt": f"Compute {left} {operator} {right}. Show concise reasoning and put the final integer in <answer> tags.",
                "response": structured_response(left, operator, right, answer),
                "task": {"+": "addition", "-": "subtraction", "*": "multiplication"}[operator],
            }
        )
    return rows


def build_grpo_rows(num_samples: int, seed: int, max_operand: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    seen: set[tuple[int, int]] = set()
    while len(rows) < num_samples:
        left, right = rng.randint(1, max_operand), rng.randint(1, max_operand)
        if (left, right) in HOLDOUT_ADDITION or (left, right) in seen:
            continue
        seen.add((left, right))
        rows.append(
            {
                "prompt": f"Compute {left} + {right}. Show concise reasoning and put the final integer in <answer> tags.",
                "answer": str(left + right),
                "task": "addition",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured cold-start SFT and rule-based GRPO arithmetic data")
    parser.add_argument("--sft_out", default="data/reasoning_sft.jsonl")
    parser.add_argument("--grpo_out", default="data/reasoning_grpo.jsonl")
    parser.add_argument("--sft_samples", type=int, default=5000)
    parser.add_argument("--grpo_samples", type=int, default=1000)
    parser.add_argument("--max_operand", type=int, default=99)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    sft_rows = build_sft_rows(args.sft_samples, args.seed, args.max_operand)
    grpo_rows = build_grpo_rows(args.grpo_samples, args.seed + 4, args.max_operand)
    save_jsonl(args.sft_out, sft_rows)
    save_jsonl(args.grpo_out, grpo_rows)
    print(f"wrote {len(sft_rows)} SFT rows to {args.sft_out}")
    print(f"wrote {len(grpo_rows)} GRPO rows to {args.grpo_out}")


if __name__ == "__main__":
    main()
