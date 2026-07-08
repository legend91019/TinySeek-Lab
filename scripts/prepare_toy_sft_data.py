from __future__ import annotations

import argparse
import random
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trainer.utils import save_jsonl


TEMPLATES = [
    ("What is a decoder-only language model?", "A decoder-only language model predicts the next token from previous tokens."),
    ("Why use RMSNorm?", "RMSNorm stabilizes hidden states with a simple root-mean-square normalization."),
    ("What does RoPE do?", "RoPE injects position information by rotating query and key vectors."),
    ("Why does MoE use routing?", "MoE routes tokens to a small number of experts so activated parameters stay lower than total parameters."),
    ("What problem does load balancing solve?", "Load balancing reduces routing collapse, where many tokens choose the same expert."),
    ("What is cold-start reasoning data?", "Cold-start reasoning data teaches readable and structured reasoning before rule-based RL."),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy_sft.jsonl")
    parser.add_argument("--num_samples", type=int, default=600)
    args = parser.parse_args()

    rng = random.Random(7)
    rows = []
    for i in range(args.num_samples):
        prompt, response = rng.choice(TEMPLATES)
        rows.append({"prompt": prompt, "response": f"{response} This is SFT example {i}."})
    save_jsonl(args.out, rows)
    print(f"wrote {len(rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
