from __future__ import annotations

import argparse
import random
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trainer.utils import save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy_grpo.jsonl")
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()

    rng = random.Random(11)
    rows = []
    for _ in range(args.num_samples):
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        rows.append({"prompt": f"Compute {a} + {b}. Answer with the final integer.", "answer": str(a + b)})
    save_jsonl(args.out, rows)
    print(f"wrote {len(rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
