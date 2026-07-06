from __future__ import annotations

import argparse
import random
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trainer.utils import save_jsonl


SEEDS = [
    "DeepSeek LLM studies scaling laws for batch size and learning rate before large scale training.",
    "A decoder-only language model learns by predicting the next token from a text corpus.",
    "SwiGLU, RMSNorm, RoPE and grouped-query attention are common modern language model blocks.",
    "Mixture-of-Experts routes each token to a small number of experts and keeps activated parameters low.",
    "Load balancing prevents routing collapse where all tokens choose the same expert.",
    "MLA compresses key value information into a latent representation to reduce cache cost.",
    "Supervised fine-tuning teaches a base model to follow an instruction format.",
    "Cold-start reasoning data gives a model a readable thinking pattern before rule-based RL.",
    "GRPO compares multiple sampled answers in a group and optimizes relative advantages.",
    "Rejection sampling keeps correct model-generated solutions and turns them back into SFT data."
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/toy_pretrain.jsonl")
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    rows = []
    rng = random.Random(42)
    for i in range(args.num_samples):
        a, b = rng.sample(SEEDS, 2)
        rows.append({"text": f"{a} {b} This is TinySeek sample {i}."})
    save_jsonl(args.out, rows)
    print(f"wrote {len(rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
