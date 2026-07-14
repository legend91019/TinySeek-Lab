# 07. Stage 5: SFT and Reasoning Cold Start

Goal: move from base LM to instruction-following and readable reasoning.

## DeepSeek Anchor

DeepSeek-R1 distinguishes:

- R1-Zero: RL directly from a pretrained base model.
- R1: cold-start reasoning data first, then RL, then rejection sampling and SFT.

The cold-start stage is best understood as a small reasoning SFT stage. It
teaches the answer format and readable reasoning style before RL optimizes
correctness.

## Implemented TinySeek Path

1. Full SFT on instruction data.
2. Reasoning cold-start SFT on high-quality math/code examples.
3. Direct rule RL from base vs rule RL after cold start.

Expected result:

- direct RL may improve reward but produce messy outputs;
- cold-start + RL should produce more readable answers.

The path is implemented:

- [`dataset/lm_dataset.py`](../dataset/lm_dataset.py) masks prompt labels with `-100` and trains only response tokens.
- [`trainer/train_sft.py`](../trainer/train_sft.py) loads a base checkpoint and runs SFT.
- [`scripts/prepare_toy_sft_data.py`](../scripts/prepare_toy_sft_data.py) creates minimal cold-start teaching data.

```bash
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --hourly_rate 2.18
```

See the [post-training code walkthrough](19_posttraining_code_walkthrough.md). The 4090 v1 run proves that SFT learns the toy format, but it worsened TinyStories PPL. Learning a format is not evidence of broad capability improvement.

<!-- tinyseek-nav -->

---

Previous: [Stage 4: MLA](06_stage4_mla.md) | [Tutorial Index](README.md) | Next: [Stage 6: GRPO Mini](08_stage6_grpo_mini.md)
