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
- [`scripts/prepare_reasoning_data.py`](../scripts/prepare_reasoning_data.py) creates the formal structured arithmetic set and excludes mini-eval cases from training.

```bash
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --hourly_rate 2.18
```

The formal GPU suite uses 5,000 structured SFT examples:

```text
<think>Add the numbers: 15 + 17 = 32.</think>
<answer>32</answer>
```

```bash
python scripts/prepare_reasoning_data.py \
  --sft_out data/reasoning_sft.jsonl \
  --grpo_out data/reasoning_grpo.jsonl
python scripts/run_gpu_completion.py --data data/tinystories.jsonl
```

Mini eval reports tagged-answer accuracy and reasoning-format score separately, so a correct answer is not conflated with readable formatting.

The formal 4090 result is deliberately less flattering than the hypothesis: reasoning-format score moves from `0.0` at base to `0.6` after SFT, while all five held-out additions remain wrong. PPL on the first 100 TinyStories rows moves from `1.718` to `12.670`. Cold start partially teaches the output convention, but this mini-eval provides no arithmetic-generalization evidence and shows a large distribution shift.

See the [post-training code walkthrough](19_posttraining_code_walkthrough.md) and the [measured report](../experiments/gpu_completion_runs/report.md).

<!-- tinyseek-nav -->

---

Previous: [Stage 4: MLA](06_stage4_mla.md) | [Tutorial Index](README.md) | Next: [Stage 6: GRPO Mini](08_stage6_grpo_mini.md)
