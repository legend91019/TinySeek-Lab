# 07. Stage 5: SFT and Reasoning Cold Start

Goal: move from base LM to instruction-following and readable reasoning.

## DeepSeek Anchor

DeepSeek-R1 distinguishes:

- R1-Zero: RL directly from a pretrained base model.
- R1: cold-start reasoning data first, then RL, then rejection sampling and SFT.

The cold-start stage is best understood as a small reasoning SFT stage. It
teaches the answer format and readable reasoning style before RL optimizes
correctness.

## TinySeek Plan

1. Full SFT on instruction data.
2. Reasoning cold-start SFT on high-quality math/code examples.
3. Direct rule RL from base vs rule RL after cold start.

Expected result:

- direct RL may improve reward but produce messy outputs;
- cold-start + RL should produce more readable answers.

Implementation status:

- SFT script skeleton comes next.
- Dataset format will use chat-style JSONL.

<!-- tinyseek-nav -->

---

Previous: [Stage 4: MLA](06_stage4_mla.md) | [Tutorial Index](README.md) | Next: [Stage 6: GRPO Mini](08_stage6_grpo_mini.md)
