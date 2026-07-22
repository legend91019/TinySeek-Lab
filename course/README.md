# TinySeek Course: Rebuild DeepSeek LM, One Experiment at a Time

[中文主线](README_zh.md) | English

This is the canonical reading path for TinySeek-Lab. It follows a **research decision loop**, not a list of Transformer components:

```text
observe a bottleneck -> read the paper clue -> write the smallest code change
-> preregister a controlled comparison -> inspect evidence -> promote or reject
```

The story is a reproducible reconstruction of the public research path. It is not a claim about DeepSeek's private chronological decision process. Paper-scale evidence and TinySeek's small-model measurements are always separated.

## The Main Line

| Unit | Starting problem | Code change | Experiment that decides the next step |
| --- | --- | --- | --- |
| [s01 Dense baseline](s01_dense_baseline/README.md) | Can we build and train a complete decoder LM? | RMSNorm, RoPE, GQA, SwiGLU, residual blocks and LM loss | Stage smoke test and a matched dense baseline |
| [s02 Training recipe](s02_training_recipe/README.md) | Which LR and batch size are usable before architecture comparisons? | No architecture change; make the training loop measurable | DeepSeek-LLM-inspired LR/batch grid |
| [s03 GQA](s03_gqa/README.md) | Can attention reduce KV state without hurting language modeling? | MHA -> grouped K/V heads | 3-seed MHA vs GQA comparison |
| [s04 DeepSeekMoE](s04_deepseek_moe/README.md) | Can capacity grow without activating every parameter? | Dense FFN -> routed and shared experts | coarse/fine/shared plus routing-balance ablations |
| [s05 MLA](s05_mla/README.md) | Can the KV cache become smaller than GQA? | Low-rank KV path plus decoupled RoPE path | control, low-rank and educational MLA comparison |
| [s06 V3 routing + MTP](s06_v3_routing_mtp/README.md) | Can balancing avoid an auxiliary loss, and can extra targets help? | routing bias and multi-token prediction | aux vs bias and MTP on/off |
| [s07 Cold-start SFT](s07_cold_start_sft/README.md) | How does a base LM learn a readable reasoning format? | masked supervised fine-tuning | base vs cold-start SFT format/answer/PPL evaluation |
| [s08 GRPO + evaluation](s08_grpo_and_evaluation/README.md) | Does rule-based RL improve the behavior learned by SFT? | group sampling, rewards and normalized advantages | direct RL vs SFT -> GRPO, with explicit limitations |

## How To Read One Unit

Every unit has the same five questions:

1. **What is the bottleneck?** A metric or implementation limitation from the previous unit.
2. **What did the paper contribute?** The public motivation and the part TinySeek keeps or simplifies.
3. **What is the smallest code diff?** The exact model file, config and tensor contract.
4. **What would count as a win?** A prewritten decision gate, including quality, speed, memory and uncertainty.
5. **What did we decide?** Promote, keep as a branch, reject, or mark inconclusive.

The negative results are intentional. A research tutorial should teach you how an experiment can prevent a fashionable architecture from being promoted.

## Evidence Map

- Architecture multi-seed evidence: [`experiments/architecture_lab_runs/report.md`](../experiments/architecture_lab_runs/report.md) and [中文报告](../experiments/architecture_lab_runs/report_zh.md).
- Formal 4090 training and post-training evidence: [`experiments/gpu_completion_runs/report.md`](../experiments/gpu_completion_runs/report.md) and [中文报告](../experiments/gpu_completion_runs/report_zh.md).
- Formula -> tensor shape -> PyTorch API: [`docs/24_math_to_pytorch.md`](../docs/24_math_to_pytorch.md).
- Full source walkthrough: [`docs/15_code_walkthrough.md`](../docs/15_code_walkthrough.md).
- Old chapter notes remain available in [`docs/README.md`](../docs/README.md) as reference material.

## Reproduction Contract

The course uses the same dataset hash, configs, seeds, token budgets and cost ledger as the archived reports. For a quick start, run:

```bash
pip install -r requirements.txt
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 20
python scripts/inspect_stage_models.py
```

Formal GPU commands and environment details belong to the linked experiment reports, so the lesson text remains readable.

## Scope

LM only: no vision, video, OCR, multimodal, embodied or agent track. The repository is a practical work created while studying the DeepSeek papers.
