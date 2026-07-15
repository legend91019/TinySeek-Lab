> This repository was created while studying the DeepSeek papers, as a practical aid for understanding them.

<div align="center">

# TinySeek-Lab

**Walk the DeepSeek LM research path with language models under a few hundred million parameters**

[中文说明](README_zh.md) | English

</div>

TinySeek-Lab is a bilingual, code-first course from model implementation through training and experiment reports. You write a complete Dense LM, evolve it into DeepSeekMoE, DeepSeek-V2, and DeepSeek-V3, then connect that base model to R1-style SFT and educational GRPO.

This repository is language-model-only. It excludes multimodal, vision, video, OCR, embodied, and agent tracks. The goal is to reproduce research questions and experimental method, not DeepSeek scale or final capability.

## Experiment-Driven, Not a Component Checklist

Every architecture transition follows the same loop:

```text
previous baseline -> measurable bottleneck -> research hypothesis
-> single-variable ablation -> prewritten decision gate
-> upgrade / retain the previous stage
```

DeepSeek papers provide problems, methods, and paper-scale evidence; TinySeek provides small-model code and runnable tests. Until GPU results exist, the repository labels a claim `Pending` instead of presenting a hypothesis as a TinySeek finding. See the [four-generation architecture map](docs/20_architecture_evolution_overview.md) and the [fair architecture experiment plan](experiments/06_architecture_evolution_plan.md).

## Four Generations, One Code Path

| Generation | Complete model you build | Main change | Code lesson |
| --- | --- | --- | --- |
| DeepSeek LLM | [`stage0_deepseek_llm.py`](model/stages/stage0_deepseek_llm.py) | Dense, RMSNorm, RoPE, SwiGLU, GQA | [Build the complete LM](docs/12_code_first_dense_lm.md) |
| DeepSeekMoE | [`stage1_deepseek_moe.py`](model/stages/stage1_deepseek_moe.py) | fine-grained routed and shared experts | [Dense to MoE](docs/21_from_dense_to_deepseek_moe.md) |
| DeepSeek-V2 | [`stage2_deepseek_v2.py`](model/stages/stage2_deepseek_v2.py) | MoE plus educational MLA | [MoE to V2](docs/22_from_moe_to_deepseek_v2.md) |
| DeepSeek-V3 | [`stage3_deepseek_v3.py`](model/stages/stage3_deepseek_v3.py) | auxiliary-loss-free routing bias and MTP | [V2 to V3](docs/23_from_v2_to_deepseek_v3.md) |

Start with the [architecture evolution map](docs/20_architecture_evolution_overview.md). Stage files teach the code; the unified [`model/tinyseek.py`](model/tinyseek.py) runs matched formal experiments.

## Current Results: Formal RTX 4090 Suite Complete

TinySeek-Lab has completed the full training and ablation suite on one RTX 4090:

```text
TinyStories -> tiny base -> dense 35M/115M -> LR/batch sweep
-> MoE -> MLA -> SFT -> GRPO mini -> mini eval -> cost and figures
```

- [48-run, 16-config, 3-seed architecture report](experiments/architecture_lab_runs/report.md): `1.5679 GPU h`, about `3.4180 CNY`.
- [11-run formal training and post-training report](experiments/gpu_completion_runs/report.md): `0.8985 GPU h`, about `1.9588 CNY`.
- Combined tracked trainer/post-training process time: `2.4664 GPU h`, corresponding to about `5.3768 CNY`; this excludes data preparation, standalone evaluation, report generation, and idle rental time.
- GQA passes its local gate: theoretical KV/token falls from `384` to `192` without a PPL regression.
- Shared experts improve PPL over coarse MoE but run about 35% slower, so the repository keeps separate quality and throughput branches.
- Educational MLA reaches `72` theoretical KV/token but regresses PPL; bias routing also fails to beat aux=0.01. Neither is promoted in this budget.
- On five held-out additions, SFT raises reasoning-format score from `0.0` to `0.6` but still scores `0/5`; GRPO then lowers format score to `0.2`. This mini-eval provides no arithmetic-generalization evidence, and a loose reward can degrade behavior.

These are TinySeek small-model measurements, not claims about DeepSeek-scale capability.

![TinySeek 3-seed architecture PPL](experiments/architecture_lab_runs/figures/architecture_ppl.svg)

## Three Quick Paths

| Path | Best for | Entry command |
| --- | --- | --- |
| CPU code course | Inspect four complete models and their shapes | `python scripts/inspect_stage_models.py` |
| Small GPU teaching run | Try tiny dense -> SFT -> GRPO | [Final GPU checklist](docs/18_gpu_fill_only_checklist.md) |
| RTX 4090 research run | Reproduce formal training and multi-seed architecture comparisons | [Experiment hub](experiments/README.md) |

Fast code-first route: read the [architecture map](docs/20_architecture_evolution_overview.md) and [Math-to-PyTorch toolkit](docs/24_math_to_pytorch.md), write the four stage models, study the [training loop](docs/16_training_loop_from_config_to_checkpoint.md), then run the [fair architecture plan](experiments/06_architecture_evolution_plan.md). The full reading path below starts with scope and papers first.

## Why "TinySeek"

DeepSeek's papers are unusually useful as a curriculum:

- DeepSeek LLM starts from training-recipe and scaling-law questions, including
  batch size and learning-rate searches.
- DeepSeekMoE explains expert specialization and load balancing.
- DeepSeek-V2 combines DeepSeekMoE with MLA for economical training and
  efficient inference.
- DeepSeek-V3 validates the MoE + MLA line at larger scale and introduces
  auxiliary-loss-free balancing and multi-token prediction.
- DeepSeek-R1 shows how a strong base model can be post-trained with cold-start
  reasoning SFT, rejection sampling, and GRPO-style rule RL.

TinySeek-Lab turns those ideas into a sequence of small experiments.

## Roadmap at a Glance

```mermaid
flowchart LR
  A["DeepSeek LLM<br/>Dense"] --> B["DeepSeekMoE<br/>Sparse FFN"]
  B --> C["DeepSeek-V2<br/>MLA"]
  C --> D["DeepSeek-V3<br/>Bias + MTP"]
  D --> E["DeepSeek-R1<br/>SFT + GRPO"]
```

## Model Evolution

```mermaid
flowchart TB
  subgraph Dense["Dense baseline"]
    T0["Byte/BPE tokens"] --> E0["Embedding"]
    E0 --> B0["Transformer blocks<br/>MHA/GQA + Dense SwiGLU"]
    B0 --> L0["LM head"]
  end

  subgraph MoE["MoE upgrade"]
    R["Router"] --> X1["Expert 1"]
    R --> X2["Expert 2"]
    R --> XN["Expert N"]
    X1 --> M["Weighted sum"]
    X2 --> M
    XN --> M
  end

  subgraph MLA["Educational MLA"]
    H["Hidden states"] --> Z["Low-rank latent KV"]
    Z --> K["Reconstructed K"]
    Z --> V["Reconstructed V"]
  end
```

## Repository Layout

```text
TinySeek-Lab/
  configs/              Small model and experiment configs
  dataset/              Dataset wrappers and byte tokenizer
  docs/                 Chapter-style tutorial notes
  experiments/          Sweep plans and report templates
  model/stages/         Four complete teaching models
  model/tinyseek.py     Unified formal experiment model
  scripts/              Data prep and generation helpers
  trainer/              Pretrain, SFT, sweep, and GRPO entry points
  tests/                Smoke tests
```

## Quick Start

Install dependencies first:

```bash
pip install -r requirements.txt
```

Create a toy dataset:

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
```

Run a tiny pretraining smoke test:

```bash
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 20
```

Generate from the checkpoint:

```bash
python scripts/generate.py --config configs/tiny_dense.json --ckpt out/tiny_dense_last.pt --prompt "DeepSeek is"
```

Run the LR / batch-size grid from the DeepSeek LLM-inspired chapter:

```bash
python trainer/sweep_pretrain.py --sweep experiments/01_lr_batch_grid.json
```

Track AutoDL GPU cost during a run:

```bash
# RTX 4090: 2.18 CNY/hour
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --hourly_rate 2.18

# Summarize all run ledgers
python scripts/summarize_costs.py --input_dir out
```

Run the post-training toy path:

```bash
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --hourly_rate 2.18

python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --hourly_rate 2.18
```

The first AutoDL RTX 4090 validation report is in
[experiments/02_autodl_4090_smoke_report.md](experiments/02_autodl_4090_smoke_report.md).
The v1 pretrain -> SFT -> GRPO smoke report is in
[experiments/03_v1_pipeline_smoke_report.md](experiments/03_v1_pipeline_smoke_report.md).
The latest 3-seed architecture measurements are in
[experiments/architecture_lab_runs/report.md](experiments/architecture_lab_runs/report.md).
The formal training, sweep, and post-training results are in
[experiments/gpu_completion_runs/report.md](experiments/gpu_completion_runs/report.md).
The earlier [RTX 4090 v1 report](experiments/05_4090_v1_results.md) remains as a record of the repository's progression from smoke validation to formal experiments.
Read the code path in [docs/15_code_walkthrough.md](docs/15_code_walkthrough.md).
The preregistered paid-GPU plan that produced the formal suite is archived in
[experiments/04_formal_experiment_plan.md](experiments/04_formal_experiment_plan.md).

## First Reading Path

Read these docs in order, or open the full [tutorial index](docs/README.md):

1. [Project Scope](docs/00_project_scope.md)
2. [DeepSeek Paper Map for LM Training](docs/01_deepseek_lm_paper_map.md)
3. [Four-Generation Architecture Map](docs/20_architecture_evolution_overview.md)
4. [Math to PyTorch: formulas, shapes, and APIs](docs/24_math_to_pytorch.md)
5. [Build the DeepSeek LLM Dense Baseline](docs/12_code_first_dense_lm.md)
6. [Dense to DeepSeekMoE](docs/21_from_dense_to_deepseek_moe.md)
7. [MoE to DeepSeek-V2](docs/22_from_moe_to_deepseek_v2.md)
8. [V2 to DeepSeek-V3](docs/23_from_v2_to_deepseek_v3.md)
9. [Training Loop: From Config to Checkpoint](docs/16_training_loop_from_config_to_checkpoint.md)
10. [SFT and Reasoning Cold Start](docs/07_stage5_sft_cold_start.md)
11. [Rule-Based GRPO Mini](docs/08_stage6_grpo_mini.md)
12. [Post-Training Code Walkthrough](docs/19_posttraining_code_walkthrough.md)

Chinese tutorial notes:

Open the full [中文教程目录](docs/zh/README.md), or read in this order:

1. [项目范围](docs/zh/00_project_scope.md)
2. [DeepSeek 语言模型论文地图](docs/zh/01_deepseek_lm_paper_map.md)
3. [四代架构演进总览](docs/zh/20_architecture_evolution_overview.md)
4. [数学到 PyTorch：公式、Shape 与 API](docs/zh/24_math_to_pytorch.md)
5. [从零写 DeepSeek LLM Dense 基线](docs/zh/12_code_first_dense_lm.md)
6. [从 Dense 改到 DeepSeekMoE](docs/zh/21_from_dense_to_deepseek_moe.md)
7. [从 MoE 改到 DeepSeek-V2](docs/zh/22_from_moe_to_deepseek_v2.md)
8. [从 V2 改到 DeepSeek-V3](docs/zh/23_from_v2_to_deepseek_v3.md)
9. [训练主循环](docs/zh/16_training_loop_from_config_to_checkpoint.md)
10. [SFT 和 Reasoning Cold Start](docs/zh/07_stage5_sft_cold_start.md)
11. [Rule-Based GRPO Mini](docs/zh/08_stage6_grpo_mini.md)
12. [后训练代码细读](docs/zh/19_posttraining_code_walkthrough.md)

Chinese supplements:

- [总训练路线图](docs/zh/02_training_roadmap.md)
- [当前进度](docs/zh/04_current_progress.md)

## DeepSeek Papers Used

The local source folder is expected at:

```text
../DeepSeek-papers/chronological-pdfs
```

The tutorial uses only LM-relevant papers: DeepSeek LLM, DeepSeekMoE,
DeepSeek-V2/V3/V3.2/V4, DeepSeek-R1, DeepSeekMath/Prover, ESFT, Native Sparse
Attention, and reward-model/RL papers. Multimodal and OCR papers are excluded
from the main path.

## Philosophy

Every experiment should have:

- Hypothesis: what are we testing?
- Setup: model size, data, token budget, hardware.
- Sweep: which hyperparameters change?
- Metrics: train loss, validation loss, tokens/sec, memory, downstream mini eval.
- Takeaway: what did we learn?

TinySeek-Lab is a lab notebook disguised as a repo.

## Current Status

The current version contains four complete DeepSeek LLM/MoE/V2/V3 teaching
models, a unified configurable experiment model with routing bias and MTP,
pretraining, sweeps, SFT, educational GRPO, mini eval, GPU cost tracking,
measured formal 4090 reports, 48 multi-seed architecture runs, and eleven long-training/post-training runs. GRPO
and MLA remain educational rather than production reproductions.
