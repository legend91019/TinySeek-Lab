# TinySeek-Lab

[中文说明](README_zh.md) | English

TinySeek-Lab is a tutorial repository for learning language-model training by
walking through a small-scale version of DeepSeek's LM research path.

The goal is not to reproduce DeepSeek's results. The goal is to first write the
initial dense model code, then reproduce the research moves at a scale that a
learner can run:

1. Write the first DeepSeek-style dense decoder-only LM by hand.
2. Train the dense baseline.
3. Reproduce small LR / batch-size sweeps inspired by DeepSeek LLM.
4. Upgrade the block: RMSNorm, RoPE, SwiGLU, GQA.
5. Replace dense FFN with a small DeepSeekMoE-style routed FFN.
6. Study MoE load balance, auxiliary loss, routing collapse, and specialization.
7. Add an educational MLA-style low-rank KV path for KV-cache experiments.
8. Run SFT, reasoning cold-start SFT, DPO, and rule-based GRPO mini experiments.

This repo intentionally focuses on language models only. It excludes
multimodal, vision, video, OCR, robotics, and tool-use/agent chapters from the
first roadmap.

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
  Z["Code First<br/>Write Dense LM"] --> A["Stage 0<br/>Train Dense LM"]
  A --> B["Stage 1<br/>LR / Batch Sweep"]
  B --> C["Stage 2<br/>RMSNorm + RoPE + SwiGLU + GQA"]
  C --> D["Stage 3<br/>Tiny DeepSeekMoE"]
  D --> E["Stage 4<br/>Educational MLA"]
  E --> F["Stage 5<br/>SFT + Reasoning Cold Start"]
  F --> G["Stage 6<br/>Rule-based GRPO Mini"]
  G --> H["Stage 7<br/>Rejection Sampling + Distillation"]
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
  model/                Dense LM, MoE FFN, educational MLA path
  scripts/              Data prep and generation helpers
  trainer/              Pretrain, SFT, sweep, DPO/GRPO skeletons
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

## First Reading Path

Read these docs in order, or open the full [tutorial index](docs/README.md):

1. [Project Scope](docs/00_project_scope.md)
2. [DeepSeek Paper Map for LM Training](docs/01_deepseek_lm_paper_map.md)
3. [Code First: Build the Initial DeepSeek-Style Dense LM](docs/12_code_first_dense_lm.md)
4. [Stage 0: Dense Baseline](docs/02_stage0_dense_baseline.md)
5. [Stage 1: LR and Batch-Size Search](docs/03_stage1_lr_batch_search.md)
6. [Stage 2: MLP and Attention Upgrades](docs/04_stage2_block_upgrades.md)
7. [Stage 3: Tiny DeepSeekMoE](docs/05_stage3_moe.md)
8. [Stage 4: Educational MLA](docs/06_stage4_mla.md)
9. [Stage 5: SFT and Reasoning Cold Start](docs/07_stage5_sft_cold_start.md)
10. [Stage 6: Rule-Based GRPO Mini](docs/08_stage6_grpo_mini.md)

Chinese tutorial notes:

Open the full [中文教程目录](docs/zh/README.md), or read in this order:

1. [项目范围](docs/zh/00_project_scope.md)
2. [DeepSeek 语言模型论文地图](docs/zh/01_deepseek_lm_paper_map.md)
3. [代码优先：从零写出最初的 DeepSeek-style Dense LM](docs/zh/12_code_first_dense_lm.md)
4. [阶段 0：Dense Baseline](docs/zh/02_stage0_dense_baseline.md)
5. [阶段 1：LR 和 Batch Size 搜索](docs/zh/03_stage1_lr_batch_search.md)
6. [阶段 2：MLP 和 Attention 升级](docs/zh/04_stage2_block_upgrades.md)
7. [阶段 3：Tiny DeepSeekMoE](docs/zh/05_stage3_moe.md)
8. [阶段 4：教学版 MLA](docs/zh/06_stage4_mla.md)
9. [阶段 5：SFT 和 Reasoning Cold Start](docs/zh/07_stage5_sft_cold_start.md)
10. [阶段 6：Rule-Based GRPO Mini](docs/zh/08_stage6_grpo_mini.md)
11. [仓库路线图](docs/zh/09_repository_roadmap.md)
12. [实验报告模板](docs/zh/10_experiment_report_template.md)
13. [MiniMind 风格结构说明](docs/zh/11_minimind_structure_notes.md)
14. [GPU 选择与成本记录](docs/zh/13_gpu_cost_tracking.md)

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

v0.1 contains runnable source files for the dense/MoE/educational-MLA model,
pretraining, generation, and LR/batch sweeps. SFT and GRPO entry points are
present as roadmap placeholders and will be filled after the base training path
is stable.
