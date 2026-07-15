# 18. Reproduce the Formal GPU Suite

This runbook was used for the completed July 2026 RTX 4090 suite. It remains the
reproduction path from a fresh machine: prepare data, run resumable experiments,
evaluate, and regenerate the archived bilingual reports.

Measured outputs: [architecture report](../experiments/architecture_lab_runs/report.md)
and [formal training/post-training report](../experiments/gpu_completion_runs/report.md).

## Three Run Paths

| Path | Goal | Hardware | Use |
| --- | --- | --- | --- |
| CPU smoke | Check imports, data, and minimal training startup | CPU | 5-minute reader validation |
| Small GPU teaching run | Run tiny dense/SFT/GRPO mini | 8-12 GB GPU | Teaching demo |
| RTX 4090 formal run | Run 35M/115M/sweep/MoE/MLA/SFT/GRPO/eval | RTX 4090 24 GB | Update formal reports |

## CPU Smoke

```bash
pip install -r requirements.txt
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 5
```

This only proves the code path is intact. It does not produce meaningful model
quality.

## Small GPU Teaching Run

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 50 --hourly_rate 2.18

python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --max_steps 50 --hourly_rate 2.18

python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --max_steps 20 --hourly_rate 2.18
```

## RTX 4090 Formal Run

Prepare data:

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl
```

First run the matched research suite: 16 architecture configs by three seeds. The runner materializes seed-specific configs and automatically skips runs with complete cost summaries:

```bash
python scripts/run_architecture_lab.py \
  --data data/tinystories.jsonl \
  --seeds 42,43,44 \
  --hourly_rate 2.18
```

Generate multi-seed tables and SVG figures:

```bash
python scripts/generate_architecture_report.py
```

Then run the complete capability and cost suite: 50M-token Dense 35M, 30M-token Dense 115M, four LR/batch arms, 30M-token MoE, a 20M-token base, structured cold-start SFT, 300-step GRPO, and three-stage mini eval:

```bash
python scripts/run_gpu_completion.py \
  --data data/tinystories.jsonl \
  --hourly_rate 2.18
```

Both runners support **run-level resume**: a run with a complete cost summary is skipped, while an interrupted individual run restarts from step 0. The trainer does not yet restore an intermediate optimizer checkpoint. Inspect commands without training:

```bash
python scripts/run_architecture_lab.py --data data/tinystories.jsonl --dry_run
python scripts/run_gpu_completion.py --data data/tinystories.jsonl --dry_run
```

## Required Post-Run Commands

```bash
python scripts/generate_v1_report_assets.py --run_dir experiments/v1_4090_plan
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
python scripts/generate_architecture_report.py
python scripts/generate_gpu_completion_report.py
```

Then inspect:

- `experiments/v1_4090_plan/auto_summary.md`
- `experiments/v1_4090_plan/figures/*.svg`
- `experiments/moe_routing_report.md`
- `experiments/architecture_lab_runs/report.md`
- `experiments/architecture_lab_runs/figures/*.svg`
- `experiments/gpu_completion_runs/report.md`
- `experiments/gpu_completion_runs/figures/*.svg`
- `out/*_history.jsonl`
- `out/*_cost_summary.json`

## Fields to Fill in Reports

| Field | Source |
| --- | --- |
| GPU, CUDA, PyTorch | `*_cost_summary.json` |
| GPU hours and cost | `cost_summary.csv` |
| Peak VRAM | `cost_summary.csv` |
| Total params / activated params | `cost_summary.csv` |
| Validation loss | `cost_summary.csv` |
| PPL / Add / Copy / QA / Format / Reasoning | `eval_*.json` |
| MoE expert load | `moe_routing_report.md` |
| Loss curves | `*_history.jsonl` |

## Reproduction Outputs

- `48/48` architecture runs: 16 configurations x seeds 42/43/44.
- `11` formal pretraining, sweep, SFT, and GRPO cost summaries.
- `2.4664` tracked trainer/post-training process hours and `5.3768 CNY` at `2.18 CNY/h`; data preparation, standalone eval, reporting, and idle rental are excluded.
- Mean/std tables, individual-run CSV, data SHA256 manifests, raw histories, and SVG figures.
- Decisions written against preregistered gates, including failed and inconclusive upgrades.

No additional GPU run is required to complete the current tutorial release. A future run would be a new research extension, such as a latent-rank sweep, bias update-rate sweep, larger token budget, or stricter GRPO reward.

<!-- tinyseek-nav -->

---

Previous: [MiniMind Quality Notes](17_minimind_quality_notes.md) | [Tutorial Index](README.md)
