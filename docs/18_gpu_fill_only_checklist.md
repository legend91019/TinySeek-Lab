# 18. Final Checklist Before Renting a GPU

This chapter pushes the repository toward the state where only GPU results are
missing. Before renting a card, code, docs, report templates, figure generators,
and command manifests should already be ready. On the GPU, the job should be:
train, evaluate, generate reports, and fill in conclusions.

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

Execute:

```bash
python scripts/run_4090_v1.py --execute --skip_data_prepare --hourly_rate 2.18
```

Small dry run:

```bash
python scripts/run_4090_v1.py --stages dense,sweep,moe --skip_data_prepare --sweep_tokens 10000 --write_manifest
```

## Required Post-Run Commands

```bash
python scripts/generate_v1_report_assets.py --run_dir experiments/v1_4090_plan
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
```

Then inspect:

- `experiments/v1_4090_plan/auto_summary.md`
- `experiments/v1_4090_plan/figures/*.svg`
- `experiments/moe_routing_report.md`
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
| PPL / Add / Copy / QA / Format | `eval_*.json` |
| MoE expert load | `moe_routing_report.md` |
| Loss curves | `*_history.jsonl` |

## Already Done Without GPU

- Train, SFT, GRPO, and eval entry points.
- History and cost-summary logging.
- v1 auto report and SVG figure generator.
- MoE routing report generator.
- Experiment report hub.
- Bilingual tutorial docs and chapter navigation.
- Front-page result entrance.

## Still Needs GPU Data

- Longer 35M dense baseline loss curves.
- Real Copy / QA scores from the expanded mini eval.
- Real MoE expert-load figures.
- Stronger cold-start SFT -> GRPO comparison.
- New conclusions in the final reports.

<!-- tinyseek-nav -->

---

Previous: [What TinySeek-Lab Learns from MiniMind](17_minimind_quality_notes.md) | [Tutorial Index](README.md)
