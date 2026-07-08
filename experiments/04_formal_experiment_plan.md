# Formal Experiment Plan

This plan is for the next paid GPU session. Do not run these on toy data.

## Dataset

Use an online dataset through `scripts/prepare_hf_dataset.py` or a MiniMind-style
local JSONL dataset. The repo does not need a heavy data-cleaning pipeline for
v1; the teaching focus is training and architecture.

Recommended options:

| Dataset | Why use it | Example command |
| --- | --- | --- |
| `roneneldan/TinyStories` | Good for tiny language models and visible loss curves | `python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --out data/tinystories.jsonl` |
| `wikitext` / `wikitext-2-raw-v1` | Classic language modeling smoke corpus | `python scripts/prepare_hf_dataset.py --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --split train --text_field text --out data/wikitext2.jsonl` |
| MiniMind-style JSONL | Closer to Chinese small-LM tutorials | use `{"text": "..."}` JSONL directly |

## Experiments

| ID | Goal | Config | Token budget | Metrics |
| --- | --- | --- | ---: | --- |
| E1 | Dense baseline | `configs/medium_dense_35m.json` | 20M-50M tokens | val loss, ppl, cost |
| E2 | Larger dense short run | `configs/medium_dense_115m.json` | 10M-30M tokens | val loss, ppl, memory |
| E3 | LR/batch sweep | `experiments/01_lr_batch_grid.json` adapted to real data | 4 runs x 5M tokens | best LR/batch |
| E4 | MoE activated-param comparison | `configs/medium_moe_activated_35m.json` | 10M-30M tokens | activated params, memory, val loss |
| E5 | MLA/KV comparison | `configs/tiny_mla.json` and dense control | 5M-10M tokens | KV elements/token, memory |
| E6 | SFT cold start | `configs/tiny_sft.json` | 2k-20k examples | format eval, addition eval |
| E7 | GRPO mini | `configs/tiny_grpo.json` | 100-500 updates | reward curve, pass rate |

## Required Report Columns

Every report table should include:

- config file,
- dataset and split,
- tokens or examples,
- model parameters and activated parameters,
- validation loss or reward,
- mini-eval score,
- peak allocated/reserved VRAM,
- GPU hours,
- estimated cost.

## Stop Rules

Stop a run early if:

- validation loss is flat or worse for three evals,
- reward stays zero after SFT has been improved,
- peak reserved VRAM exceeds 90% of the GPU,
- generated samples are broken because the data format is wrong.
