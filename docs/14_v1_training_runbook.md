# v1 Training Runbook

This is the practical runbook for a single RTX 4090 session.

## 1. Pretrain

```bash
python scripts/prepare_corpus_data.py --input_dir /path/to/texts --out data/corpus_pretrain.jsonl
python trainer/train_pretrain.py --config configs/medium_dense_35m.json --data data/corpus_pretrain.jsonl --hourly_rate 2.18
```

For online ready-made datasets, use:

```bash
pip install datasets
python scripts/prepare_hf_dataset.py \
  --dataset_name roneneldan/TinyStories \
  --split train \
  --text_field text \
  --max_samples 50000 \
  --out data/tinystories.jsonl
```

Use `configs/medium_dense_115m.json` after the 35M run is stable.

## 2. Architecture Ablations

```bash
python trainer/train_pretrain.py --config configs/tiny_moe.json --data data/corpus_pretrain.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/tiny_mla.json --data data/corpus_pretrain.jsonl --hourly_rate 2.18
```

For MoE, compare total parameters, activated parameters, validation loss, peak
memory, and cost.

## 3. SFT

```bash
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --hourly_rate 2.18
```

The SFT dataset masks prompt tokens and trains only on response tokens.

## 4. Rule-Based GRPO Mini

```bash
python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --hourly_rate 2.18
```

This is an educational GRPO shape: sample a group of answers, score with a
verifiable rule, normalize rewards inside the group, and update the policy with
a reference-model KL proxy.

The toy reward gives full credit for the exact final integer and small shaping
credit for producing a number or an answer-style format. This keeps the demo
from becoming all-zero before the model has learned arithmetic.

## 5. Cost Summary

```bash
python scripts/summarize_costs.py --input_dir out
```

Every report should include GPU hours, peak memory, estimated cost, validation
loss, and the exact config file.

Run mini eval after a checkpoint is available:

```bash
python eval/mini_eval.py --config configs/tiny_sft.json --ckpt out/tiny_sft_last.pt --data data/toy_pretrain.jsonl
```
