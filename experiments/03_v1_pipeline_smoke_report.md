# v1 Pipeline Smoke Report

Date: 2026-07-08

This report validates the v1 tutorial loop on the AutoDL RTX 4090 image:
pretraining, SFT, rule-based GRPO mini, and GPU cost logging.

## Environment

- GPU: NVIDIA GeForce RTX 4090
- PyTorch: 2.8.0+cu128
- CUDA runtime: 12.8
- Price used for accounting: 2.18 CNY/hour

## Commands

```bash
python tests/smoke_test.py
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl --num_samples 300
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 10 --hourly_rate 2.18

python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl --num_samples 300
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --max_steps 40 --hourly_rate 2.18

python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl --num_samples 40
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --max_steps 5 --hourly_rate 2.18
```

## Results

| Stage | Steps | Peak allocated GB | Metric | Cost CNY |
| --- | ---: | ---: | --- | ---: |
| Pretrain | 10 | 0.21 | val_loss 4.4721 | 0.0003 |
| SFT | 40 | 0.15 | val_loss 3.1767 | 0.0009 |
| GRPO mini | 5 | 0.14 | mean_reward 0.0500 | 0.0020 |

## Takeaways

- The v1 post-training path is executable end to end.
- SFT masks prompt tokens and trains on response tokens.
- GRPO mini produces a non-zero toy reward signal after reward shaping.
- The toy pipeline is a teaching check. Real training still needs a larger
  text corpus and longer token budget.
