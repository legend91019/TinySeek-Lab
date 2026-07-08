# v1 训练链路实测报告

日期：2026-07-08

这份报告验证 TinySeek-Lab v1 教程闭环：预训练、SFT、rule-based GRPO mini，以及 GPU 成本记录。

## 环境

- GPU：NVIDIA GeForce RTX 4090
- PyTorch：2.8.0+cu128
- CUDA runtime：12.8
- 成本单价：2.18 元/小时

## 命令

```bash
python tests/smoke_test.py
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl --num_samples 300
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 10 --hourly_rate 2.18

python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl --num_samples 300
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --max_steps 40 --hourly_rate 2.18

python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl --num_samples 40
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --max_steps 5 --hourly_rate 2.18
```

## 结果

| 阶段 | Steps | 峰值 allocated GB | 指标 | 成本/元 |
| --- | ---: | ---: | --- | ---: |
| Pretrain | 10 | 0.21 | val_loss 4.4721 | 0.0003 |
| SFT | 40 | 0.15 | val_loss 3.1767 | 0.0009 |
| GRPO mini | 5 | 0.14 | mean_reward 0.0500 | 0.0020 |

## 结论

- v1 后训练链路已经可以端到端运行。
- SFT 会 mask prompt token，只训练 response token。
- GRPO mini 在 reward shaping 后能产生非零 toy reward 信号。
- 这条 toy pipeline 是教学检查；真正训练还需要更大的文本语料和更长 token budget。
