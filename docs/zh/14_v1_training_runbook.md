# v1 训练执行手册

这是一张 RTX 4090 上的实操路线。

## 1. 预训练

```bash
python scripts/prepare_corpus_data.py --input_dir /path/to/texts --out data/corpus_pretrain.jsonl
python trainer/train_pretrain.py --config configs/medium_dense_35m.json --data data/corpus_pretrain.jsonl --hourly_rate 2.18
```

35M 跑稳以后，再切到 `configs/medium_dense_115m.json`。

## 2. 架构消融

```bash
python trainer/train_pretrain.py --config configs/tiny_moe.json --data data/corpus_pretrain.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/tiny_mla.json --data data/corpus_pretrain.jsonl --hourly_rate 2.18
```

MoE 报告里要同时比较总参数量、激活参数量、验证 loss、峰值显存和成本。

## 3. SFT

```bash
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --hourly_rate 2.18
```

SFT 数据集会 mask prompt token，只在 response token 上训练。

## 4. Rule-Based GRPO Mini

```bash
python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --hourly_rate 2.18
```

这里实现的是教学版 GRPO 形状：同一个 prompt 采样一组答案，用可验证规则打分，组内归一化 reward，再用 reference-model KL proxy 约束 policy 更新。

## 5. 成本汇总

```bash
python scripts/summarize_costs.py --input_dir out
```

每份报告都应该写清楚 GPU 小时数、峰值显存、估算成本、验证 loss 和配置文件。
