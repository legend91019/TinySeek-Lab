# v1 训练执行手册

这是一张 RTX 4090 上的实操路线。

## 0. 一条命令总控

租卡前，先在本地预览完整付费训练计划：

```bash
python scripts/run_4090_v1.py
```

这只会打印完整计划，不会开始训练。如果想在租卡前保存命令清单，运行：

```bash
python scripts/run_4090_v1.py --write_manifest
```

这会写出 `experiments/v1_4090_plan/COMMANDS.md`。到 RTX 4090 机器上以后再运行：

```bash
python scripts/run_4090_v1.py --execute --hourly_rate 2.18
```

总控脚本会准备线上数据集，训练 dense baseline，跑真实数据 LR/batch sweep，跑 MoE 和 MLA 对照，执行 tiny SFT/GRPO 教学闭环，运行 mini eval，并生成成本汇总。

post-training 路线会刻意使用 tiny base checkpoint `out/v1_tiny_base_last.pt`，因为 `configs/tiny_sft.json` 和 `configs/tiny_grpo.json` 必须和 checkpoint 的模型形状一致。35M/115M 路线用于预训练和架构实验。

## 1. 预训练

```bash
python scripts/prepare_corpus_data.py --input_dir /path/to/texts --out data/corpus_pretrain.jsonl
python trainer/train_pretrain.py --config configs/medium_dense_35m.json --data data/corpus_pretrain.jsonl --hourly_rate 2.18
```

如果用线上成品数据集：

```bash
pip install datasets
python scripts/prepare_hf_dataset.py \
  --dataset_name roneneldan/TinyStories \
  --split train \
  --text_field text \
  --max_samples 50000 \
  --out data/tinystories.jsonl
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

toy reward 对最终整数完全正确给满分；如果模型至少输出了数字或 answer/final 这类答案格式，会给少量 shaping 分数。这样可以避免模型还没学会算术时 GRPO demo 全是 0 reward。

## 5. 成本汇总

```bash
python scripts/summarize_costs.py --input_dir out
```

每份报告都应该写清楚 GPU 小时数、峰值显存、估算成本、验证 loss 和配置文件。

有 checkpoint 后可以跑 mini eval：

```bash
python eval/mini_eval.py --config configs/tiny_sft.json --ckpt out/tiny_sft_last.pt --data data/toy_pretrain.jsonl
```

<!-- tinyseek-nav -->

---

上一篇: [GPU 成本记录](13_gpu_cost_tracking.md) | [教程目录](README.md) | 下一篇: [MiniMind 质量说明](17_minimind_quality_notes.md)
