# 18. 上卡前最终 Checklist

这一章的目标是把仓库推进到“只差 GPU 结果”的状态。不开卡时应该完成代码、文档、
报告模板、图表生成器和命令清单；开卡后只做训练、评测、生成报告、回填结论。

## 三档运行路径

| 路径 | 目标 | 需要硬件 | 预计用途 |
| --- | --- | --- | --- |
| CPU smoke | 检查代码能导入、数据能读、最小训练能启动 | CPU | 新读者 5 分钟验证 |
| 小 GPU 教学 run | 跑 tiny dense/SFT/GRPO mini | 8-12 GB GPU | 教学演示 |
| 4090 正式 run | 跑 35M/115M/sweep/MoE/MLA/SFT/GRPO/eval | RTX 4090 24 GB | 更新正式实验报告 |

## CPU Smoke

```bash
pip install -r requirements.txt
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 5
```

这一步只证明代码链路没断，不产生有意义模型质量。

## 小 GPU 教学 Run

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 50 --hourly_rate 2.18

python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --max_steps 50 --hourly_rate 2.18

python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --max_steps 20 --hourly_rate 2.18
```

## 4090 正式 Run

数据准备：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl
```

完整执行：

```bash
python scripts/run_4090_v1.py --execute --skip_data_prepare --hourly_rate 2.18
```

如果要先小步验证：

```bash
python scripts/run_4090_v1.py --stages dense,sweep,moe --skip_data_prepare --sweep_tokens 10000 --write_manifest
```

## 跑完后必须执行

```bash
python scripts/generate_v1_report_assets.py --run_dir experiments/v1_4090_plan
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
```

然后检查：

- `experiments/v1_4090_plan/auto_summary_zh.md`
- `experiments/v1_4090_plan/figures/*.svg`
- `experiments/moe_routing_report.md`
- `out/*_history.jsonl`
- `out/*_cost_summary.json`

## 需要回填到报告的字段

| 字段 | 来源 |
| --- | --- |
| GPU 型号、CUDA、PyTorch | `*_cost_summary.json` |
| GPU 小时、费用 | `cost_summary.csv` |
| 峰值显存 | `cost_summary.csv` |
| total params / activated params | `cost_summary.csv` |
| validation loss | `cost_summary.csv` |
| PPL / Add / Copy / QA / Format | `eval_*.json` |
| MoE expert load | `moe_routing_report.md` |
| loss 曲线 | `*_history.jsonl` |

## 当前不开卡已经完成的部分

- 训练、SFT、GRPO、eval 入口。
- history 和 cost summary 记录。
- v1 自动报告和 SVG 图表生成器。
- MoE routing 报告生成器。
- 实验报告中心。
- 中英双语教程和章节导航。
- README 成果入口。

## 仍必须靠 GPU 填的部分

- 更长 35M dense baseline 的真实 loss 曲线。
- 新 mini eval 中 Copy / QA 的真实结果。
- MoE expert-load 图表的真实数据。
- 加强 cold-start SFT 后的 GRPO 对比结果。
- 最终实验报告中的新结论。

<!-- tinyseek-nav -->

---

上一篇: [MiniMind 质量说明](17_minimind_quality_notes.md) | [教程目录](README.md)
