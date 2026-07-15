# 18. 复现正式 GPU 实验

这份手册已经在 2026 年 7 月的 RTX 4090 正式实验中完整执行过。它现在是从新机器复现实验的路径：准备数据、运行可恢复实验、评测，并重新生成已归档的双语报告。

实测结果见[架构实验报告](../../experiments/architecture_lab_runs/report_zh.md)与[正式训练/后训练报告](../../experiments/gpu_completion_runs/report_zh.md)。

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

先做 16 个架构配置 x 3 个 seed 的公平研究套件。脚本会为每个 seed 生成独立配置，已经存在完整 cost summary 的 run 会自动跳过：

```bash
python scripts/run_architecture_lab.py \
  --data data/tinystories.jsonl \
  --seeds 42,43,44 \
  --hourly_rate 2.18
```

生成多 seed 表格和 SVG：

```bash
python scripts/generate_architecture_report.py
```

再跑完整能力与成本套件：50M-token Dense 35M、30M-token Dense 115M、四点 LR/batch、30M-token MoE、20M-token base、结构化 cold-start SFT、300-step GRPO 和三阶段 mini eval：

```bash
python scripts/run_gpu_completion.py \
  --data data/tinystories.jsonl \
  --hourly_rate 2.18
```

两个 runner 默认支持 **run 级恢复**：已有完整 cost summary 的 run 会跳过；中途打断的单个 run 会从 step 0 重跑，当前 trainer 还不支持从中间 optimizer checkpoint 续训。先检查将执行的命令而不训练：

```bash
python scripts/run_architecture_lab.py --data data/tinystories.jsonl --dry_run
python scripts/run_gpu_completion.py --data data/tinystories.jsonl --dry_run
```

## 跑完后必须执行

```bash
python scripts/generate_v1_report_assets.py --run_dir experiments/v1_4090_plan
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
python scripts/generate_architecture_report.py
python scripts/generate_gpu_completion_report.py
```

然后检查：

- `experiments/v1_4090_plan/auto_summary_zh.md`
- `experiments/v1_4090_plan/figures/*.svg`
- `experiments/moe_routing_report.md`
- `experiments/architecture_lab_runs/report_zh.md`
- `experiments/architecture_lab_runs/figures/*.svg`
- `experiments/gpu_completion_runs/report_zh.md`
- `experiments/gpu_completion_runs/figures/*.svg`
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
| PPL / Add / Copy / QA / Format / Reasoning | `eval_*.json` |
| MoE expert load | `moe_routing_report.md` |
| loss 曲线 | `*_history.jsonl` |

## 已得到的复现产物

- `48/48` 架构 runs：16 个配置 x 42/43/44 三个 seed。
- `11` 组正式预训练、sweep、SFT 和 GRPO 成本记录。
- 两套新增实验账本合计 `2.4664 h` 的训练/后训练进程，按 `2.18 元/h` 为 `5.3768 元`；不含数据准备、独立评测、报告和空闲租卡。
- 均值/标准差、逐 run CSV、数据 SHA256、原始 history 和 SVG 图表。
- 按预注册门槛写出的决策，包括失败与不确定的升级。

当前教程 release 不再缺必须补跑的 GPU 数据。以后再开卡属于新的研究扩展，例如 latent rank、bias update rate、更大 token budget 或更严格的 GRPO reward。

<!-- tinyseek-nav -->

---

上一篇: [MiniMind 质量说明](17_minimind_quality_notes.md) | [教程目录](README.md)
