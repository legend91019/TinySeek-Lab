# TinySeek-Lab 实验报告中心

中文 | [English](README.md)

这里集中放 TinySeek-Lab 已完成和计划中的实验报告。仓库的教学主线是“重走
DeepSeek 的语言模型路线”，所以每个实验都尽量回答四个问题：

- 这次改了什么：模型结构、训练 recipe、数据、后训练目标，还是评测方式？
- 训练成本是多少：GPU 型号、GPU 小时、租卡费用、峰值显存、粗略 FLOPs。
- 指标有什么变化：validation loss、PPL、mini eval、专家负载等。
- 教学结论是什么：这个结果能说明什么，不能说明什么？

## 目前最重要的报告

| 报告 | 状态 | 读它能知道什么 |
| --- | --- | --- |
| [3-seed 架构实验报告](architecture_lab_runs/report_zh.md) | 已完成 | 48 次实测、16 个配置的均值/标准差、PPL、吞吐、显存、专家负载和逐阶段决策 |
| [正式训练与后训练报告](gpu_completion_runs/report_zh.md) | 已完成 | 35M/115M/MoE 长训、LR/batch sweep、direct GRPO、SFT -> GRPO、mini eval 和成本 |
| [RTX 4090 v1 正式实验结果](05_4090_v1_results_zh.md) | 已完成 | TinyStories 上完整跑通 dense、sweep、MoE、MLA、SFT、GRPO mini 的结果 |
| [v1 自动汇总表和图表](v1_4090_plan/auto_summary_zh.md) | 已完成 | 所有 v1 run 的 PPL、显存、成本、sweep loss 和图表 |
| [v1 pipeline smoke report](03_v1_pipeline_smoke_report_zh.md) | 已完成 | 预训练 -> SFT -> GRPO mini 链路是否能跑通 |
| [AutoDL 4090 smoke report](02_autodl_4090_smoke_report_zh.md) | 已完成 | 4090 环境、显存、最小训练链路验证 |
| [正式实验计划](04_formal_experiment_plan_zh.md) | 已执行；E5 被替代 | 长训、MoE 和 GRPO 的预注册计划；MLA 移入匹配的架构套件 |
| [DeepSeek 架构演进公平实验](06_architecture_evolution_plan_zh.md) | 3 seed 已完成 | 普通/细粒度/shared MoE、aux 权重、aux/bias、低秩 KV、GQA/MLA、MTP 的实验驱动决策链 |
| [GPU 复现实验手册](../docs/zh/18_gpu_fill_only_checklist.md) | 已验证 | 从数据准备到报告生成的可恢复命令 |

## 正式实验结论

```text
TinyStories -> tiny base -> dense 35M/115M -> LR/batch sweep
-> MoE -> MLA -> SFT -> GRPO mini -> mini eval -> 成本和图表
```

- 账本合计 `2.4664 GPU h` 的训练/后训练进程和 `5.3768 元`估算费用，不含数据准备、独立评测、报告与空闲租卡时间。
- 4 点 sweep 中 `bs16_lr6e-4` 最低，但这只是本 token budget 下的 recipe，不是 scaling law。
- GQA 通过本地门槛；shared experts 形成质量/吞吐取舍；aux=0.01 是当前最佳负载与质量折中。
- 教学版 MLA、bias routing 未通过质量门槛；MTP 只在这条被拒绝的 V3-style 分支上结论不确定，尚未在选中的 GQA+aux recipe 上测试。
- SFT 学会部分格式，但 5 道留出加法仍为 `0/5`；GRPO 的代理 reward 上升没有改善这 5 题，且格式退化。这是 reward 对齐失败的实测案例。

## 最新图表

![3-seed architecture PPL](architecture_lab_runs/figures/architecture_ppl.svg)

![Architecture throughput](architecture_lab_runs/figures/architecture_throughput.svg)

![Formal GPU cost](gpu_completion_runs/figures/formal_cost.svg)

![Post-training reasoning](gpu_completion_runs/figures/posttraining_reasoning.svg)

## 和 MiniMind 学到的改进方向

MiniMind 的仓库很强的一点，是把“低成本、短时间、完整训练链路、可下载数据、
可评测、可部署”放在读者第一视野里。TinySeek-Lab 接下来也应该按这个方向补齐：

| 维度 | 现在 | 可选扩展 |
| --- | --- | --- |
| 成果入口 | README、报告中心和章节均直达正式结果 | 发布 release 或静态文档站 |
| 数据 | 50,000 条 TinyStories，保存行数、字节数和 SHA256 | 加入 BPE/packed-data 分支 |
| 评测 | PPL、加法、copy、QA、答案格式分离 | 引入更强小型 benchmark |
| 后训练 | direct GRPO 与 SFT -> GRPO 对照已实测 | 严格 reward 与完整 GRPO ratio |
| MoE 分析 | 3 seed expert-load CV 与 SVG 已归档 | CUDA/distributed expert dispatch |
| 成本叙事 | GPU 小时、单价、费用、显存、吞吐已公开 | 增加其他 GPU 的同配置对照 |
| 教程深度 | 四代代码课已用门槛和负结果闭环 | 扩大 token budget 检验结论稳定性 |

## 如何重新生成图表

```bash
python scripts/generate_v1_report_assets.py --run_dir experiments/v1_4090_plan
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
python scripts/generate_architecture_report.py
python scripts/generate_gpu_completion_report.py
```

v1 资产生成器会读取：

- `experiments/v1_4090_plan/cost_summary.csv`
- `experiments/v1_4090_plan/eval_*.json`

然后生成：

- `experiments/v1_4090_plan/auto_summary_zh.md`
- `experiments/v1_4090_plan/auto_summary.md`
- `experiments/v1_4090_plan/figures/*.svg`
- `experiments/moe_routing_report.md`

正式报告生成器读取 `out/architecture_lab/*_cost_summary.json` 与 `out/gpu_completion/*_cost_summary.json`，再写入 `architecture_lab_runs/` 和 `gpu_completion_runs/` 下的表格、原始账本、数据清单、报告与图表。

## 报告写作约定

每个正式实验报告都应该包含：

- `环境`：GPU、CUDA/PyTorch、数据集、单价。
- `命令`：能复现的训练/评测命令。
- `成本`：GPU 小时、估算费用、峰值显存、token、粗略 FLOPs。
- `指标`：validation loss、PPL、任务分数。
- `图表`：至少包含 loss/PPL、显存、成本。
- `结论`：明确说清楚哪些是有效结论，哪些只是教学演示。
