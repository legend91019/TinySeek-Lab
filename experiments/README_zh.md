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
| [RTX 4090 v1 正式实验结果](05_4090_v1_results_zh.md) | 已完成 | TinyStories 上完整跑通 dense、sweep、MoE、MLA、SFT、GRPO mini 的结果 |
| [v1 自动汇总表和图表](v1_4090_plan/auto_summary_zh.md) | 已完成 | 所有 v1 run 的 PPL、显存、成本、sweep loss 和图表 |
| [v1 pipeline smoke report](03_v1_pipeline_smoke_report_zh.md) | 已完成 | 预训练 -> SFT -> GRPO mini 链路是否能跑通 |
| [AutoDL 4090 smoke report](02_autodl_4090_smoke_report_zh.md) | 已完成 | 4090 环境、显存、最小训练链路验证 |
| [下一轮正式实验计划](04_formal_experiment_plan_zh.md) | 计划 | 后续长训、MoE 分析、GRPO 加强的实验清单 |
| [上卡前最终 Checklist](../docs/zh/18_gpu_fill_only_checklist.md) | 已完成 | 下一次租卡时只需要执行命令、生成报告、回填结论 |

## v1 一句话结论

```text
TinyStories -> tiny base -> dense 35M/115M -> LR/batch sweep
-> MoE -> MLA -> SFT -> GRPO mini -> mini eval -> 成本和图表
```

- v1 全流程已在 RTX 4090 上跑通。
- 总 GPU 时间约 `0.0867 h`，按 `2.18 元/h` 估算约 `0.19 元`。
- 35M dense 的 5M-token sweep 中，`bs16_lr3e-4` 最好。
- 115M 短训不如 35M 短训，说明 token budget 不足时模型更大不一定更好。
- MoE run 总参数 `235.06M`、激活参数约 `84.06M`，峰值 allocated 显存约 `5.46 GB`。
- 教学版 MLA 能跑通 KV latent 思想，但不是 DeepSeek-V2 生产级 MLA 复刻。
- SFT 能学 toy instruction 格式，但会损害 TinyStories PPL。
- GRPO mini 当前只是算法形状教学，不能当作严肃 RL 性能结果。

## v1 图表入口

![PPL comparison](v1_4090_plan/figures/v1_ppl.svg)

![Peak VRAM comparison](v1_4090_plan/figures/v1_peak_vram.svg)

![GPU cost comparison](v1_4090_plan/figures/v1_cost.svg)

![Sweep comparison](v1_4090_plan/figures/v1_sweep_val_loss.svg)

![VRAM versus PPL](v1_4090_plan/figures/v1_vram_vs_ppl.svg)

## 和 MiniMind 学到的改进方向

MiniMind 的仓库很强的一点，是把“低成本、短时间、完整训练链路、可下载数据、
可评测、可部署”放在读者第一视野里。TinySeek-Lab 接下来也应该按这个方向补齐：

| 维度 | 现在 | 下一步 |
| --- | --- | --- |
| 成果入口 | 有报告，但之前入口不够显眼 | README 顶部放实验报告中心和 v1 结果 |
| 数据 | 用 TinyStories/HF 数据和 toy SFT/GRPO | 增加推荐现成文本数据组合，减少读者处理数据压力 |
| 评测 | mini eval：PPL、加法、格式分 | 增加更稳定的 arithmetic/reasoning mini eval |
| 后训练 | SFT 和 GRPO mini 可跑 | 补强 cold-start SFT，再做 GRPO |
| MoE 分析 | 已记录 expert-load 快照 | 输出 routing histogram 和 expert-load 图表 |
| 成本叙事 | 已记录 GPU 小时和费用 | 首页展示“用多少钱跑到什么程度” |
| 教程深度 | 已有模型和训练主循环讲解 | 继续补逐文件逐段代码讲解 |

## 如何重新生成图表

```bash
python scripts/generate_v1_report_assets.py --run_dir experiments/v1_4090_plan
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
```

它会读取：

- `experiments/v1_4090_plan/cost_summary.csv`
- `experiments/v1_4090_plan/eval_*.json`

然后生成：

- `experiments/v1_4090_plan/auto_summary_zh.md`
- `experiments/v1_4090_plan/auto_summary.md`
- `experiments/v1_4090_plan/figures/*.svg`
- `experiments/moe_routing_report.md`

## 报告写作约定

每个正式实验报告都应该包含：

- `环境`：GPU、CUDA/PyTorch、数据集、单价。
- `命令`：能复现的训练/评测命令。
- `成本`：GPU 小时、估算费用、峰值显存、token、粗略 FLOPs。
- `指标`：validation loss、PPL、任务分数。
- `图表`：至少包含 loss/PPL、显存、成本。
- `结论`：明确说清楚哪些是有效结论，哪些只是教学演示。
