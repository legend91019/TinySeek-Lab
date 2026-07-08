# 04. 当前进度

更新时间：2026-07-08

TinySeek-Lab 现在已经从“原型仓库”推进到“v1 教程闭环可运行”的状态。它已经能在真实文本数据上跑完：

```text
TinyStories -> dense baseline -> LR/batch sweep -> MoE -> MLA
-> SFT -> GRPO mini -> mini eval -> 成本和图表报告
```

## 已完成

- 中英双语 README、教程目录和主要章节。
- 代码优先章节：先写最初的 DeepSeek-style dense decoder-only LM。
- 完整训练主循环讲解：config、JSONL、dataset、trainer、checkpoint、history、eval、report。
- Dense / MoE / educational MLA 三种模型路径。
- byte-level tokenizer 和 JSONL 数据集。
- 预训练、SFT、rule-based GRPO mini 三条训练入口。
- LR / batch size sweep 入口。
- mini eval：perplexity、加法 exact match、格式遵循分。
- GPU 成本记录：GPU 小时、费用、峰值显存、token、粗略 FLOPs。
- 4090 v1 编排脚本：`scripts/run_4090_v1.py`。
- 4090 v1 实测报告和机器可读原始结果。
- 自动报告资产生成器：`scripts/generate_v1_report_assets.py`。
- 自动生成 SVG 图表：PPL、峰值显存、成本、sweep loss、VRAM-vs-PPL。
- 每章末尾的上一篇 / 下一篇 / 目录导航。

## 当前实验结论

- v1 全流程已在 RTX 4090 上跑通。
- 总 GPU 时间约 0.0867 小时，按 2.18 元/小时估算约 0.19 元。
- 35M dense 的 5M-token sweep 中，`bs16_lr3e-4` 最好。
- 115M 短训效果不如 35M 短训，说明 token budget 不足时模型更大不一定更好。
- MoE 235M 总参数、约 84M 激活参数，4090 峰值 allocated 显存约 5.46 GB。
- 教学版 MLA 能跑通 KV latent 思想，但不是 DeepSeek-V2 生产级 MLA 复刻。
- SFT 能学到 toy SFT 格式，但会损害 TinyStories PPL。
- GRPO mini 有非零 reward，但没有解出算术 mini eval；它目前用于讲算法形状。

## 仍需加强

- 更强的算术/推理 cold-start SFT 数据，再接 GRPO。
- MoE routing histogram、expert load 统计和 routing collapse 分析。
- 更长的 35M dense baseline，作为更稳定的教程 checkpoint。
- 更细的逐文件逐段代码讲解，尤其是 SFT masking 和 GRPO objective。
- 可选：真实 BPE tokenizer、packed dataset、streaming dataset。
- 可选：CI / GitHub Actions。

## 粗略完成度

| 维度 | 进度 |
| --- | ---: |
| 仓库骨架 | 100% |
| 双语入口和章节 | 90% |
| 代码主线教学 | 85% |
| Dense/MoE/MLA 模型代码 | 85% |
| 预训练和 sweep 链路 | 90% |
| SFT / Cold Start 教学链路 | 70% |
| GRPO / RL 教学链路 | 55% |
| 实验报告和图表 | 80% |
| 精品教程 polish | 75% |

整体判断：作为 GitHub 上可学习、可运行的 v1 教程仓库，约 **85%-90%**；作为更强研究复现和精品课程，还需要继续加强后训练数据、MoE 分析和更长实验。

<!-- tinyseek-nav -->

---

[教程目录](README.md)
