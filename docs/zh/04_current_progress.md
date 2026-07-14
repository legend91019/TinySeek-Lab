# 04. 当前进度

更新时间：2026-07-14

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
- mini eval：perplexity、加法 exact match、copy accuracy、关键词 QA、格式遵循分。
- GPU 成本记录：GPU 小时、费用、峰值显存、token、粗略 FLOPs。
- 4090 v1 编排脚本：`scripts/run_4090_v1.py`。
- 4090 v1 实测报告和机器可读原始结果。
- 自动报告资产生成器：`scripts/generate_v1_report_assets.py`。
- 自动生成 SVG 图表：PPL、峰值显存、成本、sweep loss、VRAM-vs-PPL。
- MoE routing 报告生成器：`scripts/generate_moe_routing_report.py`。
- 实验报告中心：`experiments/README_zh.md` / `experiments/README.md`。
- 上卡前最终 checklist：`docs/zh/18_gpu_fill_only_checklist.md`。
- SFT masking 和 GRPO objective 代码细读章节。
- DeepSeek LLM、DeepSeekMoE、DeepSeek-V2、DeepSeek-V3 四份完整教学模型。
- Dense -> MoE -> V2 -> V3 的中英双语逐段代码课。
- V3 auxiliary-loss-free selection bias、MTP objective 与统一 trainer 日志。
- 8 份单变量架构实验配置和公平性合同测试。
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

- 更强的算术/推理 cold-start SFT 数据，再接 GRPO，需要上卡跑结果。
- MoE routing histogram 的生成器已完成，但真实 expert-load 图表需要下一次 MoE 训练数据。
- 更长的 35M dense baseline，需要 GPU 时间。
- 新增 Copy / QA mini eval 后，需要用新 checkpoint 重新评测。
- aux/bias routing、MTP off/on、MHA/GQA 和 GQA/MLA 对照已准备好，真实结果待上卡。
- 可选：真实 BPE tokenizer、packed dataset、streaming dataset。
- 可选：CI / GitHub Actions。

## 粗略完成度

| 维度 | 进度 |
| --- | ---: |
| 仓库骨架 | 100% |
| 双语入口和章节 | 98% |
| 代码主线教学 | 96% |
| Dense/MoE/MLA/V3 模型代码 | 95% |
| 预训练和 sweep 链路 | 90% |
| SFT / Cold Start 教学链路 | 80% |
| GRPO / RL 教学链路 | 65% |
| 实验报告和图表 | 88% |
| 精品教程 polish | 93% |

整体判断：作为 GitHub 上可学习、可运行的教程仓库，约 **94%**；作为有完整消融证据的研究复现课程，约 **85%**。不开卡能完成的四代代码、双语讲解、配置、合同测试和报告空表已经基本完成；主要缺口是新 V3 路线的 GPU 数字与动态 PyTorch smoke test。

<!-- tinyseek-nav -->

---

[教程目录](README.md)
