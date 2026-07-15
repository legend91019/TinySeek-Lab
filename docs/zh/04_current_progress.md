# 04. 当前进度

更新时间：2026-07-15

TinySeek-Lab 当前的教程 release 已完成：代码主线、双语讲解、真实语料训练、多 seed 架构消融、后训练对照、自动报告和成本归档都已经落地。

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
- 16 份架构实验配置和公平性合同测试，覆盖普通 MoE -> 细粒度 -> shared、aux 权重 sweep、bias、低秩 KV、MLA 与 MTP。
- 四代教学模型、训练 smoke、V3 合同和全部 16 份架构配置的 CPU 动态 forward 验证。
- 每章末尾的上一篇 / 下一篇 / 目录导航。
- 50,000 条 TinyStories 数据的行数、字节数和 SHA256 清单。
- 48 次架构实测：16 配置 x 3 seeds，包含均值、标准差、逐 run 数据和 expert-load CV。
- 11 组正式训练与后训练：35M/115M/MoE、四点 LR/batch、base、direct GRPO、SFT、SFT -> GRPO。
- 两套自动生成的中英报告、9 张 SVG 图，以及全部 raw cost/history 文件。

## 当前实验结论

- 新增正式实验账本记录 `2.4664 h` 的训练/后训练进程，按 `2.18 元/h` 计 `5.3768 元`；不含数据准备、独立评测、报告和空闲租卡。
- GQA 在理论 KV/token 减半时没有 PPL 退化，通过当前升级门槛。
- shared experts 比 coarse MoE 质量更好但慢约 35%，因此拆成质量分支与吞吐分支。
- aux=0.01 是当前质量与专家负载的最佳折中；bias routing 没有击败它。
- 教学版 MLA 压低理论 KV 状态但明显损害 PPL，当前不升级；MTP 改善落在 seed 波动内，结论不确定。
- SFT 部分学会推理格式，但 5 道留出加法全部失败；GRPO 的代理 reward 上升却让格式退化，是一个真实 reward-misalignment 负结果。

## 可选研究扩展

- 扩大 token budget，检验当前架构结论是否稳定。
- sweep MLA latent rank 与 bias update rate。
- 把教学版 GRPO 升级为带 old-policy ratio、clip 和更严格 reward 的实现。
- 加入真实 BPE tokenizer、packing/streaming，以及更强的小型 benchmark。
- 增加 CUDA 或分布式 expert dispatch，避免 Python MoE 吞吐代表生产实现。
- 可选：CI / GitHub Actions 和文档站。

## 粗略完成度

| 维度 | 进度 |
| --- | ---: |
| 仓库骨架 | 100% |
| 双语入口和章节 | 100% |
| 代码主线教学 | 98% |
| Dense/MoE/MLA/V3 模型代码 | 98% |
| 预训练和 sweep 链路 | 100% |
| SFT / Cold Start 教学链路 | 95% |
| GRPO / RL 教学链路 | 80% |
| 实验报告和图表 | 100% |
| 精品教程 polish | 98% |

整体判断：当前目标范围内的 GitHub 教程 release 已完成。它不是 DeepSeek 规模复现，也不把教学版 MLA/GRPO 写成生产实现；这些边界属于设计约束，不是未完成项。后续工作均是下一版研究扩展。

<!-- tinyseek-nav -->

---

[教程目录](README.md)
