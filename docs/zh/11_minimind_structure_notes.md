# 11. MiniMind 风格结构说明

TinySeek-Lab 借鉴了 MiniMind 清晰的目录分层：

- `model/`：模型结构代码。
- `dataset/`：tokenizer 和 dataset wrapper。
- `trainer/`：每个训练阶段一个入口脚本。
- `scripts/`：数据准备、生成、转换等工具脚本。
- `docs/`：教程章节。
- `configs/`：模型和训练配置。
- `experiments/`：sweep 计划和实验记录模板。

## TinySeek 和 MiniMind 的差异

TinySeek-Lab 更偏研究教程：

- 它首先是训练研究路线，不是模型发布项目。
- v0.x 暂时不放多模态、tool-use、agent 主线。
- 每个章节都明确对应 DeepSeek 论文里的一个研究问题。
- sweep、ablation、失败实验记录是项目的一等公民。

## 设计原则

一个读者打开某个阶段时，应该能快速找到：

1. 这章对应哪篇论文。
2. 这章要验证什么假设。
3. 跑哪个脚本。
4. 改哪个配置。
5. 看哪些指标。

这也是为什么仓库没有一开始就堆太多框架抽象。先让学习路径清楚，再逐步增加工程能力。

<!-- tinyseek-nav -->

---

上一篇: [实验报告模板](10_experiment_report_template.md) | [教程目录](README.md) | 下一篇: [GPU 成本记录](13_gpu_cost_tracking.md)
