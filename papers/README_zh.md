# TinySeek 论文主线

这里是 TinySeek-Lab 的唯一推荐阅读入口：沿着 DeepSeek 的五篇核心论文，理解它们为什么从一种结构走向下一种结构，以及论文用什么实验验证这些选择。

每篇文章都把四件事放在一起：论文问题与方法、论文自己的图表和结论、TinySeek 的对应实现、仓库已有的小规模补充证据。本仓不新增训练或消融实验；TinySeek 结果不能替代论文规模证据。

## 阅读路线

1. [DeepSeek LLM：从训练配方和 Scaling Laws 建立基线](01-deepseek-llm/README_zh.md)
2. [DeepSeekMoE：让参数容量与每 token 计算解耦](02-deepseek-moe/README_zh.md)
3. [DeepSeek-V2：用 MLA 压缩 KV cache](03-deepseek-v2/README_zh.md)
4. [DeepSeek-V3：无辅助损失负载均衡与 MTP](04-deepseek-v3/README_zh.md)
5. [DeepSeek-R1：从基座模型到推理后训练](05-deepseek-r1/README_zh.md)

English: [paper route](README.md)

## 证据标记

- **论文原始证据**：来自 DeepSeek 论文的模型、数据和实验。
- **TinySeek 补充证据**：来自本仓已经完成的低成本实验。
- **方向一致：补充性复核**：论文和 TinySeek 在相近指标上观察到相同方向，但规模不可比。
- **方向不一致：不可直接外推**：本仓没有看到论文中的收益，不能据此否定论文。
- **仅论文证据 / 仅 TinySeek 教学观察 / 证据不足或不可比较**：按证据边界使用。

论文图表来源台账位于 [`assets/`](assets/README.md)。旧的 `course/`、`docs/` 和 `experiments/` 保留为历史课程、参考手册和原始证据档案。

