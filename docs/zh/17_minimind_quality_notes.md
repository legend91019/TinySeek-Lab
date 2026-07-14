# 17. 向 MiniMind 学什么

MiniMind 做得好的地方不是单点代码，而是“学习产品”的完成度。读者进入仓库后，
很快能知道：

- 这个项目能训练出什么；
- 大概需要多少钱、多少时间、什么显卡；
- 数据从哪里来；
- 怎么训练、怎么评测、怎么推理；
- 有哪些可视化成果；
- 后续还能部署到哪些生态。

TinySeek-Lab 的主题不同：我们不是单纯做一个小模型，而是沿 DeepSeek 的 LM
研究路线逐步升级 dense、recipe、MoE、MLA、SFT、GRPO。但仓库体验上，确实
应该向 MiniMind 学“成果前置”和“闭环完整”。

## 已经补齐的地方

| 方向 | TinySeek-Lab 当前状态 |
| --- | --- |
| 成果入口 | README 顶部新增 `当前成果`，直接链接实验报告中心和 4090 v1 报告 |
| 实验报告中心 | 新增 `experiments/README_zh.md` 和 `experiments/README.md` |
| 成本叙事 | 记录 GPU 小时、租卡费用、显存、token、粗略 FLOPs |
| 图表 | 自动生成 PPL、显存、成本、sweep loss、VRAM-vs-PPL SVG |
| 完整闭环 | TinyStories -> dense/sweep/MoE/MLA/SFT/GRPO -> eval -> report |
| 代码教学 | 新增 dense LM 从零章节和训练主循环章节 |
| 导航 | 每章末尾有上一篇 / 下一篇 / 目录 |

## 仍然要补的地方

| 方向 | 为什么重要 | 建议做法 |
| --- | --- | --- |
| 更显眼的“一键体验” | 新读者通常先想跑起来，而不是先读论文 | README 增加 5 分钟 CPU smoke、30 分钟小 GPU run、4090 正式 run 三档 |
| 数据推荐 | 教程不应该让读者卡在数据处理上 | 给出现成 HF 数据集组合和命令，先不强迫做复杂 pipeline |
| 更强 mini eval | PPL/加法/格式分太弱 | 增加 arithmetic、copy、QA、instruction format、困惑度分组 |
| MoE 可视化 | DeepSeekMoE 的教学重点是 routing 和负载均衡 | 从 expert-load JSON 生成 routing histogram |
| 后训练成果 | GRPO mini 现在只能讲算法形状 | 先加强 cold-start SFT，再做 GRPO 对比 |
| checkpoints 说明 | 读者想知道哪些 checkpoint 值得保留 | 报告中明确哪些 checkpoint 是 smoke，哪些可作为教程起点 |
| 页面观感 | GitHub 首页需要更像课程封面 | README 加 badge、结果表、目录折叠和关键图 |

## TinySeek 不应该照搬的地方

TinySeek-Lab 的核心不是追求“最短时间训练一个能聊天的小模型”，而是教学
DeepSeek 的 LM 研究路径。因此有些取舍要保留：

- 不把重点放在多模态、Agent、工具调用或部署生态。
- 不为了好看的 demo 牺牲研究变量的清晰度。
- 不把 toy GRPO 包装成严肃 RL 结果。
- 不让复杂数据工程抢走模型代码和训练 recipe 的学习重点。

## 下一轮精品化优先级

1. README 增加三档快速路径：CPU smoke、小 GPU 教学 run、4090 正式 run。
2. 增加 `reports` 风格成果表：参数量、token、PPL、显存、成本一眼可见。
3. 给 MoE expert-load 生成 histogram。
4. 补一章 SFT masking 逐行讲解。
5. 补一章 GRPO objective 逐行讲解。
6. 加强 arithmetic/cold-start 数据，再跑一次后训练对比。

<!-- tinyseek-nav -->

---

上一篇: [v1 训练执行手册](14_v1_training_runbook.md) | [教程目录](README.md) | 下一篇: [上卡前 Checklist](18_gpu_fill_only_checklist.md)
