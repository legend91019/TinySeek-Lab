# DeepSeekMath：R1 之前的数学推理训练路线

中文 | [English](README.md) | [下一篇：DeepSeek-R1](../05-deepseek-r1/README_zh.md) | [推理训练线](../README_zh.md)

论文：DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（arXiv:2402.03300）。它不是新 Transformer 架构，而是 R1 的重要训练前史。

## 1. 论文定位

DeepSeekMath 7B 从 DeepSeek-Coder-Base-v1.5 7B 继续预训练，加入约 120B 数学相关 token，同时保留自然语言和代码数据。论文报告 MATH 竞赛级测试 51.7%，64 次自一致采样达到 60.9%。核心贡献是数据筛选管线和 GRPO。

## 2. 数据与训练管线

论文先从 Common Crawl 等公开网页构造数学语料，使用分类器、启发式规则和去重过滤低质量内容，再进行数学领域继续预训练。随后用数学问题—解答数据做监督微调，最后用可验证奖励强化推理。这说明推理能力不只来自 RL，数据质量和领域继续预训练先提供基础。

## 3. GRPO 与可验证奖励

Group Relative Policy Optimization 对同一问题采样一组答案，用组内相对奖励构造 advantage，省去 PPO 的独立 value model。数学答案可由规则或程序验证器判断正确与否，因此 reward 比开放式偏好更稳定。GRPO 省的是 critic 结构，不是省掉 rollout、采样和奖励设计。

## 4. 论文实验怎么读

重点看三类证据：数据筛选消融、模型规模与继续预训练对比、SFT/GRPO 对比。MATH 51.7% 与 60.9% 是论文规模结果，不是 TinySeek 的可比基准。

## 5. 与 R1 的连接

DeepSeekMath 在数学领域验证了高质量数据、可验证奖励和 GRPO 的组合；R1 将可验证任务扩展到更一般的数学、代码和 STEM 推理，并加入 cold start、多阶段 SFT、混合 reward 和蒸馏。DeepSeekMath 先验证训练机制，R1 再扩大任务范围和 pipeline。

## 6. TinySeek 补充与边界

仓库已有教学 SFT/GRPO 和 5 道留出加法题结果：格式分曾由 0.0 升到 0.6，但答案仍为 0/5，后续宽松 reward 还可能导致退化。这与论文并不矛盾，反而说明可验证奖励、数据规模、rollout 数量和评测泛化必须同时满足。

| 论文主张 | 论文证据 | TinySeek 补充 | 关系 |
| --- | --- | --- | --- |
| 数学数据筛选提升领域模型 | 数据管线与消融 | 无同规模数据筛选 | 仅论文证据 |
| GRPO 可提升数学推理 | GRPO 与 MATH 结果 | 教学 GRPO 无答案提升 | 不可直接比较 |
| 可验证奖励适合数学 RL | 规则/程序验证实验 | 小型格式与答案分离 | 方向提醒一致 |

## 7. 本章结论

DeepSeekMath 是 R1 的训练前史：它先把数据工程、继续预训练和 GRPO 放到数学这一可验证领域中验证。学习 R1 前先读它，能避免把 R1 的效果误解成只换了一个 RL 算法。

## 8. 来源

- [arXiv 页面](https://arxiv.org/abs/2402.03300)
- [论文 PDF](https://arxiv.org/pdf/2402.03300)
- [来源台账](../assets/deepseek-math/SOURCES.md)
+## 深度解读：DeepSeekMath 先验证训练机制，再交给 R1 扩展

### 1. 数据工程是第一项方法贡献

论文不是从一个通用模型直接跑 RL，而是先用分类器、启发式规则、去重和数学质量筛选构造可学习语料，再继续预训练。这个顺序很重要：如果基础模型没有数学符号、题型和解题语言的分布，后续 reward 只能在一个很差的策略空间里搜索。

### 2. 为什么需要继续预训练和 SFT 两步

继续预训练改变模型对数学文本的概率分布，SFT 则把问题—解答格式和可执行推理轨迹变成显式行为。两者分别处理“知识/语言分布”和“输出行为”。因此 MATH 提升不能简单归因于 GRPO；必须看数据管线、继续预训练、SFT 和 RL 的逐步消融。

### 3. GRPO 的适用条件

GRPO 依赖同一 prompt 的多次采样和组内相对 reward。数学任务有规则答案或程序 verifier，reward 的噪声比开放式偏好小；但组内全错时 advantage 几乎没有方向，奖励稀疏仍然是问题。GRPO 省掉 critic 只是降低一种内存/参数开销，并没有解决探索和采样成本。

### 4. 51.7% 与 60.9% 的差别

单次回答准确率和 64-sample self-consistency 衡量的是不同能力：后者允许模型多次试错并投票，不能当作单次推理质量。论文同时报告两者，是为了展示 test-time compute 的价值；教程不能把 60.9% 写成模型一次生成就达到的准确率。

### 5. 与 R1 的桥梁

DeepSeekMath 把可验证奖励和 GRPO 放在数学这一结构清晰的环境中验证；R1 把相似思想扩展到更广的可验证任务，并加入冷启动、混合 SFT、通用 reward 和蒸馏。读者应把它看成一个逐步放大实验，而不是两篇互不相关的论文。

### 6. TinySeek 的反例价值

TinySeek 的格式分提升而答案仍为 0/5，说明 reward 可以教会模型“像在解题”，却没有教会它泛化。这个结果不能反驳 DeepSeekMath，因为数据量、模型能力、验证器和 rollout 完全不同；它能帮助读者识别论文中最容易被忽略的外推条件。
+## 证据地图

| 论文位置 | 实验问题 | 作者结论 | 解读与限制 |
| --- | --- | --- | --- |
| Data construction | 数学网页筛选是否提供有效训练信号 | 精细数据管线提升数据密度 | 质量分类器可能带来分布偏差 |
| Continued pretraining / SFT | 数学知识与解题格式是否建立 | 继续预训练和 SFT 提供基础 | 不能把后续 RL 收益单独归因 |
| GRPO section | 无 critic 的相对奖励是否有效 | GRPO 提升数学推理并节省 value model | rollout 和多次采样仍昂贵 |
| MATH evaluation | 单次与 self-consistency 能力如何区别 | 51.7% 单次、60.9% 64-sample | 后者包含 test-time compute |
