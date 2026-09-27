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

