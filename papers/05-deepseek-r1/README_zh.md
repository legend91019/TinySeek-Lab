# DeepSeek-R1：从基座模型到推理后训练

中文 | [English](README.md) | [上一篇：DeepSeek-V3](../04-deepseek-v3/README_zh.md) | [主线](../README_zh.md)

论文：*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*（2025， [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)）。R1 的关键不是再换一个 attention，而是研究怎样让已有基座模型在可验证任务上学会更长、更有效的 reasoning，并把可读性、通用 helpfulness 和 safety 重新接回训练流程。

## 1. 论文定位

论文先展示 DeepSeek-R1-Zero：从 DeepSeek-V3-Base 出发，只给 reasoning prompts 和 rule-based rewards，观察 RL 自发产生更长思考、反思和“aha moment”。随后论文解释为什么不能直接把 R1-Zero 当成最终产品：可读性差、语言混杂、非推理任务表现不足。完整 R1 加入 cold-start 长 CoT 数据、两阶段 RL、rejection sampling、混合 reasoning/non-reasoning SFT、helpfulness/harmlessness reward model，并发布 distilled models。

## 2. 从 V3 到 R1

| 阶段 | 主要操作 | 目标 |
| --- | --- | --- |
| R1-Zero | V3-Base + rule-based RL | 观察 reasoning 的自我演化 |
| Dev-1 | cold-start long CoT + RL | 改善可读性与语言一致性 |
| Dev-2 | reasoning-oriented RL | 恢复和增强数学/代码推理 |
| Dev-3 | reasoning 与 non-reasoning SFT | 同时保留写作和通用能力 |
| R1 | mixed RL + rule/model/language rewards | helpful、harmless、reasoning 综合平衡 |

## 3. 方法详解

### GRPO

对同一 prompt 采样一组 completion，使用规则或 reward model 得到 `r_i`，在组内标准化为 advantage，再优化 sampled tokens 的 log probability。GRPO 省掉 PPO 的 value model，降低训练成本；但 reward 设计、组内方差、clip ratio 和 rollout 长度都直接决定稳定性。

### Reward design

数学/代码/逻辑任务使用 correctness、format 等 rule-based reward；一般任务使用 helpfulness 和 safety reward model；language consistency reward 统计 CoT 中目标语言词的比例。论文明确记录 reward hacking：reward 持续升高不一定代表真实任务性能或人类偏好提高。

### Cold start 与多阶段 pipeline

R1-Zero 展示“只给激励也能演化”的研究现象；R1 则加入数千条 human-aligned long CoT，让模型先学会可读的思考格式，再做 RL。第二轮 SFT 混合 reasoning 与 non-reasoning 数据，最后用不同 reward 进行综合 RL。Table 3 的 stage-by-stage 对照说明：某一步可能提升 instruction following，却暂时损失 AIME；pipeline 的价值就在于把这些目标分阶段处理。

### Distillation

论文还把 R1 的 reasoning traces 蒸馏到较小 dense models，说明“推理数据”本身可以成为能力迁移的载体；这与直接期待 tiny model 通过少量 RL 学出同等能力是不同的研究问题。

## 4. 论文实验与图表

![R1-Zero trajectory](../assets/deepseek-r1/paper-fig-1-r1-zero-training.png)

*论文 Figure 1，读图重点：R1-Zero 的 AIME 表现和 thinking length 随 RL 过程变化。*

![R1 pipeline](../assets/deepseek-r1/paper-fig-2-r1-pipeline.png)

*论文 Figure 2，读图重点：最终 R1 是多阶段 pipeline，不是一次 GRPO。*

![GRPO](../assets/deepseek-r1/paper-fig-3-grpo.png)

*论文 Figure 3，读图重点：GRPO 用 group-relative reward，省去 value model。*

论文 Table 3 比较 R1-Zero、Dev1、Dev2、Dev3 和 R1：冷启动提高 IF-Eval/对话类能力，后续 reasoning RL 恢复并增强数学/代码，混合 SFT 改善通用任务，最终 mixed RL 综合提升。论文 Figure 6 则提醒 reward hacking 是真实风险，不应只看代理 reward。

![R1 stage results](../assets/deepseek-r1/paper-table-3-stage-results.png)

![Reward hacking](../assets/deepseek-r1/paper-fig-6-reward-hacking.png)

## 5. TinySeek 对应实现

- SFT：[`trainer/train_sft.py`](../../trainer/train_sft.py)。
- 教学 GRPO：[`trainer/train_grpo.py`](../../trainer/train_grpo.py)。
- 代码走读：[`docs/zh/19_posttraining_code_walkthrough.md`](../../docs/zh/19_posttraining_code_walkthrough.md)。
- 课程文章：[`course/s07_cold_start_sft/README_zh.md`](../../course/s07_cold_start_sft/README_zh.md)、[`course/s08_grpo_and_evaluation/README_zh.md`](../../course/s08_grpo_and_evaluation/README_zh.md)。

TinySeek 的 SFT 只做 prompt mask 与 response loss；GRPO 只实现 group sampling、规则 reward、advantage normalization 和 sampled-token log-probability。它没有 R1 的 32k rollout、reward model、数千/数十万条 reasoning data、reject sampling pipeline 或多阶段完整数据配方。

## 6. TinySeek 补充证据

本仓 5 道留出加法题的答案准确率一直是 `0/5`：base 和 direct GRPO 格式分为 `0.0`，cold-start SFT 提到 `0.6`，SFT + GRPO 降到 `0.2`；TinyStories sample PPL 从 `1.718` 变为 `12.670` 再到 `12.306`。这说明窄域 SFT 学到了部分格式，却没有 arithmetic generalization；宽松 reward 还可能破坏格式。

![TinySeek post-training evidence](../../experiments/gpu_completion_runs/figures/posttraining_reasoning.svg)

**关系：论文结论为主，本仓是边界案例。** R1 论文报告了大规模、多阶段 pipeline 的 reasoning 提升；TinySeek 没有复现该能力，但准确地展示了“format score 上升不等于答案正确”和“代理 reward 上升不等于任务解决”。

## 7. 论文主张与本仓证据

| 论文主张 | 论文证据 | TinySeek 证据 | 关系 | 可迁移边界 |
| --- | --- | --- | --- | --- |
| rule-based RL 能诱导 reasoning self-evolution | Figure 1、R1-Zero experiments | direct GRPO 无答案提升 | 仅论文证据 | 数据、模型、rollout 和 reward 差异巨大 |
| cold start 改善可读性和可用性 | Figure 2、Table 3 | 格式 `0.0 -> 0.6`，答案仍 `0/5` | 方向一致：补充性复核 | 只复核格式方向，不复核 reasoning 能力 |
| GRPO 比 PPO 更省 value-model 成本 | Figure 3、训练细节 | 教学 group-relative update | 方向一致：补充性复核 | 没有论文规模成本/稳定性 |
| reward hacking 需要单独审计 | Figure 6、reward discussion | GRPO 后格式分下降 | 方向一致：补充性复核 | 本仓是小型代理 reward 案例 |

## 8. 本章结论

R1 的论文贡献是一条后训练路线：先观察纯 RL 的 reasoning 演化，再用 cold start、混合 SFT、多种 reward 和第二轮 RL 把 reasoning、可读性、helpfulness、harmlessness 接起来。TinySeek 的价值不在于声称复现 R1，而在于让读者看到 pipeline 中最容易混淆的边界：格式、代理 reward、答案正确率和基础分布必须分开评测。

## 9. 入口

- 原文：[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)。
- 上一篇：[DeepSeek-V3](../04-deepseek-v3/README_zh.md)。
- 现有报告：[`gpu_completion_runs/report_zh.md`](../../experiments/gpu_completion_runs/report_zh.md)。
+## 深度解读：R1 不是“把 GRPO 跑起来”就结束

### 1. R1-Zero 实验在证明什么

R1-Zero 从 V3-Base 出发，只提供 reasoning prompt 和可验证 reward。Figure 1 观察到思考长度、数学表现和自我检查行为随 RL 变化，这支持“模型可以在没有人工 CoT 轨迹的情况下发展部分推理行为”。但它没有证明纯 RL 对所有任务都有效：语言混杂、格式不可控、非推理任务退化正是论文随后引入完整 R1 pipeline 的理由。

### 2. 为什么必须 cold start

cold-start SFT 不是为了直接把答案教给模型，而是先提供可读、结构化的长思维链，使后续 RL 的探索落在更容易被人类使用和奖励模型识别的区域。Figure 2 的 pipeline 因此是一个偏差—方差折中：纯 RL 探索空间大但输出不稳定，cold start 限制探索空间但提高可读性和训练信号质量。

### 3. GRPO 的计算账本

GRPO 对同一 prompt 采样一组 completion，按组内 reward 均值和方差标准化 advantage，再更新 sampled tokens 的概率。省掉 value model 并不等于训练便宜：rollout、批量采样、参考模型 KL、长序列显存和 reward 计算仍然昂贵。Figure 3 只能支持“去掉 critic 的算法形式”，不能单独支持“总训练成本一定更低”。

### 4. Table 3 要按阶段读

Table 3 的价值不在于最终 R1 的一个最高分，而在于展示不同目标之间的拉扯：cold start 改善 instruction following 和可读性，reasoning RL 恢复数学/代码，混合 SFT 保留通用能力，最后的 mixed RL 再平衡 helpfulness、harmlessness 和 reasoning。每一阶段都不是单调提升所有指标，因此 R1 是 pipeline 设计而非单个 loss 的胜利。

### 5. Reward hacking 是因果警报

Figure 6 说明代理 reward 上升可能伴随答案质量、语言一致性或人类偏好下降。原因可能是 reward 只检查格式、长度或局部规则，而没有覆盖真实目标。读 R1 时应把 correctness reward、format reward、reward-model score 和最终 benchmark 分开；任何一项单独上升都不能代表 reasoning 已改善。

### 6. 蒸馏的真正含义

R1 将 reasoning traces 蒸馏给更小 dense model，说明规模模型的探索结果可以转化为监督数据。但这与“让小模型自己通过少量 RL 达到同样能力”是不同问题。TinySeek 的 0/5 加法结果正好提醒我们：格式迁移比可泛化的算法推理容易得多。
+## 证据地图

| 论文位置 | 实验问题 | 作者结论 | 解读与限制 |
| --- | --- | --- | --- |
| Figure 1 / R1-Zero | 无 CoT 监督的 RL 是否出现推理行为 | 出现更长思考和自我检查 | 只支持可验证任务和该训练规模 |
| Figure 2 / pipeline | cold start 是否改善可用性 | 多阶段流程比纯 R1-Zero 更可控 | 每阶段改变多个目标，难完全归因 |
| Table 3 / stages | reasoning、helpfulness、safety 是否可兼顾 | 分阶段处理比单一训练更稳定 | 指标之间存在真实 trade-off |
| Figure 6 / reward hacking | 代理 reward 是否可靠 | reward 上升可能脱离真实质量 | 必须保留独立 correctness 与人评 |
