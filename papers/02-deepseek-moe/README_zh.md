# DeepSeekMoE：让容量与激活计算解耦

中文 | [English](README.md) | [上一篇：DeepSeek LLM](../01-deepseek-llm/README_zh.md) | [主线](../README_zh.md)

论文：*DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*（2024， [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)）。它回答的问题是：如果 Dense FFN 的每个 token 都使用同一套参数，能否把总容量做大、但只激活少数参数，并让不同 expert 更专门化？

## 1. 论文定位

DeepSeekMoE 的两项核心设计是 **fine-grained expert segmentation** 和 **shared expert isolation**。第一项把一个大 FFN 切成更多更小的 routed experts，每个 token 只 top-k 激活；第二项保留少量 shared experts，处理所有 token 都需要的共同知识，避免 routed experts 被迫重复学习通用能力。论文通过 validation、消融和与 dense/GShard 的比较，主张在相近 activated parameters 下获得更高容量和更好的质量/计算折中。

## 2. 从 Dense 到 MoE

| 层 | Dense | DeepSeekMoE |
| --- | --- | --- |
| FFN 参数 | 一套大 SwiGLU | 多个 routed 小 SwiGLU + shared expert |
| 每 token 计算 | 激活整套 FFN | router 选择 top-k routed，加上 shared |
| 总容量 | 与激活计算一起增长 | 可以存更多 expert |
| 新问题 | 没有路由负载 | expert imbalance、通信、token dropping |

论文的 MoE layer 先计算 router affinity，再选择 top-k。routed expert 输出按 affinity 加权，shared expert 输出直接参与残差路径。这样“共享知识”和“可专门化知识”在结构上分离，而不是仅仅把一个 Dense FFN 复制成若干份。

## 3. 方法详解

### Fine-grained segmentation

设一个传统 FFN 的中间宽度为 `m`，DeepSeekMoE 把它分成 `N_r` 个宽度更小的 routed experts，每个 token 只选 `K_r` 个。固定 activated parameters 时，增加 expert 数意味着每个 expert 更窄、可组合的专家数更多；这提供了更细的 specialization 粒度。TinySeek 的 `FineGrainedMoE` 用一个易读的单设备 dispatch 循环表达这个过程，未实现生产环境的 all-to-all expert parallelism。

### Shared expert isolation

`N_s` 个 shared experts 对所有 token 激活；它们负责通用模式，routed experts 则学习更有区别性的模式。论文 Figure 2 把 shared（绿色）和 routed（蓝色）画成两条路径。这个设计的判断标准不是“expert 越多越好”，而是总参数、activated parameters、路由负载和质量一起看。

### 负载均衡与 token dropping

top-k 路由会产生热门 expert。论文讨论 auxiliary loss、device-limited routing 和 token dropping 等手段：既要避免少数 expert 溢出，也要避免均衡损失干扰主语言模型目标。后来的 V3 会把这条负载均衡路线继续改成 selection bias；因此 MoE 论文是理解 V3 的前置。

## 4. 论文实验与图表

![DeepSeekMoE architecture](../assets/deepseek-moe/paper-fig-2-moe-architecture.png)

*论文 Figure 2，来源见 [`SOURCES.md`](../assets/deepseek-moe/SOURCES.md)。读图重点：shared experts 负责共同知识，routed experts 负责 token-specific specialization。*

论文 validation 实验控制 activated parameters 和训练条件，比较不同 expert 粒度与 shared expert 配置。论文 Figure 3 的消融支持两条经验：fine-grained segmentation 提供更灵活的组合，shared experts 能减少 routed experts 重复学习 common knowledge。论文还把 DeepSeekMoE 16B 与更大 dense/GShard 模型比较，强调“总参数多、每 token 激活少”的效率叙事。

![DeepSeekMoE validation results](../assets/deepseek-moe/paper-table-1-validation-results.png)

![DeepSeekMoE ablation](../assets/deepseek-moe/paper-fig-3-ablation.png)

## 5. TinySeek 对应实现

- 结构：[`model/stages/stage1_deepseek_moe.py`](../../model/stages/stage1_deepseek_moe.py)。
- 课程讲义：[`course/s04_deepseek_moe/README_zh.md`](../../course/s04_deepseek_moe/README_zh.md)。
- 公式和 dispatch：[`docs/zh/21_from_dense_to_deepseek_moe.md`](../../docs/zh/21_from_dense_to_deepseek_moe.md)。
- 配置与报告：[`experiments/architecture_lab_runs/`](../../experiments/architecture_lab_runs/)。

TinySeek 保持 block 输入/输出 `[B,T,D]` 不变，只替换 FFN 子层。它记录 `expert_counts`、load CV、PPL、吞吐和显存；但没有论文规模的 distributed routing、跨节点通信和 capacity factor 工程。

## 6. TinySeek 补充证据

本仓 3-seed 架构套件比较 coarse、fine-grained、shared 和不同 auxiliary-loss 权重。报告中 shared 路线的 PPL 优于 coarse MoE，但速度约慢 35%；因此主线没有制造单一赢家，而是保留质量分支和速度分支。`aux=0.01` 的负载 CV 约为 `0.075`，no-aux 约为 `0.342`，同时 PPL 也略好。

![TinySeek MoE load](../../experiments/architecture_lab_runs/figures/moe_load_cv.svg)

**关系：方向一致，补充性复核。** 论文认为 shared experts 和更细粒度路由有助于质量/容量折中；TinySeek 观察到 shared 路线的质量收益和负载均衡收益，但也清楚看到吞吐代价。规模与数据不同，不能把本仓的 35% 直接迁移到论文系统。

## 7. 论文主张与本仓证据

| 论文主张 | 论文证据 | TinySeek 证据 | 关系 | 可迁移边界 |
| --- | --- | --- | --- | --- |
| fine-grained experts 提高 specialization 粒度 | Figure 2、Figure 3、validation ablation | fine/shared/coarse 对照 | 方向一致：补充性复核 | 本仓 expert 数量和训练预算很小 |
| shared experts 隔离 common knowledge | Figure 2、shared ablation | shared PPL 优于 coarse，但更慢 | 方向一致：补充性复核 | 没有论文规模 capacity/通信条件 |
| MoE 用较少 activated compute 支撑较大容量 | Table 1/2 与论文比较 | 总参数/激活参数/吞吐账本 | 方向一致：补充性复核 | 理论激活量不等于真实分布式成本 |
| 某种 auxiliary loss 在所有设置最好 | 论文设置内有效 | `aux=0.01` 是本仓局部折中 | 证据不足/不可比较 | 不外推为通用超参 |

## 8. 本章结论

DeepSeekMoE 的关键不是“把 Dense FFN 复制很多次”，而是用更细的专家粒度和 shared/routed 分工，把容量、激活计算与 specialization 分离。TinySeek 的补充结果复核了这个思路的两个方向，同时也保留了工程现实：均衡、吞吐和质量必须一起报告。下一篇 DeepSeek-V2 继续处理另一个瓶颈：即使 FFN 变稀疏，attention 的历史 K/V 仍然占用 cache。

## 9. 入口

- 原文：[arXiv:2401.06066](https://arxiv.org/abs/2401.06066)。
- 下一篇：[DeepSeek-V2：MLA](../03-deepseek-v2/README_zh.md)。
- 现有完整报告：[`architecture_lab_runs/report_zh.md`](../../experiments/architecture_lab_runs/report_zh.md)。
