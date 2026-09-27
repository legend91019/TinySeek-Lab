# DeepSeek-V4：面向百万 token 上下文的高效智能

中文 | [English](README.md) | [上一篇：DeepSeek-V3.2-Exp](../06-deepseek-v3-2-sparse-attention/README_zh.md) | [主线索引](../README_zh.md)

论文：DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence（arXiv:2606.19348）。这是当前主线中面向 1M token 上下文的预览技术报告。

## 1. 论文定位

V4-Pro 约 1.6T 总参数、49B 激活参数；V4-Flash 约 284B 总参数、13B 激活参数。两者支持 1M token 上下文，预训练超过 32T tokens。报告同时改变注意力、残差连接和优化器：CSA/HCA、mHC、Muon。

## 2. 三个结构变化

### 2.1 CSA/HCA 混合注意力

V4 不把所有历史 token 以同一种精度保留。Compressed Sparse Attention（CSA）负责更细粒度的选择与压缩，Heavily Compressed Attention（HCA）提供更激进的历史摘要；两者配合，使模型在百万 token 时仍能保留局部精确访问和远程信息。MLA 主要压缩 KV 表示，V4 进一步压缩需要参与计算的历史。

### 2.2 mHC

Manifold-Constrained Hyper-Connections 改造残差连接。超大模型中，残差流的混合矩阵如果任意增长，训练稳定性和信号传播会变差；mHC 用受约束的参数化保持残差变换的稳定范围。它是训练稳定性配套设计，不是注意力稀疏化的替代品。

### 2.3 Muon

V4 还报告使用 Muon 优化器以加快收敛并提高稳定性。要把优化器改善与架构改善分开：训练预算、优化器和 kernel 同时变化时，不能把所有收益归因给 CSA/HCA。

## 3. 论文实验与数字

报告摘要给出的关键效率比较是：在 1M context setting 下，V4-Pro 单 token inference FLOPs 约为 V3.2 的 27%，KV cache 约为 10%。这些是论文报告的系统级结果，依赖混合注意力、压缩布局和实现优化，不能用普通 PyTorch attention 复算。

建议按三层阅读 V4 实验：能力表、效率表（FLOPs、KV cache、吞吐）和训练稳定性/收敛曲线。来源台账见 ../assets/deepseek-v4/SOURCES.md。

## 4. TinySeek 边界

仓库已有 V2/V3 的 MLA、MoE 和 MTP 教学实现，但没有 CSA/HCA、mHC、Muon，也没有百万 token 数据或显存环境。因此 TinySeek 只能帮助画出“压缩表示 → 选择历史 → 混合注意力”的概念路线，不能验证 V4 的 27% FLOPs 或 10% KV cache。

## 5. 论文主张与本仓证据

| 论文主张 | 论文证据 | TinySeek 证据 | 边界 |
| --- | --- | --- | --- |
| V4 支持百万 token 上下文 | 论文模型规格与长上下文实验 | 无百万 token run | 仅论文证据 |
| CSA/HCA 大幅降低长上下文成本 | 1M context 效率比较 | 既有 MLA 成本记录 | 问题方向相近，不是同一方法 |
| mHC 与 Muon 改善训练稳定性/收敛 | 论文训练与优化器实验 | 无对应实现 | 仅论文证据 |

## 6. 本章结论

V4 不是单点升级，而是围绕百万 token 的协同设计：注意力减少无效历史计算，mHC 稳定残差流，Muon 改善优化。它把 V3.2 的稀疏注意力过渡推进到上下文结构、训练动力学和系统实现一起设计。

## 7. 来源

- [arXiv 页面](https://arxiv.org/abs/2606.19348)
- [论文 PDF](https://arxiv.org/pdf/2606.19348)
- [Hugging Face 模型集合](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [来源台账](../assets/deepseek-v4/SOURCES.md)
+## 深度解读：V4 是围绕百万上下文的联合优化

### 1. 为什么 V4 需要同时改三处

当上下文达到百万 token，单纯减少 KV cache 还不够：attention 计算、残差流稳定性和优化器收敛都会成为瓶颈。V4 把 CSA/HCA、mHC、Muon 放在同一份技术报告中，隐含的工程判断是这些问题相互耦合，而不是三个可以独立叠加的插件。

### 2. CSA/HCA 的信息分层

CSA 负责较细粒度的稀疏访问，HCA 负责更激进的历史压缩。可以把它理解为“精确局部证据 + 低成本远程摘要”的分层记忆，而不是把所有历史 token 统一降采样。关键实验问题是：远程摘要是否保留了跨段依赖，局部精确访问是否覆盖当前任务真正需要的位置。

### 3. 27% FLOPs 和 10% KV cache 应怎样理解

这是在 1M context setting 下相对于 V3.2 的系统级比例，不是单个 attention 算子的理论复杂度，也不是所有长度下的固定倍数。它依赖压缩布局、索引实现、kernel、batch 和 decode 方式。论文结论支持“百万上下文变得更可行”，但不能把比例直接迁移到普通 GPU 或 TinySeek。

### 4. mHC 的位置

mHC 处理的是残差连接的动力学稳定性。它不减少 attention FLOPs，也不直接提供知识；它的价值应通过训练稳定性、梯度/激活统计、收敛速度和最终质量来判断。若只看最终 benchmark，无法知道收益来自 mHC 还是更长训练、Muon 或数据变化。

### 5. Muon 的归因边界

Muon 改变参数更新的几何性质，可能提高收敛速度或稳定性。但优化器切换通常伴随学习率、批量和训练 schedule 重调；因此 V4 报告的端到端收益不能自动归因给 Muon。严谨的消融应固定架构和数据，比较 Muon 与基线优化器的 token-to-loss 曲线和最终质量。

### 6. TinySeek 的正确迁移方式

TinySeek 不具备百万 token 环境，最有价值的迁移不是伪造 V4 数字，而是分别画三条曲线：上下文长度—attention 成本、压缩率—任务损失、优化器—收敛速度。这样读者能理解 V4 的三个问题，而不会把一个小模型结果误写成规模复现。
+## 证据地图

| 论文位置 | 实验问题 | 作者结论 | 解读与限制 |
| --- | --- | --- | --- |
| CSA/HCA method | 分层压缩能否支撑 1M context | 局部精确访问与远程摘要协同 | 需看长距离失败样本 |
| mHC section | 残差连接是否成为稳定性瓶颈 | 受约束连接改善训练动力学 | 需独立梯度/激活和 optimizer 消融 |
| Muon section | 优化器是否提高收敛 | 报告更快收敛和稳定性 | 需要固定架构、数据和 schedule |
| 1M efficiency comparison | 成本是否相对 V3.2 大幅下降 | 27% FLOPs、10% KV cache | 是系统级比例，不是普适常数 |
