# DeepSeek-V3.2-Exp：从 MLA 到稀疏注意力

中文 | [English](README.md) | [上一篇：DeepSeek-V3](../04-deepseek-v3/README_zh.md) | [下一篇：DeepSeek-V4](../07-deepseek-v4/README_zh.md) | [主线索引](../README_zh.md)

论文：DeepSeek-V3.2-Exp（官方技术报告，2025），是 V3.1-Terminus 之后、V4 之前的实验性架构过渡。它的核心问题是：MLA 已经压缩 KV cache，但全量注意力在超长上下文下仍需要对每个历史 token 计算注意力。

## 1. 论文定位

V3.2-Exp 引入 DeepSeek Sparse Attention（DSA）：先用轻量 indexer 为每个 query 选择少量重要历史 token，再只对这些 token 做精确注意力。官方资料将它定位为长上下文训练和推理效率实验，并与 V3.1-Terminus 做能力对齐比较。

它不是再换一个 MoE，而是把优化目标从 KV cache 压缩推进到 attention 计算稀疏化：V2 压缩 K/V 表示，V3 继续使用 MLA 并扩大 MoE，V3.2-Exp 选择 token，V4 再发展为 CSA/HCA 混合注意力。

## 2. 方法详解

### 2.1 Indexer 与选择

对 query 位置 i，indexer 计算历史位置 j 的轻量相关性分数，再保留 top-k 位置。索引器只决定看哪里，精确注意力仍使用主干表示计算最终权重。这样把全量 QK 点积从 O(L²) 变成选择成本加 O(Lk)，其中 k 远小于上下文长度 L。

### 2.2 训练与推理

训练时必须让 indexer 学会与主注意力一致的选择，否则稀疏化会损失能力。推理时，索引结构可以缓存或增量更新，收益主要来自长上下文阶段。DSA 的关键不是任何 token 都可丢弃，而是通过可学习选择尽量保留对当前 query 有用的 token。

### 2.3 为什么需要 V3.1 基线

V3.2-Exp 是实验版本，比较重点是在能力大致对齐时，长上下文吞吐、显存和 token 成本是否下降。官方 README 列出 reasoning、数学、代码、搜索/浏览和软件工程等公共 benchmark；能力表与效率表应分开阅读。

## 3. 论文实验怎么读

官方报告的核心证据应按三层检查：

1. 选择质量：top-k 选择是否覆盖主注意力真正重要的 token。
2. 能力保持：与 V3.1-Terminus 的数学、代码、知识和对话结果是否接近。
3. 效率收益：长上下文下的 FLOPs、KV cache、吞吐和延迟是否改善。

不要把某个 benchmark 的百分点差异直接解释为 DSA 的全部效果；实验版本还包含训练配方、kernel 和推理实现变化。完整实验表见官方 PDF 的模型与 benchmark 章节；来源台账位于 ../assets/deepseek-v3-2/SOURCES.md。

## 4. 与 TinySeek 的关系

仓库已有教学版 MLA 和 dense/GQA 代码：stage2_deepseek_v2.py 展示低秩 KV，stage3_deepseek_v3.py 展示 V3 路由与 MTP，architecture_lab_runs/report_zh.md 记录小模型成本和 PPL。

本仓没有 DSA 的 indexer、top-k 稀疏 kernel 或 V3.2 规模训练，因此不能声称复现论文效率。最接近的教学迁移是先测全量 attention 的随长度增长成本，再实现一个固定 top-k 选择器，并明确标注为算法形状演示，而非论文验证。

## 5. 论文主张与本仓证据

| 论文主张 | 论文证据 | TinySeek 证据 | 关系与边界 |
| --- | --- | --- | --- |
| 稀疏选择可降低长上下文 attention 成本 | DSA 方法与效率实验 | 没有 DSA 实验 | 仅论文证据；缺少 indexer、kernel、长上下文训练规模 |
| 能力可与 V3.1-Terminus 对齐 | 官方 benchmark 对照表 | 既有 V3 教学评测，不是 V3.2 对照 | 不可直接比较；模型、数据和版本不同 |
| MLA 之后仍需稀疏化 attention | V3.2 动机与长上下文实验 | MLA 教学版 PPL/成本记录 | 只有问题方向相近；TinySeek 结果不能证明 DSA 收益 |

## 6. 本章结论

V3.2-Exp 是从压缩状态到选择计算的桥梁：MLA 减少每个 token 要保存的内容，DSA 减少每个 query 真正参与注意力的历史 token。它为 V4 的 CSA/HCA 铺路，但实验版本的效率依赖索引器训练、专用 kernel 和系统实现，不能用小模型的普通 PyTorch attention 直接复刻。

## 7. 来源

- [官方仓库](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- [官方技术报告 PDF](https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/DeepSeek_V3_2.pdf)
- [来源台账](../assets/deepseek-v3-2/SOURCES.md)
