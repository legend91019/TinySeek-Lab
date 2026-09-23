# DeepSeek LLM：先把训练问题问清楚

中文 | [English](README.md) | [论文主线](../README_zh.md)

论文：*DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*（2024， [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)）。这是五篇文章的起点：DeepSeek LLM 不只是“再训练一个 Dense Transformer”，而是先问清楚在给定计算预算下，模型、数据、batch size 和 learning rate 应该怎样一起扩大。

## 1. 论文定位

论文的主张是：开源模型不能只在几个固定参数规模上追 benchmark，还需要研究 scaling behavior，并把 scaling 观察用于 7B 和 67B 模型的设计。论文报告了三条主线：batch size/learning rate 的经验 scaling law；用更精确的 non-embedding FLOPs/token 表示模型规模并研究 model/data 分配；不同数据质量会改变 scaling 曲线。最终的 DeepSeek 67B Chat 在论文报告的中文、英文、代码和数学评测中超过 LLaMA-2 70B，并通过 SFT 与 DPO 改善对话质量。

## 2. 从上一代到这一代

| 问题 | 论文做法 | TinySeek 对应 |
| --- | --- | --- |
| 训练规模怎么分配 | 用 IsoFLOP profile 和 scaling curve 选择模型/数据规模 | 小网格 sweep，不声称拟合普适 scaling law |
| 优化参数怎么选 | 研究 batch size、learning rate 随 compute 的变化 | `s02_training_recipe` 的 4 点 sweep |
| Dense LM 的效率 | Pre-Norm、RMSNorm、SwiGLU、RoPE，67B 使用 GQA | `stage0_deepseek_llm.py` 的教学实现 |
| 数据如何提高有效训练量 | aggressive deduplication、质量过滤、domain remixing、BBPE | toy/TinyStories 数据管线 |

DeepSeek LLM 的架构表给出 7B 为 30 层、32 个 query/KV heads，67B 为 95 层、64 个 query heads 和 8 个 KV heads；67B 通过 GQA 减少推理时的 KV 状态。论文使用约 2T tokens、4096 context，并采用多阶段 learning-rate schedule，而不是只追求一个漂亮的训练曲线。

## 3. 方法详解

### 数据和 tokenizer

论文把数据处理拆成 deduplication、filtering 和 remixing。跨多个 Common Crawl dump 去重比单 dump 去重移除更多重复文档；论文 Table 1 报告 91 个 dump 的 deduplication rate 达 89.8%。tokenizer 使用 byte-level BPE，词表约 100k，再加入 special tokens，训练时词表大小设为 102,400；数字按单个 digit 切分，避免数字组合过于稀疏。

### 模型结构

核心仍是 decoder-only next-token prediction。每个 block 采用 Pre-Norm 和 RMSNorm，FFN 用 SwiGLU，位置编码用 RoPE。FFN 中间维度约为 `8/3 * d_model`，再按实现对齐。论文将 67B 的预算更多放在深度，而不是只扩大 FFN 宽度；这既影响表达能力，也方便 pipeline partitioning。GQA 保留 query head 数量，同时让多组 query 共享 K/V head。

### 训练配方和基础设施

AdamW 使用 `beta1=0.9`、`beta2=0.95`、weight decay `0.1`，gradient clipping 为 `1.0`。multi-step scheduler 在 warmup 后于 80% 和 90% token 位置降到最大值的 31.6% 与 10%；论文 Figure 1 的结论是 multi-step 与 cosine 的最终性能相近，但更方便继续训练。基础设施使用 data/tensor/sequence parallelism、1F1B pipeline、FlashAttention、ZeRO-1、通信与计算重叠、bf16 权重和 fp32 梯度累积。

### Scaling law 的逻辑

论文先把 batch size 和 learning rate 当作 compute 的函数，再用更准确的 `M`（non-embedding FLOPs/token）替代仅用参数量 `N`。在固定 compute 的 IsoFLOP 曲线上寻找最优模型规模，最后拟合性能随 model/data scale 的变化。论文特别提醒：数据质量会改变最优分配，不能把一个数据集上的 scaling law 直接外推到另一个数据集。

## 4. 论文实验与图表

![DeepSeek LLM model specifications](../assets/deepseek-llm/paper-table-2-model-specs.png)

*论文 Table 2，来源见 [`SOURCES.md`](../assets/deepseek-llm/SOURCES.md)。读图重点：架构超参数、context、batch、learning rate 和 2T token 预算是一个联合设计。*

![DeepSeek LLM scaling curves](../assets/deepseek-llm/paper-fig-3-scaling-curves.png)

*论文 Figure 3。读图重点：batch size 与 learning rate 不是孤立的手工设置，而是随着 compute 变化的实验对象。*

论文的实验结论包括：优化 hyperparameter 后才能公平比较 model/data scaling；使用 `M` 比参数量更贴近真实 compute；高质量数据倾向于支持更大的模型；这些拟合能够预测 7B/67B 的表现。论文 Table 5 再把 base model 放到多项 benchmark 中，证明 scaling 研究最终服务于可测的模型质量，而非只服务于曲线拟合。

![DeepSeek LLM main results](../assets/deepseek-llm/paper-table-5-main-results.png)

## 5. TinySeek 对应实现

- 完整 Dense 数据流：[`model/stages/stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py)。
- 训练配方：[`course/s02_training_recipe/README_zh.md`](../../course/s02_training_recipe/README_zh.md)、[`trainer/sweep_pretrain.py`](../../trainer/sweep_pretrain.py)。
- attention 对照：[`course/s03_gqa/README_zh.md`](../../course/s03_gqa/README_zh.md)。
- 公式与张量 shape：[`docs/zh/24_math_to_pytorch.md`](../../docs/zh/24_math_to_pytorch.md)。

TinySeek 把 `input_ids [B,T] -> embedding -> blocks -> RMSNorm -> tied LM head` 做成可读的最小路径；它没有复现 HAI-LLM、ZeRO、FlashAttention 或 2T token。仓库的目标是让读者追踪 shape 和控制变量，而不是伪装成论文训练基础设施。

## 6. TinySeek 补充证据

正式 4090 sweep 在本仓预算下选择 `bs16_lr6e-4`（报告 validation loss `0.6475`）；这与论文“先测 recipe，再比较架构”的方法论方向一致，但不是 DeepSeek scaling law。GQA 多 seed 实验把理论 KV elements/token 从 `384` 降到 `192`，PPL `2.017 -> 2.006`，因此在 TinySeek 的门槛下通过。

![TinySeek architecture PPL](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

**关系：方向一致，补充性复核。** 论文把 GQA 作为大模型推理效率设计；TinySeek 观察到减少 KV 状态且本地 PPL 没有退化。两者数据、规模和实现完全不同，不能把 TinySeek 数字当作论文数字的复现。

## 7. 论文主张与本仓证据

| 论文主张 | 论文证据 | TinySeek 证据 | 关系 | 可迁移边界 |
| --- | --- | --- | --- | --- |
| batch size/LR 应随 compute 研究 | Figure 3、scaling section | 4 点 LR/batch sweep | 方向一致：补充性复核 | 本仓只做局部选择，不拟合 scaling law |
| GQA 可降低推理 KV 状态 | Table 2、架构说明 | `384 -> 192` theoretical KV/token | 方向一致：补充性复核 | 没有论文规模 kernel/latency 证据 |
| 数据质量改变 model/data 最优分配 | 论文不同数据集 scaling 对比 | TinyStories 单一数据源 | 仅论文证据 | 本仓没有对应数据质量实验 |
| DeepSeek LLM 在大规模 benchmark 上超过基线 | Table 5 | 仅有 toy/mini eval | 仅论文证据 | 不能用本仓小评测外推 |

## 8. 本章结论

DeepSeek LLM 首先建立的是一套“把训练 recipe 当作科学问题”的基线：数据处理、优化超参、模型/数据规模和架构选择要一起测量。TinySeek 最适合补充两点直觉：recipe 会影响后续架构比较，GQA 的 KV 压缩方向可以在小模型上观察到。下一篇论文把 Dense FFN 的固定容量/固定激活计算矛盾变成 DeepSeekMoE 的专家路由问题。

## 9. 引用、代码和复现入口

- 原文：[arXiv:2401.02954](https://arxiv.org/abs/2401.02954)。
- 代码：[`stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py)。
- 现有报告：[`architecture_lab_runs/report_zh.md`](../../experiments/architecture_lab_runs/report_zh.md)、[`gpu_completion_runs/report_zh.md`](../../experiments/gpu_completion_runs/report_zh.md)。
- 下一篇：[DeepSeekMoE](../02-deepseek-moe/README_zh.md)。
