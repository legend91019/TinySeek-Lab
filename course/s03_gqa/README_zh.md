# s03 GQA：先减少 K/V 状态，再讨论 MLA

中文 | [English](README.md) | [课程目录](../README_zh.md)

## 瓶颈

自回归生成时，每层都要保存历史 K/V。MHA 每个 token 保存 `2 * num_heads * head_dim` 个元素；GQA 让 K/V head 分组共享，可以减少这部分状态。

## 假设

保留 query head 的容量、只减少 K/V head，应该可以在不显著损害 PPL 的情况下减少 KV 状态。这是第一个很干净的 attention 消融：block 和训练配方都不变。

## 代码差异

对照 [`configs/architecture_lab/dense_mha.json`](../../configs/architecture_lab/dense_mha.json) 和 [`dense_gqa.json`](../../configs/architecture_lab/dense_gqa.json) 的 `num_kv_heads`，再阅读 [`model/stages/stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py) 中的 `repeat_kv` 和 `Attention.forward`。

```text
MHA: Q [B,H,T,d]，K/V [B,H,T,d]
GQA: Q [B,H,T,d]，K/V [B,H_kv,T,d] -> repeat_kv -> [B,H,T,d]
```

公式和 PyTorch shape 对应见 [`docs/zh/24_math_to_pytorch.md`](../../docs/zh/24_math_to_pytorch.md)。

## 实验卡片

| 对照 | 候选 |
| --- | --- |
| `dense_mha` | `dense_gqa` |
| 不变项 | 数据 hash、2,048,000 tokens、3 个 seed、optimizer 和验证集 |
| 指标 | PPL、tokens/s、峰值显存、理论 KV elements/token |
| 门槛 | KV elements 减半，同时没有稳定的 PPL 退化 |

## 证据

4090 多 seed 报告测得 PPL `2.017 -> 2.006`，理论 KV elements/token `384 -> 192`。GQA 通过本地门槛，因此成为后续 MoE 实验的 Dense 对照。它只是小模型测量，不是关于 GQA 的普遍定理。

![架构 PPL](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

## 代码练习

为 `num_heads % num_kv_heads == 0` 加断言。构造一个单 token 输入，验证 K/V head 的 repeat factor 正好是 `num_heads // num_kv_heads`。

## 下一章

attention 变便宜了，但 Dense FFN 仍让每个 token 激活同一套参数。这个容量与计算量的矛盾引出 [s04 DeepSeekMoE](../s04_deepseek_moe/README_zh.md)。

<!-- tinyseek-nav -->

上一篇：[s02 训练配方](../s02_training_recipe/README_zh.md) | [课程目录](../README_zh.md) | 下一篇：[s04 DeepSeekMoE](../s04_deepseek_moe/README_zh.md)
