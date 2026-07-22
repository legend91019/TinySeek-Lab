# s05 MLA：压缩是假设，不是免费升级

中文 | [English](README.md) | [课程目录](../README_zh.md)

## 瓶颈

在本仓配置中，GQA 把每层每 token 的理论 K/V 状态从 `384` 个元素降到 `192`，但 cache 仍随序列长度线性增长。低秩 latent 能否替代大部分完整 K/V？

## 论文线索

DeepSeek-V2 把 DeepSeekMoE 与 Multi-head Latent Attention（MLA）结合：联合压缩 K/V 内容，并拆开与 RoPE 相关的位置路径，使推理可以缓存更短的表示。论文规模的系统结果提供动机；TinySeek 不声称复现生产 kernel。

## 代码差异

阅读 [`model/stages/stage2_deepseek_v2.py`](../../model/stages/stage2_deepseek_v2.py)。attention 子层改变，但残差 block 和 MoE FFN 的接口保持稳定：

```text
x [B,T,D] -> kv_down -> latent c [B,T,R]
c -> k_content_up 和 v_up
x -> k_rope_proj
拼接 content K 与 RoPE K -> causal attention
```

公式、`torch.split`、`torch.cat`、重建 shape 和 cache 账本见 [`docs/zh/22_from_moe_to_deepseek_v2.md`](../../docs/zh/22_from_moe_to_deepseek_v2.md)。

## 实验：逐层控制变量

不要直接拿 GQA 和复杂 MLA block 对比，再把所有差异归因于一个创新。使用分层对照：

| Run | 改变什么 | 隔离什么因素 |
| --- | --- | --- |
| `v2_attention_control` | GQA 基线 | 上一代质量 |
| `v2_low_rank_control` | 参数匹配对照 | 参数量混杂因素 |
| `v2_low_rank_kv` | 朴素低秩 K/V | 压缩本身的代价 |
| `v2_mla` | 低秩 content + 解耦 RoPE | 教学版 MLA 路径 |

配置在 [`configs/architecture_lab/`](../../configs/architecture_lab/) 中。门槛要求理论 cache 明显下降，同时没有稳定的 PPL 退化。

## 证据与决定

理论 KV elements/token 从 `192 -> 72`，但 PPL 从 `2.009 -> 2.194`；朴素低秩 K/V 已经达到 `2.190`。因此 TinySeek **否决 MLA 进入当前小模型主分支**，只把压缩假设保留为 latent-rank sweep 的研究分支。

这正是实验教学的重点：某次小规模实现没有通过门槛，不会推翻论文贡献，但也不能假装本仓已经复现成功。

![架构质量比较](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

## 缺失证据

教学 forward 会显式重建完整 K/V，而且没有真正的 cached decode loop。`72` 是理论 cache 账本，不是实测 decode 显存或延迟。生产级结论需要 cached generation、长上下文 profiler 和合适 kernel。

## 代码练习

当 `kv_lora_rank=64`、`qk_rope_head_dim=8` 时，手算为什么是 `72` 个缓存元素。然后在新配置中增大 rank，运行前先预测质量与 cache 的权衡。

## 下一章

即使 MLA 分支被否决，仍可隔离两个 V3 问题：不用辅助损失的路由均衡，以及额外未来 token 监督。继续 [s06 V3 路由 + MTP](../s06_v3_routing_mtp/README_zh.md)。

<!-- tinyseek-nav -->

上一篇：[s04 DeepSeekMoE](../s04_deepseek_moe/README_zh.md) | [课程目录](../README_zh.md) | 下一篇：[s06 V3 路由 + MTP](../s06_v3_routing_mtp/README_zh.md)
