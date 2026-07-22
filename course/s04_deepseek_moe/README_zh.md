# s04 DeepSeekMoE：把 FFN 变成研究实验

中文 | [English](README.md) | [课程目录](../README_zh.md)

## 瓶颈

增大 Dense FFN 会同时增大存储容量和每 token 计算量。MoE 想把两件事解耦：存更多 expert，但每个 token 只路由到少数 expert。

## 论文线索

DeepSeekMoE 强调 fine-grained expert segmentation 和 shared expert isolation。论文给出公开动机；TinySeek 实现的是易读的单卡 dispatch 循环，不是 expert parallel 的 all-to-all 基础设施。

## 代码差异

模型外部数据流不变。在 [`model/stages/stage1_deepseek_moe.py`](../../model/stages/stage1_deepseek_moe.py) 中，把 block 的 Dense `SwiGLU` 换成 `FineGrainedMoE`：

```python
router_probs = F.softmax(self.router(flat), dim=-1)
top_weights, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
expert_out = expert(flat[token_indices])
out[token_indices] += expert_out * weight.unsqueeze(-1)
```

公式、shape 和 dispatch 解释见 [`docs/zh/21_from_dense_to_deepseek_moe.md`](../../docs/zh/21_from_dense_to_deepseek_moe.md)。

## 实验卡片

不要把两个问题混在一起：

1. coarse/fine/shared 拓扑：在主要 FFN 容量和激活宽度匹配时，DeepSeekMoE 思路是否有收益？
2. aux-loss/bias routing：拓扑确定后，哪一种负载均衡机制更好？

运行 [`configs/architecture_lab/`](../../configs/architecture_lab/) 中的配置，并阅读 3-seed 报告。

| 问题 | 证据 | 决定 |
| --- | --- | --- |
| coarse -> fine/shared | PPL `2.071/2.081/2.039`；fine/shared 更慢 | fine 单独否决；shared 保留质量分支，coarse 保留速度分支 |
| no aux -> aux | load CV `0.342 -> 0.075`，PPL `2.020 -> 2.009` | 后续受控分支保留 `aux=0.01` |

![MoE 负载均衡](../../experiments/architecture_lab_runs/figures/moe_load_cv.svg)

## 决策边界

总参数、激活参数、router 负载、PPL 和 tokens/s 必须一起报告。“expert 更多”不是升级证据。单个 seed 也不足以晋级一个拓扑。

## 代码练习

当 `B=2,T=128,E=8,K=2` 时，先预测 `router_probs`、`top_indices`、`token_indices` 和最终 `out` 的 shape，再验证 `sum(expert_counts) == B*T*K`。

## 下一章

MoE 减少了 FFN 激活，但 attention 仍携带历史 K/V。[s05 MLA](../s05_mla/README_zh.md) 将验证低秩路径能否进一步减少这部分状态。

<!-- tinyseek-nav -->

上一篇：[s03 GQA](../s03_gqa/README_zh.md) | [课程目录](../README_zh.md) | 下一篇：[s05 MLA](../s05_mla/README_zh.md)
