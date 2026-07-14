# 04. 阶段 2：MLP 和 Attention 升级

目标：在 Dense baseline 稳定后，逐步升级 Transformer block。

> 这一章现在作为**组件消融实验课**。RMSNorm、RoPE、SwiGLU 本来就是 DeepSeek LLM Dense 基线的一部分，不应被误画成 DeepSeek 的下一代产品。完整起点代码见 [`stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py)。

## 组件

这一章关注四个现代 LM 常见组件：

- RMSNorm：常见的 pre-norm 归一化方式。
- RoPE：旋转位置编码。
- SwiGLU：带门控的 MLP。
- GQA：query head 多，key/value head 少，降低 KV cache 成本。

## DeepSeek 对应关系

DeepSeek LLM 使用 RMSNorm、RoPE 和 SwiGLU，大模型上也使用 GQA 来优化推理成本。DeepSeek-V2/V3 进一步走向 MLA，但在理解 MLA 之前，先理解 MHA/GQA 更自然。

## 建议实验

1. Dense FFN vs SwiGLU。
2. MHA vs GQA。
3. context length 从 128 扩到 256、512。

观察指标：

- validation loss。
- tokens/sec。
- 显存占用。
- KV-cache elements per token。

## 代码入口

当前代码中的相关模块：

- `RMSNorm`
- `CausalSelfAttention`
- `DenseFFN`
- `SwiGLU`

`num_kv_heads` 控制 GQA。当 `num_kv_heads < num_heads` 时，就进入 grouped-query attention 的设置。

## 初学者先做 MHA/GQA 单变量实验

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/dense_mha.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/dense_gqa.json --data data/tinystories.jsonl --hourly_rate 2.18
```

这两个配置只改变 `num_kv_heads`。先手算单层每 token 的 KV 元素：

```text
MHA: 2 * num_heads * head_dim
GQA: 2 * num_kv_heads * head_dim
```

再比较 validation loss 和训练吞吐。训练脚本没有 cached decoding，因此这里的理论 KV 减少不能直接写成实际生成加速。

下一步不是再堆组件，而是读[四代架构演进总览](20_architecture_evolution_overview.md)，然后把 Dense FFN 改成完整 DeepSeekMoE。

<!-- tinyseek-nav -->

---

上一篇: [阶段 1：LR/Batch 搜索](03_stage1_lr_batch_search.md) | [教程目录](README.md) | 下一篇: [阶段 3：MoE](05_stage3_moe.md)
