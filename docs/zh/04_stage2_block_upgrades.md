# 04. 阶段 2：MLP 和 Attention 升级

目标：在 Dense baseline 稳定后，逐步升级 Transformer block。

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
