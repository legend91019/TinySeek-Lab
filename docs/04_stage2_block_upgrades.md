# 04. Stage 2: MLP and Attention Upgrades

Goal: upgrade the block before upgrading the whole training pipeline.

## Components

- RMSNorm: stable pre-norm normalization.
- RoPE: rotary positional encoding.
- SwiGLU: gated MLP used by many modern LMs.
- GQA: fewer KV heads than query heads.

## Experiments

1. FFN vs SwiGLU.
2. MHA vs GQA.
3. Context length 128 -> 256 -> 512.

Metrics:

- Validation loss.
- Training speed.
- KV-cache elements per token.

## TinySeek Code

The current model has:

- `CausalSelfAttention`
- `DenseFFN`
- `RMSNorm`
- `SwiGLU`

GQA is controlled by `num_kv_heads`.
