# 04. Stage 2: MLP and Attention Upgrades

Goal: upgrade the block before upgrading the whole training pipeline.

> This is now a **component ablation lab**. RMSNorm, RoPE, and SwiGLU already belong to the DeepSeek LLM Dense baseline; they are not a later DeepSeek generation. See the complete [`stage0_deepseek_llm.py`](../model/stages/stage0_deepseek_llm.py).

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

## Beginner MHA/GQA Experiment

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/dense_mha.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/dense_gqa.json --data data/tinystories.jsonl --hourly_rate 2.18
```

Only `num_kv_heads` changes. First calculate per-layer cache elements as `2 * num_kv_heads * head_dim`, then compare validation loss and throughput. The trainer has no cached decoding path, so theoretical cache reduction is not a measured generation speedup.

Continue with the [four-generation architecture map](20_architecture_evolution_overview.md) and the complete Dense-to-MoE code change.

<!-- tinyseek-nav -->

---

Previous: [Stage 1: LR/Batch Search](03_stage1_lr_batch_search.md) | [Tutorial Index](README.md) | Next: [Stage 3: MoE](05_stage3_moe.md)
