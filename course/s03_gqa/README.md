# s03 GQA: Reduce K/V State Before Inventing MLA

[中文](README_zh.md) | English | [Course index](../README.md)

## Bottleneck

During autoregressive decoding, every layer keeps historical K/V. MHA stores `2 * num_heads * head_dim` elements per token; GQA shares K/V heads and can reduce that state.

## Hypothesis

If query heads keep their capacity while K/V heads are grouped, KV elements can fall without a meaningful PPL regression. This is the cleanest first attention ablation because the block and training recipe stay fixed.

## Code Diff

Compare `num_kv_heads` in [`configs/architecture_lab/dense_mha.json`](../../configs/architecture_lab/dense_mha.json) and [`dense_gqa.json`](../../configs/architecture_lab/dense_gqa.json). Then read `repeat_kv` and `Attention.forward` in [`model/stages/stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py).

```text
MHA: Q [B,H,T,d], K/V [B,H,T,d]
GQA: Q [B,H,T,d], K/V [B,H_kv,T,d] -> repeat_kv -> [B,H,T,d]
```

The formula and PyTorch shape mapping are in [`docs/24_math_to_pytorch.md`](../../docs/24_math_to_pytorch.md).

## Experiment Card

| Control | Candidate |
| --- | --- |
| `dense_mha` | `dense_gqa` |
| Same | data hash, 2,048,000 tokens, 3 seeds, optimizer and validation |
| Metrics | PPL, tokens/s, peak VRAM, theoretical KV elements/token |
| Gate | halve KV elements without a stable PPL regression |

## Evidence

The 4090 multi-seed report measures PPL `2.017 -> 2.006` and theoretical KV elements/token `384 -> 192`. GQA passes this local gate, so it becomes the dense control for the MoE experiments. The result is a small-model measurement, not a universal theorem about GQA.

![Architecture PPL](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

## Code Exercise

Add an assertion that `num_heads % num_kv_heads == 0`. Create a one-token example and verify that the K/V head repeat factor is exactly `num_heads // num_kv_heads`.

## Next

Attention is cheaper, but the dense FFN still activates the same parameters for every token. That capacity/compute tension motivates [s04 DeepSeekMoE](../s04_deepseek_moe/README.md).

<!-- tinyseek-nav -->

Previous: [s02 Training recipe](../s02_training_recipe/README.md) | [Course index](../README.md) | Next: [s04 DeepSeekMoE](../s04_deepseek_moe/README.md)
