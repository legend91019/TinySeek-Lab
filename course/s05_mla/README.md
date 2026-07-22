# s05 MLA: Compression Is a Hypothesis, Not a Free Upgrade

[中文](README_zh.md) | English | [Course index](../README.md)

## Bottleneck

GQA reduced the theoretical K/V state from `384` to `192` elements per token per layer in our setup, but the cache still grows linearly with sequence length. Can a low-rank latent replace most of the stored K/V?

## Paper Clue

DeepSeek-V2 combines DeepSeekMoE with Multi-head Latent Attention (MLA): jointly compress K/V content and separate the RoPE-related positional path so inference can cache a smaller representation. Paper-scale system results motivate the idea; TinySeek does not claim to reproduce the production kernel.

## Code Diff

Read [`model/stages/stage2_deepseek_v2.py`](../../model/stages/stage2_deepseek_v2.py). The attention sublayer changes, while the residual block and MoE FFN contract remain stable:

```text
x [B,T,D] -> kv_down -> latent c [B,T,R]
c -> k_content_up and v_up
x -> k_rope_proj
concat(content K, RoPE K) -> causal attention
```

The exact formulas, `torch.split`, `torch.cat`, reconstruction shapes and cache accounting are explained in [`docs/22_from_moe_to_deepseek_v2.md`](../../docs/22_from_moe_to_deepseek_v2.md).

## Experiment: Controlled Ladder

Do not compare GQA directly with a complicated MLA block and blame any difference on one idea. Use the staged controls:

| Run | What changes | What it isolates |
| --- | --- | --- |
| `v2_attention_control` | GQA baseline | previous quality |
| `v2_low_rank_control` | parameter-matched control | parameter-count confound |
| `v2_low_rank_kv` | naive low-rank K/V | compression cost itself |
| `v2_mla` | low-rank content + decoupled RoPE | educational MLA path |

Configs live in [`configs/architecture_lab/`](../../configs/architecture_lab/). The gate requires both a clear theoretical cache reduction and no stable PPL regression.

## Evidence and Decision

Theoretical KV elements/token fall `192 -> 72`, but PPL changes `2.009 -> 2.194`; naive low-rank K/V already reaches `2.190`. TinySeek therefore **rejects MLA for the promoted small-model branch** and keeps the compression idea as a research branch for latent-rank sweeps.

This is the key experimental lesson: the paper contribution remains important even when one small implementation and budget fail its gate.

![Architecture quality comparison](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

## Missing Evidence

The educational forward pass reconstructs full K/V and has no real cached decode loop. Its `72` figure is a theoretical cache ledger, not measured decode VRAM or latency. A production claim would require cached generation, long-context profiling and suitable kernels.

## Code Exercise

For `kv_lora_rank=64` and `qk_rope_head_dim=8`, derive the `72` cached elements. Then increase rank in a new config and predict the quality/cache trade-off before running it.

## Next

Even on the rejected MLA branch, two V3 questions can be isolated: routing balance without auxiliary loss, and extra future-token supervision. Continue to [s06 V3 routing + MTP](../s06_v3_routing_mtp/README.md).

<!-- tinyseek-nav -->

Previous: [s04 DeepSeekMoE](../s04_deepseek_moe/README.md) | [Course index](../README.md) | Next: [s06 V3 routing + MTP](../s06_v3_routing_mtp/README.md)
