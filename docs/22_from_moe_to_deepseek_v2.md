# 22. From DeepSeekMoE to DeepSeek-V2: MLA in Code

DeepSeekMoE sparsifies the FFN, while attention still stores K/V for every previous token. DeepSeek-V2 combines both paths: keep DeepSeekMoE and replace attention with MLA.

- Previous model: [`stage1_deepseek_moe.py`](../model/stages/stage1_deepseek_moe.py)
- New model: [`stage2_deepseek_v2.py`](../model/stages/stage2_deepseek_v2.py)
- Formal implementation: [`CausalSelfAttention`](../model/tinyseek.py)

## Research Card: Establish the Cache Bottleneck First

Before changing code, calculate the previous stage's KV elements per layer and token:

```text
GQA cache = 2 * num_kv_heads * head_dim
this lab    = 2 * 2 * 48 = 192 elements/token/layer
```

That value is multiplied by batch, context length, and layer count. The formula establishes the growth trend, but it does not replace a real decoding profiler.

Test two hypotheses in order:

| Step | Candidate | Question | Currently observable |
| --- | --- | --- | --- |
| B0 | GQA control | previous-stage baseline | `192` theoretical elements; measured PPL `2.009 +/- 0.011` |
| B1 | naive low-rank KV | can low rank preserve LM quality? | measured PPL `2.190 +/- 0.001`; no latent-only cache claim because RoPE remains coupled |
| B2 | educational MLA with decoupled RoPE | does separating content and position reduce cacheable state? | `72` theoretical elements and PPL `2.194 +/- 0.012`; real cached decoding not implemented |

**Decision gate:** multi-seed validation PPL for B1/B2 must not be materially worse than B0, and B2 must materially reduce the theoretical cache ledger. Passing both gates supports only the structural result. Actual memory or throughput claims require cached decoding plus long-context peak-memory, post-prefill throughput, and latency measurements.

**Measured decision:** B2 passes the theoretical cache ledger but fails the quality gate, while B1 shows that low rank itself is already costly at this rank. Retain GQA and sweep latent rank before another MLA attempt. Full runs: [architecture report](../experiments/architecture_lab_runs/report.md).

## 1. The Remaining Bottleneck

Standard GQA stores, per layer and historical token:

```text
2 * num_kv_heads * head_dim
```

The two terms are K and V. Cache grows roughly linearly with context length, batch size, and layer count. GQA lowers the number of KV heads, but aggressive reduction can hurt attention capacity.

## 2. MLA Motivation

DeepSeek-V2 uses low-rank joint KV compression. Instead of caching full K/V, it caches a shorter latent and reconstructs content K/V for attention. It also decouples RoPE position components from the compressed content path.

The paper reports 93.3% less KV cache and 5.76x maximum generation throughput relative to DeepSeek 67B. Those are paper-system results, not TinySeek measurements.

## 3. Educational Boundary

TinySeek keeps low-rank `compressed_kv`, content K/V reconstruction, separate K RoPE, split Q content/position paths, and theoretical cache accounting. It omits fused kernels and a real cross-step latent cache. Training still materializes K/V, so this is `EducationalMLA`, not production MLA.

For old v1 checkpoint compatibility, the unified model keeps a legacy `mla_decoupled_rope=false` low-rank K/V branch; that branch no longer claims latent-only caching. The matched architecture configs explicitly set `mla_decoupled_rope=true` and `qk_rope_head_dim=8`, exercising the same decoupled content/RoPE path taught here.

## 4. Dimensions

```python
head_dim = hidden_size // num_heads
content_head_dim = head_dim - qk_rope_head_dim
```

For `hidden=192`, `heads=4`, and `rope_dim=16`, each head has 32 content dimensions and 16 RoPE dimensions. Concatenating them restores the 48-dimensional attention head.

## 5. Query Path

```python
q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
q_content, q_rope = torch.split(
    q, [self.content_head_dim, self.rope_head_dim], dim=-1
)
q_rope = apply_rope(q_rope, self.rope_cos, self.rope_sin)
q = torch.cat((q_content, q_rope), dim=-1)
```

```text
q_content: [B, H, T, content_dim]
q_rope:    [B, H, T, rope_dim]
q:         [B, H, T, head_dim]
```

Only the positional path is rotated.

## 6. KV Compression Path

```python
compressed_kv = self.kv_down(x)
k_content = self.k_content_up(compressed_kv).view(
    batch, seq_len, self.num_kv_heads, self.content_head_dim
).transpose(1, 2)
v = self.v_up(compressed_kv).view(
    batch, seq_len, self.num_kv_heads, self.head_dim
).transpose(1, 2)
k_content = repeat_kv(k_content, self.kv_repeats)
v = repeat_kv(v, self.kv_repeats)
```

The two `view(...).transpose(1,2)` calls recover `[B,H_kv,T,dim]` from packed
linear outputs. `repeat_kv` then aligns KV heads with query heads. These shape
operations satisfy the attention interface without changing the cached latent.

These projections implement:

$$
c_t=W_{down}x_t\in\mathbb{R}^{R},\qquad
k_t^{content}=W_{K,up}c_t,\qquad v_t=W_{V,up}c_t.
$$

`nn.Linear(D,R)` creates the shared latent and two `nn.Linear(R,...)` layers
reconstruct content K and V. Since $R\ll D$, this is a low-rank path; whether it
preserves quality is an experimental question.

Both content K and V come from one `[B,T,R]` latent. The separate K RoPE component is projected and rotated before concatenation:

Here `R=kv_lora_rank`; the complete shape path is `[B,T,D] -> [B,T,R]`, then
content K `[B,H_kv,T,content_dim]` and V `[B,H_kv,T,head_dim]` after
reconstruction and head reshaping.

```python
k_rope = self.k_rope_proj(x).view(
    batch, seq_len, 1, self.rope_head_dim
).transpose(1, 2)
k_rope = apply_rope(k_rope, self.rope_cos, self.rope_sin)
k_rope = k_rope.expand(batch, self.num_heads, seq_len, self.rope_head_dim)
k = torch.cat((k_content, k_rope), dim=-1)
```

`torch.split(...,dim=-1)` separates content and positional features;
`torch.cat(...,dim=-1)` restores `head_dim`; `expand` shares the positional K
view across heads.

The teaching cache estimate is:

```text
kv_lora_rank + qk_rope_head_dim
```

That is:

$$C_{MLA/token}=R+d_{rope},$$

versus $2H_{kv}d_h$ K/V elements for GQA. `kv_cache_elements_per_token()`
returns this theoretical count; it does not measure CUDA allocation.

## 7. Training Versus Cached Decoding

Training processes a whole sequence and reconstructs full K/V before causal SDPA. A production decoder would cache only the latent and RoPE state across generation steps and use absorbed projections or specialized kernels. Keep three claims separate: training data-path correctness, theoretical cache width, and measured decoding memory/throughput.

## 8. Complete V2 Block

```python
self.attn = EducationalMLA(config)
self.ffn = FineGrainedMoE(config)
```

The same two pre-norm residual branches remain. V2 inherits MoE and upgrades attention; it does not replace the rest of the language model.

## 9. Verification

```bash
python tests/stage_models_test.py
python scripts/inspect_stage_models.py
```

Verify logits shape, matching Q/K head dimensions, `rank + rope_dim` cache accounting, and the continued presence of MoE in every V2 block.

## 10. Matched Experiment

First isolate low-rank projection:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_control.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_kv.json --data data/tinystories.jsonl --hourly_rate 2.18
```

This pair measures the quality cost of low-rank attention. Because `mla_decoupled_rope=false`, it must not be reported as a latent-only cache result.

Then test educational MLA with decoupled RoPE:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data data/tinystories.jsonl --hourly_rate 2.18
```

Only `attention_impl` changes. Record validation loss, theoretical KV elements, training memory, and tokens/s, while noting that training memory does not prove latent-cache decoding savings. Results belong in the [architecture experiment plan](../experiments/06_architecture_evolution_plan.md).

If MLA PPL is worse, sweep `mla_latent_size` and report the quality-versus-theoretical-cache Pareto curve instead of selecting one favorable rank. Lower theoretical cache with slower training tokens/s is not contradictory: the teaching forward reconstructs K/V and is not a training-kernel speed experiment.

<!-- tinyseek-nav -->

---

Previous: [Dense to DeepSeekMoE](21_from_dense_to_deepseek_moe.md) | [Tutorial Index](README.md) | Next: [DeepSeek-V2 to DeepSeek-V3](23_from_v2_to_deepseek_v3.md)
