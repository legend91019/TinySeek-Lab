# 22. From DeepSeekMoE to DeepSeek-V2: MLA in Code

DeepSeekMoE sparsifies the FFN, while attention still stores K/V for every previous token. DeepSeek-V2 combines both paths: keep DeepSeekMoE and replace attention with MLA.

- Previous model: [`stage1_deepseek_moe.py`](../model/stages/stage1_deepseek_moe.py)
- New model: [`stage2_deepseek_v2.py`](../model/stages/stage2_deepseek_v2.py)
- Formal implementation: [`CausalSelfAttention`](../model/tinyseek.py)

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

## 4. Dimensions

```python
head_dim = hidden_size // num_heads
content_head_dim = head_dim - qk_rope_head_dim
```

For `hidden=192`, `heads=4`, and `rope_dim=16`, each head has 32 content dimensions and 16 RoPE dimensions. Concatenating them restores the 48-dimensional attention head.

## 5. Query Path

```python
q = q_proj(x)
q_content, q_rope = split(q)
q_rope = apply_rope(q_rope, cos, sin)
q = cat(q_content, q_rope)
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
k_content = self.k_content_up(compressed_kv)
v = self.v_up(compressed_kv)
```

Both content K and V come from one `[B,T,R]` latent. The separate K RoPE component is projected and rotated before concatenation:

```python
k_rope = apply_rope(self.k_rope_proj(x), cos, sin)
k = torch.cat((k_content, k_rope), dim=-1)
```

The teaching cache estimate is:

```text
kv_lora_rank + qk_rope_head_dim
```

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

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data data/tinystories.jsonl --hourly_rate 2.18
```

Only `attention_impl` changes. Record validation loss, theoretical KV elements, training memory, and tokens/s, while noting that training memory does not prove latent-cache decoding savings. Results belong in the [architecture experiment plan](../experiments/06_architecture_evolution_plan.md).

<!-- tinyseek-nav -->

---

Previous: [Dense to DeepSeekMoE](21_from_dense_to_deepseek_moe.md) | [Tutorial Index](README.md) | Next: [V2 to V3](23_from_v2_to_deepseek_v3.md)
