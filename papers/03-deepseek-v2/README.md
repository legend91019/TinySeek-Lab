# DeepSeek-V2: compress the inference state with MLA

English | [中文](README_zh.md) | [Previous: DeepSeekMoE](../02-deepseek-moe/README.md) | [Paper route](../README.md)

Paper: *DeepSeek-V2: A Strong Mixture-of-Experts Language Model* (2024, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)). V2 combines sparse FFN computation with Multi-head Latent Attention (MLA) for a smaller KV cache.

## 1. Position

MLA jointly compresses key/value content into a latent representation and keeps a decoupled RoPE path. The question is whether a much smaller cached state can preserve quality and long-context behavior. V2 also includes device-limited routing, token dropping, long-context extension and post-training.

## 2. Transition from MoE

DeepSeekMoE addresses activated FFN compute. MLA addresses historical attention state. MHA stores `2 * n_h * d_h` values per token/layer; GQA reduces KV heads; MLA stores a latent rank plus a small positional path. The paper evaluates this accounting together with quality and efficiency.

## 3. Method

`h_t` is projected down to a KV latent; separate projections reconstruct content K and V. Query content and decoupled RoPE components are combined for attention. The production benefit comes from caching the latent representation rather than reconstructing and storing full K/V for every head. The paper also describes the MoE routing and a community-facing V2-Lite model.

## 4. Paper experiments

![Architecture](../assets/deepseek-v2/paper-fig-2-v2-architecture.png)

![KV cache](../assets/deepseek-v2/paper-table-1-kv-cache.png)

![Benchmarks](../assets/deepseek-v2/paper-table-2-benchmark-results.png)

NIAH, benchmark, training and inference results connect cache compression to model quality and efficiency. The paper's conclusion is a usable quality/efficiency trade-off at its scale.

## 5. TinySeek implementation

See [`stage2_deepseek_v2.py`](../../model/stages/stage2_deepseek_v2.py), [`s05`](../../course/s05_mla/README.md), and [`22_from_moe_to_deepseek_v2.md`](../../docs/22_from_moe_to_deepseek_v2.md). TinySeek explicitly reconstructs K/V and has no fused cached-decode kernel, so theoretical KV/token is not measured decode memory or latency.

## 6. TinySeek supplemental evidence

The existing comparison moves theoretical KV/token from `192` for the GQA control to `72` for teaching MLA, while PPL moves `2.009 -> 2.194`; a naive low-rank control is similarly degraded. TinySeek therefore keeps MLA as a research branch rather than the promoted local branch.

![TinySeek PPL](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

The cache direction is **directionally consistent: supplemental check**; the quality result is **direction different: not directly transferable**, not a refutation of the paper.

## 7. Claim/evidence table

| Paper claim | Paper evidence | TinySeek evidence | Relation | Boundary |
| --- | --- | --- | --- | --- |
| MLA reduces KV cache/token | Table 1 | `192 -> 72` theoretical values | Directionally consistent: supplemental check | Not decode memory/latency |
| Quality survives compression | Table 2, NIAH, efficiency | PPL regression | Direction different: not directly transferable | Scale, data and implementation differ |
| MoE and MLA form one system | Figure 2 | Stage2 interface composition | Directionally consistent: supplemental check | No production system reproduction |

## 8. Conclusion

V2 separates activated compute from cached history. TinySeek makes the cache ledger concrete and shows why compression is not a free upgrade.

## 9. Sources and next paper

Paper: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434). Continue to [DeepSeek-V3](../04-deepseek-v3/README.md).
