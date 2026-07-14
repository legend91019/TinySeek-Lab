# 21. From Dense LM to DeepSeekMoE in Complete Code

Prerequisite: [`12_code_first_dense_lm.md`](12_code_first_dense_lm.md). This chapter changes only the FFN sublayer of Stage 0.

- Previous model: [`stage0_deepseek_llm.py`](../model/stages/stage0_deepseek_llm.py)
- New model: [`stage1_deepseek_moe.py`](../model/stages/stage1_deepseek_moe.py)
- Formal implementation: [`MoEFFN`](../model/tinyseek.py)

## 1. What the Dense Stage Cannot Separate

A Dense block runs the same SwiGLU FFN for every token. Widening that FFN increases parameter capacity and per-token computation together. MoE separates them: store many expert FFNs, but activate only top-k experts for each token.

## 2. What DeepSeekMoE Adds

Beyond ordinary top-k routing, DeepSeekMoE emphasizes:

- **fine-grained expert segmentation**: split a large intermediate width into smaller experts and provide more combinations;
- **shared-expert isolation**: always-active experts carry common knowledge while routed experts specialize.

TinySeek reproduces this readable single-device data path. It does not reproduce expert parallelism or all-to-all communication.

## 3. New Configuration

```python
@dataclass
class Stage1Config(Stage0Config):
    num_routed_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    expert_ffn_multiplier: float = 1.0
    moe_aux_loss_weight: float = 0.01
```

With `hidden_size=192`, the Stage 0 Dense intermediate width is `768`, while each Stage 1 expert uses `192`. The model stores eight routed experts, activates two per token, and always runs one shared expert. Exact compute still requires activated-parameter and throughput measurements.

## 4. Router Shapes

```text
x:            [B, T, D]
flat:         [B*T, D]
router_probs: [B*T, E]
top_indices:  [B*T, K]
top_weights:  [B*T, K]
```

The code is:

```python
flat = x.reshape(-1, x.size(-1))
router_probs = softmax(router(flat), dim=-1)
top_weights, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
```

A token routed to experts `[5, 2]` with weights `[0.72, 0.28]` runs only those two FFNs and combines their outputs with those weights.

## 5. Dispatch Loop

```python
out = torch.zeros_like(flat)
for expert_id, expert in enumerate(self.routed):
    token_mask = top_indices == expert_id
    token_indices, slots = token_mask.nonzero(as_tuple=True)
    expert_out = expert(flat[token_indices])
    out[token_indices] += expert_out * top_weights[token_indices, slots].unsqueeze(-1)
```

For each expert, the loop gathers assigned tokens, runs only those tokens, and scatters weighted outputs back. It is intentionally readable, not a high-performance MoE kernel.

## 6. Shared Experts

Shared experts skip top-k routing and run for every token:

```python
output = routed_output + shared_output
```

They provide a common path while routed experts can spend capacity on differentiated patterns.

## 7. Load Balancing

Routing collapse occurs when a few experts receive nearly all tokens. Stage 1 compares average router probability (`importance`) with actual first-choice assignment:

```python
balance_proxy = num_experts * torch.sum(importance * assignment)
aux_loss = weight * balance_proxy
```

The trainer optimizes `lm_loss + aux_loss`. Too little weight may not prevent collapse; too much can compete with language modeling. Stage 3 replaces this mechanism with selection bias.

## 8. The Complete Model Still Matters

The main structural edit is:

```diff
- self.ffn = SwiGLU(...)
+ self.ffn = FineGrainedMoE(config)
```

The block now returns its auxiliary loss, and the full model sums that value across layers. Embedding, attention, residual paths, tied head, logits shape, and shifted causal loss remain unchanged.

## 9. Total Versus Activated Parameters

`parameter_count()` includes every expert. `activated_parameter_estimate()` counts only the `top_k / num_experts` routed fraction plus always-active attention, embeddings, head, and shared experts. Real FLOPs and speed also depend on dispatch overhead and hardware utilization.

## 10. Verify and Experiment

```bash
python tests/stage_models_test.py
python scripts/inspect_stage_models.py
```

Important invariants are:

```text
sum(expert_counts) = num_layers * B * T * top_k
activated_params < total_params
logits remain [B, T, V]
```

Then run the matched routing comparison:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data data/tinystories.jsonl --hourly_rate 2.18
```

See the [architecture experiment plan](../experiments/06_architecture_evolution_plan.md). Earlier 4090 MoE results prove that the old pipeline runs; they do not replace the new fine-grained matched comparison.

<!-- tinyseek-nav -->

---

Previous: [Build DeepSeek LLM](12_code_first_dense_lm.md) | [Tutorial Index](README.md) | Next: [MoE to DeepSeek-V2](22_from_moe_to_deepseek_v2.md)
