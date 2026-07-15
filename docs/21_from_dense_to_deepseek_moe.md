# 21. From Dense LM to DeepSeekMoE in Complete Code

Prerequisite: [`12_code_first_dense_lm.md`](12_code_first_dense_lm.md). This chapter changes only the FFN sublayer of Stage 0.

- Previous model: [`stage0_deepseek_llm.py`](../model/stages/stage0_deepseek_llm.py)
- New model: [`stage1_deepseek_moe.py`](../model/stages/stage1_deepseek_moe.py)
- Formal implementation: [`MoEFFN`](../model/tinyseek.py)

## Research Card: MoE Is Not an Automatic Upgrade

Separate two questions:

1. **Dense -> sparse MoE:** can parameter capacity be decoupled from per-token activation? Compare total and activated parameters, LM loss, tokens/s, and memory.
2. **ordinary MoE -> DeepSeekMoE:** with matched expert-FFN capacity and active width, do fine-grained segmentation and shared-expert isolation add value?

The second question uses three configurations:

| Candidate | routed x width | shared x width | active width/token | total expert-FFN capacity |
| --- | ---: | ---: | ---: | ---: |
| coarse ordinary MoE | `2 x 4D`, top-1 | `0` | `4D` | `8D` |
| fine-grained MoE | `8 x 1D`, top-4 | `0` | `4D` | `8D` |
| shared isolation | `7 x 1D`, top-3 | `1 x 1D` | `4D` | `8D` |

Use [`moe_coarse.json`](../configs/architecture_lab/moe_coarse.json), [`moe_fine_grained.json`](../configs/architecture_lab/moe_fine_grained.json), and [`moe_shared.json`](../configs/architecture_lab/moe_shared.json). Router parameters differ slightly with expert count, so reports must still list actual total parameters; the table matches the dominant expert-FFN capacity and active width.

**Decision gate:** first rule out routing collapse, then compare multi-seed validation LM loss/PPL. Keep the simpler MoE if fine-grained or shared variants show no stable benefit, or if throughput cost outweighs quality. More experts alone are not evidence of an effective innovation.

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

With `hidden_size=192`, the Stage 0 Dense intermediate width is `768`, while each Stage 1 expert uses `192`. The model stores eight routed experts, activates two per token, and always runs one shared expert. Activated-parameter, throughput, and expert-load measurements are now archived in the [3-seed architecture report](../experiments/architecture_lab_runs/report.md).

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
router_probs = F.softmax(self.router(flat), dim=-1)
top_weights, top_indices = torch.topk(
    router_probs, k=self.config.top_k, dim=-1
)
top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
```

For flattened token $n$:

$$
z_n=W_r x_n,\qquad
p_{n,e}=\frac{\exp z_{n,e}}{\sum_{j=1}^{E}\exp z_{n,j}}.
$$

For $S_n=\mathrm{TopK}(p_n)$, selected weights are renormalized:

$$
\alpha_{n,e}=\frac{p_{n,e}}{\sum_{j\in S_n}p_{n,j}},\qquad e\in S_n.
$$

`nn.Linear(D,E,bias=False)` implements $W_r$. `softmax(...,dim=-1)` normalizes
over experts. `torch.topk` returns values and expert IDs, while the kept
`[B*T,1]` denominator broadcasts over `[B*T,K]`.

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

The loop implements:

$$y_n^{routed}=\sum_{e\in S_n}\alpha_{n,e}E_e(x_n).$$

Equality creates a `[B*T,K]` boolean mask. `nonzero(as_tuple=True)` returns token
rows and top-k slots; advanced indexing selects assigned vectors;
`unsqueeze(-1)` broadcasts scalar route weights over hidden features; `+=`
accumulates selected-expert contributions.

## 6. Shared Experts

Shared experts skip top-k routing and run for every token:

```python
output = routed_output + shared_output
```

They provide a common path while routed experts can spend capacity on differentiated patterns.

For $E_s>0$, the complete output is:

$$y_n=y_n^{routed}+\frac{1}{E_s}\sum_{s=1}^{E_s}E_s^{shared}(x_n).$$

For the supported $E_s=0$ configuration, code leaves `shared_output=0` and the
equation becomes $y_n=y_n^{routed}$ without division.

## 7. Load Balancing

Routing collapse occurs when a few experts receive nearly all tokens. Stage 1 compares average router probability (`importance`) with actual first-choice assignment:

```python
balance_proxy = num_experts * torch.sum(importance * assignment)
aux_loss = weight * balance_proxy
```

The teaching proxy is:

$$
I_e=\frac{1}{N}\sum_n p_{n,e},\quad
A_e=\frac{1}{N}\sum_n\mathbf{1}[\mathrm{top1}(n)=e],\quad
L_{balance}=E\sum_e I_eA_e.
$$

`F.one_hot` constructs assignment indicators, `.mean(dim=0)` estimates $A_e$,
and `torch.sum` reduces over experts. This readable proxy is not a claim to
reproduce every DeepSeek load-balancing objective exactly.

That equation exactly describes the teaching stage model. Formal ablations use
`model/tinyseek.py`, which calls `sequence_load_balance_loss` with per-sequence
importance and all top-k assignments:

$$
I_{b,e}=\frac{1}{T}\sum_t p_{b,t,e},\qquad
A_{b,e}=\frac{1}{TK}\sum_t\sum_{k=1}^{K}
\mathbf{1}[i_{b,t,k}=e],
$$

$$L_{seq}=\frac{E}{B}\sum_b\sum_e I_{b,e}A_{b,e}.$$

`F.one_hot(top_indices,E)` creates `[B*T,K,E]`; summing the top-k axis, viewing
as `[B,T,E]`, averaging over time, and dividing by $K$ produces $A_{b,e}$.
Thus the formal report's auxiliary-loss evidence comes from the sequence-wise
formula, not the global top-1 teaching proxy.

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

First run the DeepSeekMoE innovation chain:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_coarse.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_fine_grained.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_shared.json --data data/tinystories.jsonl --hourly_rate 2.18
```

Auxiliary loss versus selection bias is the later V3 balancing experiment; do not mix it into the DeepSeekMoE structural ablation:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data data/tinystories.jsonl --hourly_rate 2.18
```

See the [architecture experiment plan](../experiments/06_architecture_evolution_plan.md) and completed [3-seed report](../experiments/architecture_lab_runs/report.md). Fine-grained alone misses the gate; shared experts improve PPL over coarse MoE but reduce throughput, so the report keeps separate quality and speed branches.

Write conclusions as observation, decision, and next action. If the fine-grained run has healthy loads, consistently lower PPL, and acceptable throughput, proceed to shared isolation. If only one seed improves, run repeats instead of declaring an upgrade.

<!-- tinyseek-nav -->

---

Previous: [Code First Dense LM](12_code_first_dense_lm.md) | [Tutorial Index](README.md) | Next: [DeepSeekMoE to DeepSeek-V2](22_from_moe_to_deepseek_v2.md)
