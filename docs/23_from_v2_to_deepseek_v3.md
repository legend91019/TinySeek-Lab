# 23. From DeepSeek-V2 to DeepSeek-V3: Routing Bias and MTP

V3 keeps the V2 backbone of MLA plus DeepSeekMoE. This chapter adds two training-time changes: auxiliary-loss-free load balancing and Multi-Token Prediction (MTP).

- Previous model: [`stage2_deepseek_v2.py`](../model/stages/stage2_deepseek_v2.py)
- New model: [`stage3_deepseek_v3.py`](../model/stages/stage3_deepseek_v3.py)
- Formal model: [`model/tinyseek.py`](../model/tinyseek.py)
- Trainer: [`trainer/train_pretrain.py`](../trainer/train_pretrain.py)

## Research Card: Measure the Trade-off Before Replacing the Mechanism

Split the V3 transition into three experiments instead of jumping directly to the final config:

| Step | Experiment | Observe first | Continue when |
| --- | --- | --- | --- |
| C0 | aux weights `0 / 0.001 / 0.01 / 0.1` | expert-load balance and any simultaneous degradation in main `val_lm_loss` | the load-improvement versus main-task-cost curve is visible, not just one favorable point |
| C1 | best reasonable aux baseline vs selection bias | load spread, PPL, bias oscillation, and tokens/s | bias keeps loads healthy and main LM is no worse than the aux baseline |
| C2 | MTP off vs on | main LM loss/PPL, mini eval, throughput, and memory; never rank by total objective | multi-seed main-task benefit is stable and added compute is acceptable |

The aux sweep uses [`moe_aux_none.json`](../configs/architecture_lab/moe_aux_none.json), [`moe_aux_weak.json`](../configs/architecture_lab/moe_aux_weak.json), [`moe_aux.json`](../configs/architecture_lab/moe_aux.json), and [`moe_aux_strong.json`](../configs/architecture_lab/moe_aux_strong.json).

If C0 shows no main-task cost, TinySeek cannot claim that its small-model experiment proved severe auxiliary-loss interference. It can cite the V3 paper motivation and still run C1 as a mechanism comparison. If C1 or C2 misses its gate, do not promote that tested mechanism; retain the comparator within that matched group. That is the difference between an experiment-driven route and appending paper components by default.

## 1. Why Change V2 Routing

Earlier stages optimize `LM loss + balance auxiliary loss`. A strong balance term can force uniform routing at the expense of assigning a token to its most suitable expert. V3 seeks load control with less direct interference in the language-model gradient.

## 2. Two Scores With Different Jobs

```python
affinity = torch.sigmoid(self.gate(flat))
selection_score = affinity + self.expert_bias.to(affinity.dtype)
top_indices = torch.topk(
    selection_score, k=self.config.top_k, dim=-1
).indices
top_weights = affinity.gather(1, top_indices)
top_weights = top_weights / top_weights.sum(
    dim=-1, keepdim=True
).clamp_min(1e-9)
```

In equations:

$$
a_{n,e}=\sigma((W_gx_n)_e),\qquad s_{n,e}=a_{n,e}+b_e,
$$

$$
S_n=\operatorname{TopK}(s_n),\qquad
\alpha_{n,e}=\frac{a_{n,e}}{\sum_{j\in S_n}a_{n,j}}.
$$

Sigmoid treats expert affinities independently. `topk` selects with adjusted
scores and `gather` retrieves raw affinities for mixture weights. The real code
uses `clamp_min(1e-9)` on the denominator. `expert_bias` is a registered buffer,
not an optimizer parameter.

`selection_score` decides which experts are selected. Raw `affinity` decides their mixture weights. Adding bias to the final weights would change expert contribution, which is not the selection-bias mechanism taught here.

## 3. Bias Update

The forward pass records routed-token counts. After the optimizer step, the trainer calls `model.update_router_bias()`:

```python
target = counts.mean()
direction = torch.sign(target - counts)
self.expert_bias.add_(self.config.router_bias_update_rate * direction)
self.expert_bias.sub_(self.expert_bias.mean())
```

Formally:

$$
b_e\leftarrow b_e+\eta\,\operatorname{sign}(\bar c-c_e),\qquad
b_e\leftarrow b_e-\frac{1}{E}\sum_j b_j.
$$

`torch.bincount(...,minlength=E)` measures $c_e$, `torch.sign` keeps only update
direction, and `torch.no_grad()` keeps in-place buffer updates out of autograd.

Overloaded experts receive lower bias; underloaded experts receive higher bias. Mean centering prevents global drift. The teaching update omits distributed aggregation, group-limited routing, and communication constraints.

## 4. Gradient Learning Versus Load Feedback

```text
forward -> LM/MTP/optional auxiliary loss -> backward
-> optimizer step -> router-bias update -> next batch
```

Model parameters learn through gradients. Selection bias changes through a load-feedback rule. The unified model defaults to `router_balance_strategy="aux"`; V3 behavior requires an explicit `"bias"` configuration.

## 5. MTP Is More Than Another Shift

The main hidden state at position `i` predicts token `i+1`. The first educational MTP module combines that hidden state with the known embedding of token `i+1`, then predicts token `i+2`. A second depth combines its prior hidden state with token `i+2` and predicts token `i+3`.

For the first depth:

$$
h_i^{(1)}=F_{mtp}\left(W_c[\operatorname{RMSNorm}(h_i);
\operatorname{RMSNorm}(e_{i+1})]\right),
$$

$$L_{mtp}^{(1)}=\operatorname{CE}
\left(W_{head}\operatorname{RMSNorm}(h_i^{(1)}),y_{i+2}\right).$$

The semicolon denotes concatenation over the final feature axis.
The final RMSNorm maps to
`self.lm_head(self.norm(previous_hidden[:, :-1]))` and is part of the exact
repository objective.

The module code is:

```python
combined = torch.cat(
    (self.previous_norm(previous_hidden), self.token_norm(future_token_embed)),
    dim=-1,
)
return self.block(self.concat_proj(combined))
```

The outer model aligns targets with:

```python
future_embed = self.embed(input_ids[:, depth + 1 :])
previous_hidden = module(previous_hidden[:, :-1], future_embed)
mtp_logits = self.lm_head(self.norm(previous_hidden[:, :-1]))
mtp_targets = labels[:, depth + 2 :]
```

```text
main hidden at i + embedding(token i+1)
  -> concatenation projection
  -> one Transformer block
  -> shared LM head
  -> token i+2
```

## 6. Shape Alignment

For the first depth:

```text
previous_hidden[:, :-1] : [B, T-1, D]
embed(input_ids[:, 1:]) : [B, T-1, D]
MTP hidden              : [B, T-1, D]
MTP logits[:, :-1]      : [B, T-2, V]
targets[:, 2:]          : [B, T-2]
```

`torch.cat(...,dim=-1)` creates `[B,T-1,2D]`; `nn.Linear(2D,D)` returns to
block width. Multiple depth losses use `torch.stack(...).mean()` so device,
dtype, and gradients stay in the tensor graph.

An off-by-one error can still produce a decreasing loss while training the wrong prediction task, so these slices are part of the test contract.

## 7. Shared Embedding and Head

MTP reuses the main token embedding and tied LM head. This limits extra parameters and keeps all depths in one vocabulary space. TinySeek uses a Dense Transformer block inside each educational MTP module; it preserves sequential prediction, shared embedding/head, and target alignment without claiming every V3 engineering detail.

## 8. Objective and Logging

```python
loss = lm_loss + mtp_loss_weight * mtp_loss
```

Optional MoE `aux_loss` remains separate and is added by the trainer. Logs now retain total train loss, main LM loss, MTP loss, auxiliary loss, expert load, and selection bias. Since MTP-on has an extra objective, compare main LM loss, validation PPL, and mini-eval metrics rather than total loss alone.

## 9. Complete Stage 3 Flow

```text
Embedding
-> N x (Educational MLA + BiasBalancedMoE)
-> main RMSNorm + shared LM head
-> main next-token loss
-> sequential MTP modules during training
-> weighted total loss
```

Without labels, MTP loss is skipped and generation uses the main logits.

## 10. Verification

```bash
python tests/stage_models_test.py
python tests/unified_v3_contract_test.py
```

Tests check bias direction under an artificial `[20,2,2,2]` load, stable main logits shape, exact loss composition, and backward-compatible defaults.

## 11. Matched Experiments

First run the auxiliary-weight sweep to establish the problem shape:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_none.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_weak.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_strong.json --data data/tinystories.jsonl --hourly_rate 2.18
```

Then compare balancing mechanisms:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_no_mtp.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_mtp.json --data data/tinystories.jsonl --hourly_rate 2.18
```

The [3-seed report](../experiments/architecture_lab_runs/report.md) finds that bias routing is worse than aux=0.01 on both PPL and load CV under this setup. On the same educational-MLA+bias branch, MTP's mean PPL is slightly lower, but the difference lies inside seed variation while memory and time increase. That V3-style branch is rejected. This experiment does **not** decide whether MTP helps the promoted GQA+aux recipe; that requires a new matched pair. The local negative result does not contradict V3 at paper scale.

Formal conclusions require multiple seeds. When the budget permits only one seed, label it as a pilot and preserve complete history, expert-load statistics, and the cost ledger for the next pass.

## 12. Why R1 Comes Next

The base-model architecture path now reaches V3. R1 changes how that base model learns instruction following and reasoning through SFT, cold-start data, GRPO, and rejection sampling. It is a training-data and objective evolution, not another attention class.

<!-- tinyseek-nav -->

---

Previous: [DeepSeekMoE to DeepSeek-V2](22_from_moe_to_deepseek_v2.md) | [Tutorial Index](README.md) | Next: [Training Loop](16_training_loop_from_config_to_checkpoint.md)
