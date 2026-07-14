# 23. From DeepSeek-V2 to DeepSeek-V3: Routing Bias and MTP

V3 keeps the V2 backbone of MLA plus DeepSeekMoE. This chapter adds two training-time changes: auxiliary-loss-free load balancing and Multi-Token Prediction (MTP).

- Previous model: [`stage2_deepseek_v2.py`](../model/stages/stage2_deepseek_v2.py)
- New model: [`stage3_deepseek_v3.py`](../model/stages/stage3_deepseek_v3.py)
- Formal model: [`model/tinyseek.py`](../model/tinyseek.py)
- Trainer: [`trainer/train_pretrain.py`](../trainer/train_pretrain.py)

## 1. Why Change V2 Routing

Earlier stages optimize `LM loss + balance auxiliary loss`. A strong balance term can force uniform routing at the expense of assigning a token to its most suitable expert. V3 seeks load control with less direct interference in the language-model gradient.

## 2. Two Scores With Different Jobs

```python
affinity = torch.sigmoid(self.gate(flat))
selection_score = affinity + expert_bias
top_indices = topk(selection_score)
top_weights = affinity.gather(1, top_indices)
```

`selection_score` decides which experts are selected. Raw `affinity` decides their mixture weights. Adding bias to the final weights would change expert contribution, which is not the selection-bias mechanism taught here.

## 3. Bias Update

The forward pass records routed-token counts. After the optimizer step, the trainer calls `model.update_router_bias()`:

```python
target = counts.mean()
direction = sign(target - counts)
expert_bias += update_rate * direction
expert_bias -= expert_bias.mean()
```

Overloaded experts receive lower bias; underloaded experts receive higher bias. Mean centering prevents global drift. The teaching update omits distributed aggregation, group-limited routing, and communication constraints.

## 4. Gradient Learning Versus Load Feedback

```text
forward -> LM/MTP/optional auxiliary loss -> backward
-> optimizer step -> router-bias update -> next batch
```

Model parameters learn through gradients. Selection bias changes through a load-feedback rule. The unified model defaults to `router_balance_strategy="aux"`; V3 behavior requires an explicit `"bias"` configuration.

## 5. MTP Is More Than Another Shift

The main hidden state at position `i` predicts token `i+1`. The first educational MTP module combines that hidden state with the known embedding of token `i+1`, then predicts token `i+2`. A second depth combines its prior hidden state with token `i+2` and predicts token `i+3`.

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

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_no_mtp.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_mtp.json --data data/tinystories.jsonl --hourly_rate 2.18
```

Results remain pending in the [architecture experiment plan](../experiments/06_architecture_evolution_plan.md). Until those cells are measured, the repository can claim implementation and experiment readiness, not a TinySeek performance win.

## 12. Why R1 Comes Next

The base-model architecture path now reaches V3. R1 changes how that base model learns instruction following and reasoning through SFT, cold-start data, GRPO, and rejection sampling. It is a training-data and objective evolution, not another attention class.

<!-- tinyseek-nav -->

---

Previous: [MoE to DeepSeek-V2](22_from_moe_to_deepseek_v2.md) | [Tutorial Index](README.md) | Next: [Training Loop](16_training_loop_from_config_to_checkpoint.md)
