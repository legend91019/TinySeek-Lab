# Fair Experiments for DeepSeek Architecture Evolution

[中文](06_architecture_evolution_plan_zh.md) | English

These experiments do not ask whether TinySeek matches DeepSeek capability. They ask a smaller, testable question: with data, token budget, and optimizer held fixed, how does one architecture change affect loss, throughput, memory, and routing?

## CPU Structure Check

```bash
python scripts/inspect_stage_models.py --out out/stage_model_inspection.json
```

The script runs one forward pass for each complete teaching model and reports logits shape, total and activated parameters, loss fields, and theoretical per-layer KV-cache elements per token. This validates code contracts, not model quality.

## Fairness Rules

Each A/B pair keeps the following fixed:

- the same JSONL corpus and train/validation split;
- `max_seq_len=128`, batch size, learning rate, warmup, and seed;
- the same number of steps and therefore the same token budget;
- the same GPU, PyTorch version, and precision;
- every model field except the named treatment variable.

`tests/architecture_lab_contract_test.py` checks the matched configurations automatically.

## Matrix

| ID | Comparison | Only change | Paper motivation | TinySeek status |
| --- | --- | --- | --- | --- |
| A1 | MHA vs GQA | `num_kv_heads` | DeepSeek LLM 67B uses GQA to reduce inference cost | Pending GPU run |
| A2 | auxiliary loss vs bias balance | routing strategy and auxiliary weight | V3 avoids interference from a strong balance loss | Pending GPU run |
| A3 | MTP off vs on | `mtp_depth` | V3 adds future-token objectives | Pending GPU run |
| A4 | GQA control vs educational MLA | `attention_impl` | V2 compresses KV state into a low-rank latent | Pending GPU run |

## Commands

Use the same corpus path for every command, for example `data/tinystories.jsonl`:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/dense_mha.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/dense_gqa.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_no_mtp.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_mtp.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data "$DATA" --hourly_rate 2.18
```

## Result Table

`val loss` means token-weighted `val_lm_loss`, and PPL is calculated from that main language-model loss only. Record `val_objective`, `val_mtp_loss`, and `val_aux_loss` separately; do not mix them into PPL or use them directly to rank A/B arms.

| Run | val loss | PPL | tokens/s | peak VRAM | GPU h | cost | architecture metric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `arch_dense_mha` | Pending | Pending | Pending | Pending | Pending | Pending | KV elements/token |
| `arch_dense_gqa` | Pending | Pending | Pending | Pending | Pending | Pending | KV elements/token |
| `arch_moe_aux` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, aux loss |
| `arch_moe_bias` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, selection bias |
| `arch_v3_no_mtp` | Pending | Pending | Pending | Pending | Pending | Pending | LM loss |
| `arch_v3_mtp` | Pending | Pending | Pending | Pending | Pending | Pending | LM loss, MTP loss |
| `arch_v2_attention_control` | Pending | Pending | Pending | Pending | Pending | Pending | theoretical KV elements |
| `arch_v2_mla` | Pending | Pending | Pending | Pending | Pending | Pending | theoretical KV elements |

## Keep Claims Separate

| Source | Supported statement | Unsupported shortcut |
| --- | --- | --- |
| DeepSeek LLM | Its grid searches found a broad near-optimal LR/batch region and observed larger batches and smaller LR as compute increased | Four TinySeek runs reproduce a scaling law |
| DeepSeekMoE | It proposes fine-grained experts and shared-expert isolation and reports specialization and compute benefits at its scale | TinySeek's Python dispatch proves distributed MoE throughput |
| DeepSeek-V2 | Relative to DeepSeek 67B, it reports 42.5% lower training cost, 93.3% less KV cache, and 5.76x maximum throughput | A theoretical cache estimate gives TinySeek the same speedup |
| DeepSeek-V3 | It uses auxiliary-loss-free balancing and MTP, supported by its ablations | Pending TinySeek runs are guaranteed to improve loss |

## Reading Failures

- Worse GQA loss can mean too few KV heads; cache is not the only metric.
- MoE can lose when the token budget is too small, routing has not specialized, or activated parameters are mismatched.
- Persistent bias-routing collapse suggests too small an update rate; oscillation suggests too large a rate.
- MTP increases the reported total objective; compare main `lm_loss`, validation PPL, and task metrics.
- Educational MLA may not reduce training VRAM because the trainer has no production latent-KV decoding kernel.

<!-- tinyseek-nav -->

---

[Experiment Hub](README.md)
