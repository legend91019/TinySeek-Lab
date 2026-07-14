# Fair Experiments for DeepSeek Architecture Evolution

[中文](06_architecture_evolution_plan_zh.md) | English

These experiments do not ask whether TinySeek matches DeepSeek capability. They ask a smaller, testable question: with data, token budget, and optimizer held fixed, how does one architecture change affect loss, throughput, memory, and routing?

Do not enable every new component at once. Use this order:

```text
lock the Dense recipe
-> ablate ordinary MoE into DeepSeekMoE
-> measure the KV-cache bottleneck and test low rank/MLA
-> map the auxiliary-loss trade-off, then compare selection bias
-> test MTP last
```

Proceed only when a stage passes its prewritten decision gate. Keep failed runs: they decide whether to revise the hypothesis, spend more training budget, or retain the previous generation.

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

Repository configs use one seed for pilot runs. A formal conclusion should use at least three seeds and report means, standard deviations, and every individual run. If budget permits one seed only, label the result `pilot`, not stable evidence.

`tests/architecture_lab_contract_test.py` checks the matched configurations automatically.

## Matrix

| ID | Comparison | Only change | Paper motivation | TinySeek status |
| --- | --- | --- | --- | --- |
| A1 | MHA vs GQA | `num_kv_heads` | DeepSeek LLM 67B uses GQA to reduce inference cost | Pending GPU run |
| A2a | coarse MoE vs fine-grained MoE vs shared isolation | expert count, width, top-k, and shared split; expert-FFN capacity and active width are matched | DeepSeekMoE fine-grained segmentation and shared-expert isolation | Pending GPU run |
| A2b | aux weights `0/0.001/0.01/0.1` | `moe_aux_loss_weight` | measure the load-balance versus main-task trade-off first | Pending GPU run |
| A2c | reasonable aux baseline vs bias balance | routing strategy and auxiliary weight | V3 avoids direct LM-gradient interference from balance loss | Pending GPU run |
| A3 | MTP off vs on | `mtp_depth` | V3 adds future-token objectives | Pending GPU run |
| A4a | GQA control vs naive low-rank KV | `attention_impl`, with RoPE still coupled | quality cost of low-rank projection itself | Pending GPU run |
| A4b | GQA control vs educational MLA | `attention_impl`, with decoupled RoPE | V2 compresses cacheable KV state into a low-rank latent | Pending GPU run |

## Commands

Use the same corpus path for every command, for example `data/tinystories.jsonl`:

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/dense_mha.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/dense_gqa.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/moe_coarse.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_fine_grained.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_shared.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_none.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_weak.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_strong.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/v3_no_mtp.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_mtp.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_control.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_kv.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data "$DATA" --hourly_rate 2.18
```

## Result Table

`val loss` means token-weighted `val_lm_loss`, and PPL is calculated from that main language-model loss only. Record `val_objective`, `val_mtp_loss`, and `val_aux_loss` separately; do not mix them into PPL or use them directly to rank A/B arms.

| Run | val loss | PPL | tokens/s | peak VRAM | GPU h | cost | architecture metric |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `arch_dense_mha` | Pending | Pending | Pending | Pending | Pending | Pending | KV elements/token |
| `arch_dense_gqa` | Pending | Pending | Pending | Pending | Pending | Pending | KV elements/token |
| `arch_moe_coarse` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, total/active params |
| `arch_moe_fine_grained` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, total/active params |
| `arch_moe_shared` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, total/active params |
| `arch_moe_aux_none` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, aux=0 |
| `arch_moe_aux_weak` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, aux=0.001 |
| `arch_moe_aux` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, aux loss |
| `arch_moe_aux_strong` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, aux=0.1 |
| `arch_moe_bias` | Pending | Pending | Pending | Pending | Pending | Pending | expert load, selection bias |
| `arch_v3_no_mtp` | Pending | Pending | Pending | Pending | Pending | Pending | LM loss |
| `arch_v3_mtp` | Pending | Pending | Pending | Pending | Pending | Pending | LM loss, MTP loss |
| `arch_v2_low_rank_control` | Pending | Pending | Pending | Pending | Pending | Pending | full KV elements |
| `arch_v2_low_rank_kv` | Pending | Pending | Pending | Pending | Pending | Pending | low-rank quality; no latent-cache claim |
| `arch_v2_attention_control` | Pending | Pending | Pending | Pending | Pending | Pending | theoretical KV elements |
| `arch_v2_mla` | Pending | Pending | Pending | Pending | Pending | Pending | theoretical KV elements |

## Stage Decision Gates

| Transition | Upgrade when | Keep or investigate when |
| --- | --- | --- |
| Dense -> GQA | theoretical KV state drops materially and multi-seed PPL does not regress significantly | quality loss exceeds run variation; add KV heads or retain MHA |
| ordinary MoE -> fine-grained/shared | no collapse; PPL improves consistently or similar quality supports more capacity, with acceptable throughput | only one seed wins, loads collapse, or Python dispatch cost dominates |
| GQA -> MLA | the `192 -> 72` theoretical-element reduction holds and PPL remains acceptable | sweep latent rank when quality drops; never publish real throughput without cached decoding |
| aux -> bias | bias loads are at least as healthy as a reasonable aux baseline and main LM/PPL is no worse | sweep bias update rate on collapse or oscillation; retain aux when it is more stable |
| MTP off -> on | multi-seed main LM/PPL or mini-eval improves and added memory/time is acceptable | lower total objective alone does not count; disable MTP without main-task benefit |

End every stage report with `observation`, `gate result`, `decision`, and `next experiment`. Before GPU cells are filled, the decision remains `Pending`, never "upgrade succeeded."

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
