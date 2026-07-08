# 05. Stage 3: Tiny DeepSeekMoE

Goal: replace dense FFN with routed experts.

## DeepSeek Anchor

DeepSeekMoE argues for expert specialization and discusses load balance. The
practical failure mode is routing collapse: most tokens choose a small subset of
experts, leaving other experts undertrained.

## TinySeek Implementation

Set `use_moe: true` in config.

```bash
python trainer/train_pretrain.py --config configs/tiny_moe.json --data data/toy_pretrain.jsonl
```

Implemented:

- top-k routing;
- routed experts;
- optional shared experts;
- auxiliary load-balance proxy loss;
- total vs activated parameter estimate.
- lightweight expert-load snapshots in `*_history.jsonl` and
  `*_cost_summary.json`.

## Experiments

1. Dense vs MoE with similar activated parameters.
2. `top_k = 1` vs `top_k = 2`.
3. `moe_aux_loss_weight = 0, 0.001, 0.01, 0.05`.
4. With and without shared experts.

## Report Questions

- Did routing collapse?
- Which experts get most tokens?
- Does auxiliary loss improve load balance?
- Is validation loss better at equal activated params?

The expert-load snapshot records per-layer expert counts from the latest
forward pass. It is not a full routing trace, but it is enough to spot obvious
routing collapse during tutorial-scale experiments.

<!-- tinyseek-nav -->

---

Previous: [Stage 2: Block Upgrades](04_stage2_block_upgrades.md) | [Tutorial Index](README.md) | Next: [Stage 4: MLA](06_stage4_mla.md)
