# MoE Routing Report

This report is generated from `expert_load` snapshots in `*_cost_summary.json`.
The snapshot is lightweight: it records the latest observed expert assignment counts, not a full training trace.

No expert-load data found yet.

Run a MoE config with the current trainer, then regenerate this report:

```bash
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
```
