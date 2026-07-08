# GPU Choice and Cost Tracking

This chapter is the hardware ledger for TinySeek-Lab. The goal is not only to
train a small model, but to teach readers how to report the cost of each
experiment: GPU memory, wall-clock time, GPU hours, rental price, and total
estimated cost.

## 3080 Ti or 4090?

For this repo, both can be useful, but they are useful at different points.

| GPU | Typical VRAM | AutoDL price | Best use in TinySeek-Lab |
| --- | ---: | ---: | --- |
| RTX 3080 Ti | 12 GB | 0.98 CNY/hour | Cheap smoke tests, dense baseline, small LR/batch sweeps, early code validation |
| RTX 4090 | 24 GB | 2.18 CNY/hour | Full v1.0 experiment path, larger batch sizes, MoE/MLA runs, fewer memory compromises |

Recommendation:

- Use a 3080 Ti if the goal is learning, debugging, and validating every stage
  cheaply.
- Use a 4090 if the goal is to finish a publishable v1.0 tutorial with
  repeatable experiment tables.

The 3080 Ti is cheaper per hour, but its 12 GB memory often forces smaller
batches, shorter sequence lengths, and more gradient accumulation. The 4090 is
more expensive per hour, but the extra VRAM usually makes the whole experiment
path smoother.

## Rough Budget

These are planning estimates, not promises. Actual cost depends on dataset
size, model width/depth, sequence length, batch size, and how much debugging is
needed.

| Milestone | Expected work | 4090 cost at 2.18 CNY/h | 3080 Ti cost at 0.98 CNY/h |
| --- | ---: | ---: | ---: |
| v0.2 | 12-24 GPU hours | 26-52 CNY | 12-24 CNY |
| v0.5 | 35-70 GPU hours | 76-153 CNY | 34-69 CNY |
| v1.0 | 60-120 GPU hours | 131-262 CNY | 59-118 CNY |

In practice, 3080 Ti runs may need more hours because of smaller batches and
more fragmented experiments, so the real gap is smaller than the hourly prices
suggest.

## Per-Run Logging

`trainer/train_pretrain.py` writes a cost summary after every run:

```bash
python trainer/train_pretrain.py \
  --config configs/tiny_dense.json \
  --data data/toy_pretrain.jsonl \
  --max_steps 2000 \
  --hourly_rate 2.18 \
  --currency CNY
```

The output file is:

```text
out/<run_name>_cost_summary.json
```

It records:

- GPU name, CUDA availability, CUDA version, and total VRAM.
- Peak allocated VRAM and peak reserved VRAM.
- Elapsed seconds and GPU hours.
- Hourly rate, currency, and estimated cost.
- Estimated training tokens and a rough training FLOPs estimate.
- Model parameter count and activated-parameter estimate.
- Final training loss and validation loss.

For AutoDL examples:

```bash
# RTX 4090
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --hourly_rate 2.18

# RTX 3080 Ti
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --hourly_rate 0.98
```

## Sweep Logging

The sweep runner forwards the same price metadata:

```bash
python trainer/sweep_pretrain.py \
  --sweep experiments/01_lr_batch_grid.json \
  --hourly_rate 2.18 \
  --currency CNY
```

After the sweep, summarize all run ledgers:

```bash
python scripts/summarize_costs.py --input_dir out
```

This writes:

```text
out/cost_summary.md
out/cost_summary.csv
```

Use the Markdown table in tutorial reports and the CSV file for spreadsheets.

## What to Record in Reports

Every serious experiment report should include:

| Field | Why it matters |
| --- | --- |
| GPU model | Hardware changes training speed and memory headroom |
| Peak allocated VRAM | Shows the actual tensor memory pressure |
| Peak reserved VRAM | Shows PyTorch allocator pressure |
| GPU hours | Makes the experiment reproducible as a budget |
| Total cost | Makes tradeoffs visible to readers |
| Tokens processed | Lets readers compare efficiency across runs |
| Approximate FLOPs | Gives a rough compute scale for experiment comparison |
| Validation loss | Ties cost back to model quality |

TinySeek-Lab should treat cost as a real metric, not an afterthought.
