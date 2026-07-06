# 03. Stage 1: LR and Batch-Size Search

Goal: turn the DeepSeek LLM scaling-law spirit into a small reproducible sweep.

## DeepSeek Anchor

DeepSeek LLM reports that they first examined scaling laws of batch size and
learning rate, then studied model/data scale allocation. TinySeek cannot match
that scale, but it can teach the method:

- fix model and token budget;
- sweep batch size and learning rate;
- compare validation loss and stability;
- only then scale the model.

## Tiny Sweep

```bash
python trainer/sweep_pretrain.py --sweep experiments/01_lr_batch_grid.json
```

The first sweep varies:

- batch size: 8, 16
- learning rate: 3e-4, 6e-4

For a real run, expand to:

```text
batch tokens: 32K, 64K, 128K, 256K
lr: 1e-4, 3e-4, 6e-4, 1e-3
warmup: 1%, 2%, 5%
```

## Report Template

| Run | Batch tokens | LR | Warmup | Val loss | Tokens/sec | Notes |
|---|---:|---:|---:|---:|---:|---|
| bs8_lr3e-4 | TBD | 3e-4 | TBD | TBD | TBD | baseline |

## Takeaway

The most important habit is not the exact best LR. It is learning to run
controlled sweeps before adding architectural complexity.
