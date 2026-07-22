# s02 Training Recipe: Fix the Control Group

[中文](README_zh.md) | English | [Course index](../README.md)

## Research Question

If the learning rate or effective batch size is poor, an architecture comparison is not interpretable. Which small region of the recipe should become our control setting?

## Paper Clue

DeepSeek LLM reports systematic training-recipe studies, including batch size and learning-rate choices. TinySeek turns that idea into a small grid; it is a method lesson, not a scaling-law claim.

## Code Change

There is deliberately no model change. Read [`trainer/train_pretrain.py`](../../trainer/train_pretrain.py) and [`trainer/sweep_pretrain.py`](../../trainer/sweep_pretrain.py): config loading, effective tokens per optimizer step, warmup/cosine scheduling, AMP, clipping, validation and the cost ledger.

The experiment configs are the executable research question:

- [`experiments/gpu_completion_runs/configs/`](../../experiments/gpu_completion_runs/configs/)
- [`configs/architecture_lab/`](../../configs/architecture_lab/)

## Experiment Card

```text
same model + same data + same token budget
change only batch size and learning rate
repeat the selected candidates with the same validation split
```

The formal 4090 sweep uses `bs16/bs32 x lr3e-4/lr6e-4`. Record loss, PPL, throughput, peak VRAM, GPU hours and cost together; a lower loss that costs an unreasonable amount is not automatically the best teaching control.

## Result and Decision

The formal report selects `formal_sweep_bs16_lr6e-4` within this tiny budget (`val loss 0.6475`). This is a local recipe choice, not a universal LR rule. Freeze the recipe for the architecture lab, then change one architectural variable at a time.

Read the complete report in [`experiments/gpu_completion_runs/report.md`](../../experiments/gpu_completion_runs/report.md) and the original lesson in [`docs/03_stage1_lr_batch_search.md`](../../docs/03_stage1_lr_batch_search.md).

## Code Exercise

Print the number of tokens consumed by one optimizer update. Then explain why changing `batch_size` can change both gradient noise and peak memory, while changing `grad_accum_steps` can preserve the effective batch but still change wall-clock behavior.

## Next

The control group is fixed. The first architecture question is attention-side: [s03 GQA](../s03_gqa/README.md) asks whether K/V state can be reduced without losing the baseline's language-model quality.

<!-- tinyseek-nav -->

Previous: [s01 Dense baseline](../s01_dense_baseline/README.md) | [Course index](../README.md) | Next: [s03 GQA](../s03_gqa/README.md)
