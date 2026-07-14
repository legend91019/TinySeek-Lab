# TinySeek-Lab Experiment Reports

[中文](README_zh.md) | English

This directory is the experiment-report hub for TinySeek-Lab. The tutorial path
is to revisit DeepSeek's language-model research route at a small scale, so each
report should answer:

- What changed: architecture, recipe, data, post-training objective, or eval?
- What did it cost: GPU, GPU hours, rental cost, peak VRAM, rough FLOPs?
- What moved: validation loss, PPL, mini eval, expert load?
- What did we learn, and what should not be overclaimed?

## Main Reports

| Report | Status | What it shows |
| --- | --- | --- |
| [RTX 4090 v1 Results](05_4090_v1_results.md) | Done | End-to-end TinyStories run through dense, sweep, MoE, MLA, SFT, and GRPO mini |
| [v1 Auto Summary and Figures](v1_4090_plan/auto_summary.md) | Done | PPL, VRAM, cost, sweep loss, and generated SVG figures |
| [v1 Pipeline Smoke Report](03_v1_pipeline_smoke_report.md) | Done | Pretrain -> SFT -> GRPO mini pipeline sanity check |
| [AutoDL 4090 Smoke Report](02_autodl_4090_smoke_report.md) | Done | RTX 4090 environment and minimal training validation |
| [Next Formal Experiment Plan](04_formal_experiment_plan.md) | Planned | Longer baseline, MoE analysis, and stronger GRPO follow-ups |
| [Final GPU Checklist](../docs/18_gpu_fill_only_checklist.md) | Done | Commands and report steps for the next rented-GPU run |

## v1 Headline

```text
TinyStories -> tiny base -> dense 35M/115M -> LR/batch sweep
-> MoE -> MLA -> SFT -> GRPO mini -> mini eval -> cost and figures
```

- The full v1 path ran successfully on an RTX 4090.
- Total GPU time was about `0.0867 h`, or about `0.19 CNY` at `2.18 CNY/h`.
- In the 35M dense 5M-token sweep, `bs16_lr3e-4` performed best.
- The short 115M run underperformed the 35M run, showing that bigger models do
  not automatically win under too little token budget.
- The MoE run had `235.06M` total parameters and about `84.06M` activated
  parameters, with about `5.46 GB` peak allocated VRAM.
- The educational MLA path demonstrates the KV-latent idea, but is not a
  production-grade DeepSeek-V2 MLA reproduction.
- GRPO mini is currently for teaching the algorithm shape, not serious RL
  performance.

## v1 Figures

![PPL comparison](v1_4090_plan/figures/v1_ppl.svg)

![Peak VRAM comparison](v1_4090_plan/figures/v1_peak_vram.svg)

![GPU cost comparison](v1_4090_plan/figures/v1_cost.svg)

![Sweep comparison](v1_4090_plan/figures/v1_sweep_val_loss.svg)

![VRAM versus PPL](v1_4090_plan/figures/v1_vram_vs_ppl.svg)

## MiniMind-Inspired Improvements

MiniMind is strong because the repository immediately tells readers the cost,
time, complete pipeline, data access, evaluation path, and deployment options.
TinySeek-Lab should keep moving in that direction:

| Area | Now | Next |
| --- | --- | --- |
| Results entrance | Reports exist, but used to be too hidden | Put this report hub and v1 result near the README top |
| Data | TinyStories/HF data plus toy SFT/GRPO | Add a recommended ready-made text-data mix |
| Eval | PPL, addition, format score | Add stronger arithmetic/reasoning mini evals |
| Post-training | Runnable SFT and GRPO mini | Strengthen cold-start SFT before GRPO |
| MoE analysis | Expert-load snapshots | Generate routing histograms and expert-load figures |
| Cost story | GPU hours and cost logged | Show "what cost buys what result" on the front page |
| Code teaching | Model and training-loop walkthroughs | Keep adding file-by-file and block-by-block explanations |

## Regenerate Figures

```bash
python scripts/generate_v1_report_assets.py --run_dir experiments/v1_4090_plan
python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md
```

It reads:

- `experiments/v1_4090_plan/cost_summary.csv`
- `experiments/v1_4090_plan/eval_*.json`

It writes:

- `experiments/v1_4090_plan/auto_summary.md`
- `experiments/v1_4090_plan/auto_summary_zh.md`
- `experiments/v1_4090_plan/figures/*.svg`
- `experiments/moe_routing_report.md`

## Report Template

Each formal report should include:

- `Environment`: GPU, CUDA/PyTorch, dataset, hourly rate.
- `Commands`: reproducible train/eval commands.
- `Cost`: GPU hours, cost, peak VRAM, tokens, rough FLOPs.
- `Metrics`: validation loss, PPL, task scores.
- `Figures`: at least loss/PPL, VRAM, and cost.
- `Takeaways`: what the result supports, and what it does not prove.
