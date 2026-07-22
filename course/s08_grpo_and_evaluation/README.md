# s08 GRPO and Evaluation: Let Evidence Stop the Story

[中文](README_zh.md) | English | [Course index](../README.md)

## Research Question

Given multiple sampled completions for one prompt, can a rule-based reward make the SFT-initialized model more correct without destroying format behavior or the base distribution?

## Paper Clue and Scope

DeepSeek-R1 uses GRPO-style reinforcement learning as part of a much larger pipeline. TinySeek implements an educational group-relative update: sample a group, score completions, normalize rewards into advantages, and optimize sampled-token log probabilities.

This is **not** a faithful R1-scale reproduction. It lacks the full objective, infrastructure, data scale, reward coverage and repeated pipeline stages.

## Code Path

Read [`trainer/train_grpo.py`](../../trainer/train_grpo.py) and [`docs/19_posttraining_code_walkthrough.md`](../../docs/19_posttraining_code_walkthrough.md):

```text
prompt -> G completions -> rule rewards r_i
-> A_i = (r_i - mean(r)) / (std(r) + eps)
-> gather sampled-token log probabilities
-> loss = -mean(A_i * log pi(completion_i))
```

The code must gather only sampled completion tokens, detach advantages, and handle groups with nearly zero reward variance.

## Experiment Card

Compare two causal paths, not just two final checkpoints:

1. `base -> direct GRPO`
2. `base -> cold-start SFT -> GRPO`

Report answer accuracy, format score, mean reward and a common-distribution PPL probe. Keep reward score separate from held-out evaluation; optimizing the reward is not evidence that the task was solved.

## Evidence

| Checkpoint | Answer | Format | TinyStories sample PPL |
| --- | ---: | ---: | ---: |
| base | `0/5` | `0.000` | `1.718` |
| direct GRPO | `0/5` | `0.000` | `1.995` |
| cold-start SFT | `0/5` | `0.600` | `12.670` |
| SFT + GRPO | `0/5` | `0.200` | `12.306` |

## Decision

The loose reward does not produce arithmetic generalization and reduces the SFT format score. GRPO remains an educational mechanism demo, not a promoted reasoning result. The honest endpoint of this course is therefore a failed local RL hypothesis plus a concrete next research question: improve held-out evaluation, reward design and objective fidelity before scaling runs.

![Post-training evidence](../../experiments/gpu_completion_runs/figures/posttraining_reasoning.svg)

## Reproduce and Audit

Use the archived report and raw outputs:

- [`experiments/gpu_completion_runs/report.md`](../../experiments/gpu_completion_runs/report.md)
- [`experiments/gpu_completion_runs/eval_formal_reasoning_grpo.json`](../../experiments/gpu_completion_runs/eval_formal_reasoning_grpo.json)
- [`docs/08_stage6_grpo_mini.md`](../../docs/08_stage6_grpo_mini.md)

## What You Should Have Learned

You can now trace one stable LM interface through dense training, recipe search, GQA, MoE, MLA, V3-style objectives, SFT and educational GRPO. More importantly, you can distinguish a paper motivation, a code implementation, an experiment result and an engineering decision. They are related, but they are not interchangeable.

<!-- tinyseek-nav -->

Previous: [s07 Cold-start SFT](../s07_cold_start_sft/README.md) | [Course index](../README.md) | Next: [Experiment reports](../../experiments/README.md)
