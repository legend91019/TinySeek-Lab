# DeepSeekMath: the mathematical-reasoning path before R1

English | [中文](README_zh.md) | [Next: DeepSeek-R1](../05-deepseek-r1/README.md) | [Paper index](../README.md)

Paper: [DeepSeekMath](https://arxiv.org/abs/2402.03300). It is not a new Transformer architecture; it is a training predecessor to R1.

## 1. Position

DeepSeekMath 7B continues pretraining DeepSeek-Coder-Base-v1.5 7B with about 120B math-related tokens while retaining natural-language and code data. The paper reports 51.7% on competition-level MATH and 60.9% with 64-sample self-consistency. Its central contributions are data selection and GRPO.

## 2. Data and training

The paper mines mathematical web data, filters it with classifiers, heuristics, and deduplication, then continues pretraining. Supervised math fine-tuning comes before reinforcement learning with verifiable rewards. The sequence matters: RL is not the only source of reasoning; data quality and domain pretraining provide the foundation.

## 3. GRPO and verifiers

Group Relative Policy Optimization samples several answers for one problem, builds advantages from relative group rewards, and removes the separate PPO value model. Rule or program verifiers can judge mathematical correctness more reliably than open-ended preference rewards. GRPO removes critic parameters, not rollout, sampling, or reward-design costs.

## 4. Reading the experiments

Focus on data-filtering ablations, continued-pretraining comparisons, and SFT/GRPO comparisons. The 51.7% and 60.9% MATH results are paper-scale numbers, not TinySeek baselines.

## 5. Link to R1

DeepSeekMath tests high-quality data, verifiable rewards, and GRPO in mathematics. R1 extends verifiable reasoning to broader math, coding, and STEM tasks, adding cold start, multi-stage SFT, mixed rewards, and distillation. DeepSeekMath validates the training ingredients; R1 broadens the task and pipeline.

## 6. TinySeek boundary

The repository's teaching SFT/GRPO run improved a format score from 0.0 to 0.6 while held-out addition accuracy stayed 0/5; a loose reward could later degrade the format. This does not contradict the paper. It shows that verifiable rewards, data scale, rollouts, and generalization must be evaluated together.

| Paper claim | Paper evidence | TinySeek supplement | Relation |
| --- | --- | --- | --- |
| Math data filtering improves the domain model | Data pipeline and ablations | No same-scale filtering | Paper-only |
| GRPO improves math reasoning | GRPO and MATH results | Teaching GRPO has no answer gain | Not directly comparable |
| Verifiable rewards suit math RL | Rule/program verification | Small format-vs-answer separation | Directional reminder |

## 7. Conclusion

DeepSeekMath is R1's training prehistory: it validates data engineering, continued pretraining, and GRPO in a verifiable domain. Reading it first prevents reducing R1 to a single RL algorithm swap.

## 8. Sources

- [arXiv](https://arxiv.org/abs/2402.03300)
- [PDF](https://arxiv.org/pdf/2402.03300)
- [Source ledger](../assets/deepseek-math/SOURCES.md)

