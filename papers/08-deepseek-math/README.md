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
+## Deep reading: DeepSeekMath validates the recipe before R1 scales it

### 1. Data engineering is a method contribution

The paper builds mathematical text with classifiers, heuristics, deduplication, and quality filters before RL. This order matters: without mathematical symbols, problem types, and solution language in the base distribution, a verifier only searches a poor policy space.

### 2. Why continued pretraining and SFT are separate

Continued pretraining changes the probability distribution over mathematical text; SFT turns problem/solution formats and reasoning traces into explicit behavior. MATH gains should therefore not be attributed to GRPO alone; the pipeline and stepwise ablations matter.

### 3. GRPO's conditions

GRPO needs multiple samples per prompt and relative group rewards. Mathematical verifiers make rewards cleaner than open-ended preference scores, but all-wrong groups still provide little learning signal. Removing a critic lowers one memory/parameter cost, not exploration or rollout cost.

### 4. Interpreting 51.7% versus 60.9%

Single-sample accuracy and 64-sample self-consistency measure different capabilities. The latter includes test-time compute and voting; it is not a single-generation accuracy claim.

### 5. The bridge to R1

DeepSeekMath validates verifiable rewards and GRPO in mathematics. R1 broadens the verifiable-task setting and adds cold start, mixed SFT, general rewards, and distillation. It is a staged research progression, not two unrelated RL papers.

### 6. TinySeek's boundary case

TinySeek improves format score while held-out addition remains 0/5. This does not refute DeepSeekMath because data, model, verifier, and rollout scale differ; it teaches the reader to separate format learning from reasoning generalization.
+## Evidence map

| Paper location | Question | Authors' conclusion | Reading boundary |
| --- | --- | --- | --- |
| Data construction | Does filtering improve math data? | The pipeline raises data density | Classifiers can shift the distribution |
| Continued pretraining / SFT | Can knowledge and format be established? | Both provide the RL foundation | RL cannot be isolated from the pipeline |
| GRPO | Does relative reward work without a critic? | Math reasoning improves while removing the value model | Rollouts and sampling remain expensive |
| MATH evaluation | How do single-shot and self-consistency differ? | 51.7% versus 60.9% is reported | The latter includes test-time compute |
