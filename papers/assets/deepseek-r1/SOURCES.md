# DeepSeek-R1 sources

Paper: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (arXiv:2501.12948).

Source: <https://arxiv.org/pdf/2501.12948>

| Local asset | Paper source | Page | Reading note |
| --- | --- | ---: | --- |
| `paper-fig-1-r1-zero-training.png` | Figure 1, R1-Zero training trajectory | 4 | Rule-based RL is associated with rising reasoning performance and longer thinking. |
| `paper-fig-2-r1-pipeline.png` | Figure 2, multi-stage R1 pipeline | 6 | The final R1 is a pipeline, not a single GRPO call. |
| `paper-table-3-stage-results.png` | Table 3, stage-by-stage results | 9 | Cold start improves usability while later stages recover and extend reasoning quality. |
| `paper-fig-3-grpo.png` | Figure 3, PPO versus GRPO | 15 | GRPO removes the value model and uses group-relative rewards. |
| `paper-fig-6-reward-hacking.png` | Figure 6, reward hacking | 36 | More reward is not automatically more task correctness or helpfulness. |

