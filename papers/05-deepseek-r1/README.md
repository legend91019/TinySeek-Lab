# DeepSeek-R1: post-training a reasoning model

English | [中文](README_zh.md) | [Previous: DeepSeek-V3](../04-deepseek-v3/README.md) | [Paper route](../README.md)

Paper: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025, [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)). R1 changes the post-training pipeline rather than the Transformer block.

## 1. Position

R1-Zero starts from DeepSeek-V3-Base with reasoning prompts and rule-based rewards. The paper observes longer thinking and self-reflection, but also poor readability and language mixing. Full R1 adds cold-start long-CoT data, rejection sampling, mixed reasoning/non-reasoning SFT, two RL stages, and helpfulness/harmlessness reward models.

## 2. Pipeline transition

R1-Zero demonstrates self-evolution; Dev1 adds cold start; Dev2 applies reasoning RL; Dev3 mixes reasoning and general SFT; R1 applies mixed RL. Distillation transfers reasoning traces to smaller dense models.

## 3. Method

GRPO samples a group per prompt, normalizes group rewards into advantages and optimizes sampled-token log probabilities without a PPO value model. Rule rewards score verifiable math/code/logic tasks; reward models score general helpfulness and safety; language-consistency rewards reduce mixed-language CoT. The paper explicitly documents reward hacking.

## 4. Paper experiments

![R1-Zero](../assets/deepseek-r1/paper-fig-1-r1-zero-training.png)

![Pipeline](../assets/deepseek-r1/paper-fig-2-r1-pipeline.png)

![GRPO](../assets/deepseek-r1/paper-fig-3-grpo.png)

![Stage results](../assets/deepseek-r1/paper-table-3-stage-results.png)

![Reward hacking](../assets/deepseek-r1/paper-fig-6-reward-hacking.png)

Table 3 shows why the pipeline is staged: cold start improves usability, reasoning RL recovers reasoning quality, mixed SFT improves general behavior, and final RL combines objectives.

## 5. TinySeek implementation

See [`train_sft.py`](../../trainer/train_sft.py), [`train_grpo.py`](../../trainer/train_grpo.py), [`19_posttraining_code_walkthrough.md`](../../docs/19_posttraining_code_walkthrough.md), [`s07`](../../course/s07_cold_start_sft/README.md) and [`s08`](../../course/s08_grpo_and_evaluation/README.md). The teaching path has prompt masking and group-relative rule rewards, not R1's rollout scale, reward models or complete pipeline.

## 6. TinySeek supplemental evidence

On five held-out addition problems, answer accuracy stays `0/5`. Format score moves `0.0 -> 0.6` after cold-start SFT and falls to `0.2` after SFT+GRPO; TinyStories PPL moves `1.718 -> 12.670 -> 12.306`.

![Post-training](../../experiments/gpu_completion_runs/figures/posttraining_reasoning.svg)

This is a teaching boundary, not a contradiction of R1: format compliance and proxy reward are not answer generalization.

## 7. Claim/evidence table

| Paper claim | Paper evidence | TinySeek evidence | Relation | Boundary |
| --- | --- | --- | --- | --- |
| Rule RL can induce reasoning evolution | Figure 1 and R1-Zero experiments | Direct GRPO has no answer gain | Paper-only | Rollout, data and model differ |
| Cold start improves usability | Figure 2 and Table 3 | Format `0.0 -> 0.6`, answers `0/5` | Directionally consistent: supplemental check | Only format direction is checked |
| GRPO removes the value-model cost | Figure 3 | Group-relative teaching update | Directionally consistent: supplemental check | No paper-scale cost/stability |
| Reward hacking needs auditing | Figure 6 | Format degradation after GRPO | Directionally consistent: supplemental check | Small proxy-reward case |

## 8. Conclusion

R1 is a staged post-training recipe: observe pure-RL reasoning, add readable cold start, mix supervised data, then balance reasoning, helpfulness and safety with multiple rewards. TinySeek's value is to keep answer accuracy, format, reward and base-distribution PPL separate.

## 9. Source

Paper: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948). This is the final paper in the route.
