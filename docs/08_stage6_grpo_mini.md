# 08. Stage 6: Rule-Based GRPO Mini

Goal: make R1-style rule RL concrete at toy scale.

## DeepSeek Anchor

DeepSeek-R1 uses GRPO and rule rewards for verifiable tasks. R1-Zero avoids
supervised reasoning traces before RL; R1 adds cold-start data for readability
and language consistency.

## Tiny Rewards

Start with tasks where correctness is cheap:

- arithmetic answer match;
- exact string extraction;
- simple Python unit tests;
- answer format reward;
- repetition penalty.

## Minimal GRPO Loop

For each prompt:

1. sample `G` completions;
2. score each completion with rule rewards;
3. normalize rewards inside the group;
4. update with `-advantage * completion_logprob` plus a reference KL proxy.

The boundary matters: this code does not store old-policy log probabilities and does not implement the paper-style clipped importance ratio. It teaches group sampling, rule rewards, within-group advantages, and reference regularization; it is not a complete GRPO reproduction.

The first implementation can be slow and small. Clarity beats throughput.

## Implemented Code Path

- [`JsonlPromptDataset`](../dataset/lm_dataset.py) reads prompts and verifiable answers.
- [`sample_group`](../trainer/train_grpo.py) samples multiple completions per prompt.
- `rule_reward` scores final-integer correctness and format shaping.
- Group-normalized rewards become advantages, combined with a reference KL proxy.

```bash
python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --hourly_rate 2.18
```

The formal suite gives both arms the same **GRPO update budget** and reports the cold-start arm's additional SFT cost separately:

```text
pretrained base -> direct GRPO
pretrained base -> structured cold-start SFT -> GRPO
```

Both routes are evaluated on tagged-answer accuracy over five held-out additions, reasoning format, PPL on the first 100 TinyStories rows, tracked training-process time, and estimated cost. Only a stable multi-metric gain supports a TinySeek-specific cold-start benefit.

The formal run makes the reward-hacking risk concrete. Direct GRPO ends with mean reward `0.1`, but held-out answer accuracy and reasoning-format score are both `0.0`. SFT -> GRPO reaches mean reward `0.2`, yet answer accuracy remains `0.0` and format falls from SFT's `0.6` to `0.2`. The proxy reward improved while the target behavior did not.

The supported conclusion is that the educational loop runs and exposes reward-design failure, not that RL taught reasoning. See the [post-training code walkthrough](19_posttraining_code_walkthrough.md) and [measured report](../experiments/gpu_completion_runs/report.md).

## Suggested Experiments

1. Direct RL from base.
2. RL after cold-start reasoning SFT.
3. Reward with format term vs correctness-only reward.
4. Group size 4 vs 8 vs 16.

<!-- tinyseek-nav -->

---

Previous: [Stage 5: SFT](07_stage5_sft_cold_start.md) | [Tutorial Index](README.md) | Next: [Post-Training Code Walkthrough](19_posttraining_code_walkthrough.md)
