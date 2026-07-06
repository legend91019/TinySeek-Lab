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
4. update model with clipped policy ratio and KL to reference model.

The first implementation can be slow and small. Clarity beats throughput.

## Experiments

1. Direct RL from base.
2. RL after cold-start reasoning SFT.
3. Reward with format term vs correctness-only reward.
4. Group size 4 vs 8 vs 16.
