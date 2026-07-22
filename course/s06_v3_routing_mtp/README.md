# s06 DeepSeek-V3: Routing Bias and Multi-Token Prediction

[中文](README_zh.md) | English | [Course index](../README.md)

## Two New Conflicts

1. An auxiliary routing loss can improve expert balance but also perturbs the language-model objective.
2. Next-token prediction gives one target per position; MTP asks whether predicting additional future tokens provides useful training signal.

These are separate hypotheses and need separate ablations.

## Paper Clue

DeepSeek-V3 introduces an auxiliary-loss-free load-balancing strategy based on expert selection bias, and uses multi-token prediction during training. TinySeek keeps readable versions of both mechanisms in [`model/stages/stage3_deepseek_v3.py`](../../model/stages/stage3_deepseek_v3.py) and the configurable formal model [`model/tinyseek.py`](../../model/tinyseek.py).

## Code Diff A: Selection Bias

Selection scores use router affinity plus a non-gradient bias; mixture weights still come from the original affinity. After an optimizer update, underloaded experts receive a positive bias adjustment and overloaded experts a negative one.

The distinction matters: changing the selection decision is not the same as changing the differentiable mixture weight. Read the formulas and exact `torch.no_grad` update in [`docs/23_from_v2_to_deepseek_v3.md`](../../docs/23_from_v2_to_deepseek_v3.md).

## Code Diff B: MTP

The extra head aligns a previous hidden state with a farther future target. It is not merely “shift labels twice”; sequence lengths and target positions must match exactly. The main loss remains present:

```text
total loss = LM loss + routing term + mtp_loss_weight * MTP loss
```

## Experiment Cards

| Hypothesis | Control | Candidate | Gate |
| --- | --- | --- | --- |
| bias routing | `moe_aux` | `moe_bias` | comparable PPL and lower load imbalance without aux loss |
| MTP | `v3_no_mtp` | `v3_mtp` | consistent main-task gain worth the extra memory/compute |

## Evidence and Decisions

- Bias routing: PPL `2.009 -> 2.024`, load CV `0.075 -> 0.081`. It fails this hyperparameter setting; keep `aux=0.01` and sweep the bias update rate before reconsidering.
- MTP: PPL `2.207 +/- 0.017` vs `2.196 +/- 0.034`, peak VRAM `0.197 -> 0.246 GB`. This is inconclusive and was measured only on the already rejected MLA-style branch. It does **not** decide whether MTP should be enabled on the promoted GQA+aux branch.

![VRAM comparison](../../experiments/architecture_lab_runs/figures/architecture_vram.svg)

## Research Lesson

An experiment can explain an idea without validating it in every regime. Do not stack winners and losers from incompatible topology groups into a fictional “best model.” Report the branch on which each result was measured.

## Next

Architecture work gives a pretrained base model. DeepSeek-R1 changes the training pipeline rather than replacing the Transformer block. Continue to [s07 Cold-start SFT](../s07_cold_start_sft/README.md).

<!-- tinyseek-nav -->

Previous: [s05 MLA](../s05_mla/README.md) | [Course index](../README.md) | Next: [s07 Cold-start SFT](../s07_cold_start_sft/README.md)
