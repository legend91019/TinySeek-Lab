# 10. Experiment Report Template

Use this template for every chapter. The point is to learn research habits, not
just run scripts.

## Title

Short name of the experiment.

## DeepSeek Anchor

Which paper and section motivated the experiment?

Example:

- Paper: DeepSeek LLM.
- Idea: batch-size and learning-rate search before scaling.

## Hypothesis

What do you expect to happen?

Example:

> At fixed token budget, a larger effective batch will prefer a slightly higher
> learning rate, but too high a LR will destabilize validation loss.

## Setup

- Model config:
- Data:
- Token budget:
- Hardware:
- Seed:

## Sweep

| Run | Changed variables |
|---|---|
| baseline | none |

## Metrics

- Train loss.
- Validation loss.
- Tokens/sec.
- Peak memory.
- Mini downstream eval if relevant.

## Results

| Run | Train loss | Val loss | Tokens/sec | Peak memory | Notes |
|---|---:|---:|---:|---:|---|

## Takeaways

What should the next experiment change?

## Failure Cases

Record broken runs too. Failed runs are often the most useful part of the lab.
