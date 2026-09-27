# TinySeek Paper Route

This is the repository's only recommended reading entry. It separates the architectural route from the reasoning-training route so that V3.2/V4 and R1 are not presented as the same kind of transition.

Each article combines the paper's method and evidence with the existing TinySeek implementation and small-scale results. No new training or ablation runs are introduced. TinySeek measurements supplement, but never replace, paper-scale evidence.

Each article now follows one analytical order: identify the bottleneck, explain why the proposed mechanism should help, decompose controls and ablations, test what the reported results actually support, and state what TinySeek can and cannot transfer. The goal is to learn how to read the experiment logic, not to memorize abbreviations.

## Main architecture route

1. [DeepSeek LLM: training recipes and scaling laws](01-deepseek-llm/README.md)
2. [DeepSeekMoE: separating capacity from activated compute](02-deepseek-moe/README.md)
3. [DeepSeek-V2: compressing the KV cache with MLA](03-deepseek-v2/README.md)
4. [DeepSeek-V3: auxiliary-loss-free balancing and MTP](04-deepseek-v3/README.md)
5. [DeepSeek-V3.2-Exp: DeepSeek Sparse Attention](06-deepseek-v3-2-sparse-attention/README.md)
6. [DeepSeek-V4: CSA/HCA, mHC, and million-token context](07-deepseek-v4/README.md)

## Reasoning-training route

1. [DeepSeekMath: mathematical data, GRPO, and verifiable rewards](08-deepseek-math/README.md)
2. [DeepSeek-R1: post-training a reasoning model](05-deepseek-r1/README.md)

Chinese: [中文主线](README_zh.md)

## Evidence labels

- **Paper evidence**: the DeepSeek paper's own models, data and experiments.
- **TinySeek supplemental evidence**: an experiment already present in this repository.
- **Directionally consistent: supplemental check**: both observe a similar direction, but scales are not comparable.
- **Direction differs: not directly transferable**: TinySeek does not reproduce the paper's gain; this does not refute the paper.
- **Paper-only / TinySeek-only / insufficient or incomparable evidence**: use the narrowest label that fits.

Paper asset provenance is tracked in [`assets/`](assets/README.md). The old `course/`, `docs/` and `experiments/` trees remain as historical lessons, reference notes and evidence archives.
