# TinySeek Paper Route

This is the repository's only recommended reading path. Follow five DeepSeek papers to understand the research questions, the architectural transitions, and the experiments used to justify them.

Each article combines the paper's method and evidence with the existing TinySeek implementation and small-scale results. No new training or ablation runs are introduced. TinySeek measurements supplement, but never replace, paper-scale evidence.

## Reading route

1. [DeepSeek LLM: training recipes and scaling laws](01-deepseek-llm/README.md)
2. [DeepSeekMoE: separating capacity from activated compute](02-deepseek-moe/README.md)
3. [DeepSeek-V2: compressing the KV cache with MLA](03-deepseek-v2/README.md)
4. [DeepSeek-V3: auxiliary-loss-free balancing and MTP](04-deepseek-v3/README.md)
5. [DeepSeek-R1: post-training a reasoning model](05-deepseek-r1/README.md)

Chinese: [中文主线](README_zh.md)

## Evidence labels

- **Paper evidence**: the DeepSeek paper's own models, data and experiments.
- **TinySeek supplemental evidence**: an experiment already present in this repository.
- **Directionally consistent: supplemental check**: both observe a similar direction, but scales are not comparable.
- **Direction differs: not directly transferable**: TinySeek does not reproduce the paper's gain; this does not refute the paper.
- **Paper-only / TinySeek-only / insufficient or incomparable evidence**: use the narrowest label that fits.

Paper asset provenance is tracked in [`assets/`](assets/README.md). The old `course/`, `docs/` and `experiments/` trees remain as historical lessons, reference notes and evidence archives.

