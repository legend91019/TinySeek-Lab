# 00. Project Scope

TinySeek-Lab is a language-model-only tutorial.

Included:

- Decoder-only LM pretraining.
- Training recipe sweeps.
- Data mixture experiments.
- RMSNorm, RoPE, SwiGLU, GQA.
- DeepSeekMoE-style routed experts.
- Educational MLA-style low-rank KV compression.
- SFT, reasoning cold start, DPO, rule-based GRPO mini.
- Rejection sampling and distillation as later chapters.

Excluded from the main path:

- Vision-language models.
- Image/video generation.
- OCR.
- Embodied AI and robotics.
- Tool-use agents as a first-class target.

Why exclude them? The goal is to make the language-model training path legible.
DeepSeek's multimodal work is interesting, but it would blur the first repo's
core lesson.

## Output of the Project

The repo should produce:

- A trainable 30M-150M dense baseline.
- A 100M-300M total-parameter MoE variant with much smaller activated params.
- Reproducible training curves for LR / batch-size sweeps.
- A report template for each experiment.
- A small reasoning post-training demo.

## Non-Goal

We do not claim to reproduce DeepSeek model quality. We reproduce the training
questions and architecture moves at toy scale.
