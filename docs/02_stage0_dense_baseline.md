# 02. Stage 0: Dense Baseline

Goal: train the simplest useful decoder-only language model.

## DeepSeek Anchor

DeepSeek LLM's architecture follows the modern LLaMA-style recipe:

- Pre-Norm Transformer.
- RMSNorm.
- RoPE.
- SwiGLU.
- GQA for larger models.

TinySeek starts with the same family of choices, but at a tiny scale.

## Experiment

Hypothesis: before doing MoE or RL, we need a stable dense baseline.

Setup:

- Model: `configs/tiny_dense.json`.
- Tokenizer: byte-level tokenizer for deterministic smoke tests.
- Objective: next-token prediction.
- Data: `data/toy_pretrain.jsonl` first, then a real text corpus later.

Run:

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 100
```

Metrics:

- Training loss.
- Validation loss.
- Tokens/sec.
- Sample generation.

Takeaway:

If this stage is unstable, every later stage will be noisy.
