# Code Walkthrough

This chapter is the code-reading path. Read it after the high-level roadmap and
before running larger experiments.

If chapter 12 explains how the initial model is built, chapter 16 explains how
the training program turns that model into checkpoints and reports. This chapter
is the map between those two views.

## Reading Order

1. `model/tinyseek_dense.py`: the smallest dense decoder-only LM.
2. `model/tinyseek.py`: the experiment model with GQA, MoE, and educational MLA.
3. `dataset/lm_dataset.py`: pretraining, SFT, and prompt datasets.
4. `trainer/train_pretrain.py`: base-model training.
5. `trainer/train_sft.py`: supervised fine-tuning and cold-start formatting.
6. `trainer/train_grpo.py`: rule-based GRPO mini.
7. `eval/mini_eval.py`: small checks for loss, addition, and format following.
8. `scripts/generate_v1_report_assets.py`: convert machine-readable results
   into tables and SVG figures.

## `model/tinyseek_dense.py`

This file is the teaching model. It should feel like building GPT from first
principles:

- A config dataclass stores model width, depth, heads, sequence length, and
  dropout.
- Token embeddings turn integer token IDs into hidden states.
- Each block applies causal self-attention and an MLP.
- The final LM head predicts the next token.

The important lesson is tensor shape. A decoder-only LM repeatedly transforms:

```text
input_ids [batch, seq]
-> embeddings [batch, seq, hidden]
-> blocks [batch, seq, hidden]
-> logits [batch, seq, vocab]
```

Read this file before `model/tinyseek.py`; it removes the extra DeepSeek-style
features so the core training objective is easy to see.

## `model/tinyseek.py`

This is the main experiment model. It keeps the same decoder-only skeleton but
adds the architecture ideas used in the tutorial path.

### `TinySeekConfig`

The config controls the architecture:

- `hidden_size`, `num_layers`, `num_heads`: dense Transformer scale.
- `num_kv_heads`: enables grouped-query attention when smaller than
  `num_heads`.
- `attention_impl`: selects normal attention or educational MLA.
- `use_moe`, `num_experts`, `top_k`: switch the FFN into a routed MoE layer.

### `RMSNorm`

RMSNorm rescales hidden states using their root-mean-square magnitude. It is
lighter than LayerNorm and common in modern LMs.

### RoPE Helpers

`precompute_rope`, `rotate_half`, and `apply_rope` implement rotary position
embeddings. The key idea is that position is applied to query/key vectors
before attention scores are computed.

### `CausalSelfAttention`

This module implements:

- Q projection for all attention heads.
- K/V projection for either all heads or fewer KV heads.
- KV repetition for GQA.
- Optional educational MLA, where K/V are reconstructed from a smaller latent.
- PyTorch scaled-dot-product attention with `is_causal=True`.

The DeepSeek connection:

- GQA helps reduce KV-cache cost.
- MLA goes further by compressing KV information through a latent path.

### `SwiGLU` and `DenseFFN`

The dense FFN uses SwiGLU:

```text
SwiGLU(x) = W2(silu(W1 x) * W3 x)
```

This is the MLP upgrade chapter: replace older ReLU/GELU MLPs with a modern LM
block.

### `MoEFFN`

The MoE layer is the DeepSeekMoE-inspired part:

- A router predicts probabilities over experts.
- Each token selects `top_k` experts.
- Expert outputs are weighted and summed.
- Shared experts can be added for common knowledge.
- A load-balancing auxiliary loss discourages routing collapse.

The tutorial should compare total parameters and activated parameters. MoE can
have many total parameters while activating only a small subset per token.

### `TinySeekForCausalLM`

The full model ties embeddings and LM head weights, computes cross entropy on
next-token prediction, and exposes:

- `generate`: simple autoregressive sampling.
- `parameter_count`: total parameter count.
- `activated_parameter_estimate`: dense vs MoE compute comparison.

## `dataset/lm_dataset.py`

TinySeek keeps data simple on purpose.

### `JsonlTextDataset`

Input format:

```json
{"text": "raw text for language modeling"}
```

It encodes text with the byte tokenizer, pads to `max_seq_len`, and masks pad
tokens with `-100` so cross entropy ignores them.

### `JsonlInstructionDataset`

Input format:

```json
{"prompt": "question", "response": "answer"}
```

It formats examples as:

```text
### Instruction
...

### Response
...
```

Prompt tokens are masked with `-100`; only response tokens contribute to SFT
loss. This is the practical meaning of cold-start SFT in this toy repo: teach
the model a readable response format before RL.

### `JsonlPromptDataset`

This is used by GRPO mini. It stores prompts and verifiable answers, such as
small arithmetic targets.

## `trainer/train_pretrain.py`

This is the base-model training loop:

1. Load config and dataset.
2. Build model and optimizer.
3. Run next-token prediction.
4. Apply AMP if `dtype` is `bfloat16` or `float16`.
5. Support gradient accumulation.
6. Save checkpoints and cost summaries.

Cost summaries record GPU name, peak memory, elapsed time, estimated cost,
tokens, and rough FLOPs. This makes every experiment reportable.

The trainer also writes `out/<run_name>_history.jsonl` at validation points.
Those rows are intentionally simple JSON so later chapters can draw loss curves
without depending on an external experiment tracker.

## `trainer/train_sft.py`

SFT reuses the same model and optimizer pattern, but swaps the dataset:

- Pretraining predicts all non-pad text tokens.
- SFT predicts only response tokens.

Use SFT to teach:

- instruction formatting,
- concise answers,
- cold-start reasoning style,
- domain-specific response conventions.

## `trainer/train_grpo.py`

This is an educational GRPO shape, not industrial RL.

For each prompt:

1. Sample a group of completions.
2. Score each completion with a rule reward.
3. Normalize rewards inside the group.
4. Increase log probability for above-average completions.
5. Add a small reference-model KL proxy.

The current reward is arithmetic-oriented: exact final integer gets full credit,
and answer-like formatting gets small shaping credit.

## `eval/mini_eval.py`

Mini eval gives quick feedback:

- Perplexity on JSONL text.
- Addition exact match.
- Format-following score.

It is not a benchmark. It is a sanity check for tutorial experiments.

## What to Read After This

For the full end-to-end program flow, read
[Training Loop: From Config to Checkpoint](16_training_loop_from_config_to_checkpoint.md).

After reading the code once, run the v1 runbook:

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 20
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --max_steps 20
```

<!-- tinyseek-nav -->

---

Previous: [Training Loop](16_training_loop_from_config_to_checkpoint.md) | [Tutorial Index](README.md) | Next: [Stage 0: Dense Baseline](02_stage0_dense_baseline.md)
