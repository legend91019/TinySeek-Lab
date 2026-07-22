# s01 Dense LM: Build the Whole Model First

[中文](README_zh.md) | English | [Course index](../README.md)

## Research Question

Before discussing MoE or MLA, can we write a complete next-token language model whose external interface stays stable while its internal blocks are readable?

## Paper Clue

DeepSeek LLM is a useful starting point because its training recipe makes the base model explicit: a decoder-only LM, modern normalization and positional encoding, SwiGLU, and grouped attention. TinySeek keeps the educational version small; this is not a claim that the paper's full-scale recipe can be copied unchanged.

## Code Change

Start at [`model/stages/stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py), then trace the complete path:

```text
input_ids [B,T] -> Embedding [B,T,D] -> N x Block
-> RMSNorm -> tied LM head [B,T,V] -> shifted cross entropy
```

Read the formula-to-API mapping in [`docs/24_math_to_pytorch.md`](../../docs/24_math_to_pytorch.md), and the longer end-to-end walkthrough in [`docs/12_code_first_dense_lm.md`](../../docs/12_code_first_dense_lm.md).

The important lesson is the contract: each block returns `[B,T,D]`; only the final head changes the last dimension to `V`. That contract is what lets later units replace one sublayer at a time.

## Run Before Changing Anything

```bash
python tests/stage_models_test.py
python scripts/inspect_stage_models.py
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 20
```

## Experiment Card

| Item | TinySeek choice |
| --- | --- |
| Baseline | `dense_gqa` |
| Main metrics | validation LM loss/PPL, tokens/s, peak VRAM |
| Invariants | logits `[B,T,V]`, causal shift, stable checkpoint interface |
| Gate | Do not compare later architectures until this path trains and the metrics are logged |

## Decision

This unit does not try to prove a new architecture. It establishes the control group. Every later result is meaningful only if it uses the same data split, token budget, evaluation code and accounting conventions.

![Architecture PPL evidence](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

## Code Exercise

Change one dimension in the config and predict every affected shape before running. Then inspect `RMSNorm.forward`, `Attention.forward`, `SwiGLU.forward`, `Block.forward`, and `DenseCausalLM.forward` in that order. Do not start from the class names; start from the tensor contract.

## Next

The model is runnable, but the training recipe is still a confounder. Before inventing architecture, choose a reproducible LR and batch-size region in [s02 Training recipe](../s02_training_recipe/README.md).

<!-- tinyseek-nav -->

Previous: [Course index](../README.md) | Next: [s02 Training recipe](../s02_training_recipe/README.md)
