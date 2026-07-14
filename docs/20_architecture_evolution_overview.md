# 20. DeepSeek Architecture Evolution as One Code Path

This chapter is the map for the code-first course. The next chapters keep one decoder-only language model intact and change only the subsystem introduced by each paper. Input IDs, embeddings, residual connections, the LM head, and next-token loss remain visible throughout.

## What You Will Build

| Generation | Complete teaching model | Only new change | Problem addressed |
| --- | --- | --- | --- |
| DeepSeek LLM | [`stage0_deepseek_llm.py`](../model/stages/stage0_deepseek_llm.py) | starting point | a stable Dense decoder-only LM |
| DeepSeekMoE | [`stage1_deepseek_moe.py`](../model/stages/stage1_deepseek_moe.py) | Dense FFN to routed and shared experts | grow capacity without activating every parameter |
| DeepSeek-V2 | [`stage2_deepseek_v2.py`](../model/stages/stage2_deepseek_v2.py) | MHA/GQA to educational MLA | compress the growing KV cache |
| DeepSeek-V3 | [`stage3_deepseek_v3.py`](../model/stages/stage3_deepseek_v3.py) | routing bias and MTP | reduce balance-loss interference and add future-token supervision |

Formal experiments still use [`model/tinyseek.py`](../model/tinyseek.py). Stage files teach the complete code path; the unified model provides configuration-controlled comparisons.

## The Backbone That Does Not Change

```text
input_ids [B, T]
  -> token embedding [B, T, D]
  -> N x Transformer block [B, T, D]
  -> final RMSNorm [B, T, D]
  -> tied LM head [B, T, V]
  -> shifted next-token cross entropy
```

When comparing adjacent stages, ask three questions: did attention change, did the FFN change, or did the training objective change? Everything else should remain stable.

## Actual Research Order

```mermaid
flowchart LR
  LLM["DeepSeek LLM<br/>Dense + modern block"] --> MOE["DeepSeekMoE<br/>fine-grained + shared experts"]
  MOE --> V2["DeepSeek-V2<br/>DeepSeekMoE + MLA"]
  V2 --> V3["DeepSeek-V3<br/>bias balance + MTP"]
  V3 --> R1["DeepSeek-R1<br/>cold-start SFT + GRPO pipeline"]
```

R1 is primarily a training-pipeline evolution, not a new Transformer block. R1-Zero applies RL directly to a pretrained base model. The full R1 route adds cold-start SFT, RL, rejection sampling, another SFT stage, and later RL. See [`19_posttraining_code_walkthrough.md`](19_posttraining_code_walkthrough.md).

## From a Paper List to Experiment-Driven Upgrades

The next architecture is not treated as correct in advance. Every transition follows the same research loop:

```mermaid
flowchart LR
  A["Run the previous baseline"] --> B["Record a measurable bottleneck"]
  B --> C["State a research hypothesis"]
  C --> D["Implement the smallest candidate change"]
  D --> E["Run a matched ablation"]
  E --> F{"Pass the decision gate?"}
  F -- "yes" --> G["Upgrade and record the cost"]
  F -- "no" --> H["Keep the baseline or revise the hypothesis"]
```

Historical discipline matters. Public papers tell us which problems DeepSeek addressed and which ablations it reported. They usually do not prove that an internal team first saw one public table and only then invented a method. TinySeek therefore reconstructs a **testable research path**, not an undocumented invention story.

## Four Research Cards

| Stage | Measurable bottleneck | Research hypothesis | Matched experiment | Decision gate | Evidence status |
| --- | --- | --- | --- | --- | --- |
| Dense recipe | Loss is sensitive to LR and batch at a fixed token budget | A stable recipe is required before architecture comparisons | LR by batch sweep | choose a stable region with the lowest validation LM loss | measured in v1; a formal multi-seed run remains useful |
| Dense -> DeepSeekMoE | Wider Dense FFNs couple capacity to per-token compute | Fine-grained routing adds combinations; shared experts carry common knowledge | coarse MoE -> fine-grained MoE -> shared isolation | no material LM-loss regression, no routing collapse, and correct capacity/activation accounting | code and configs ready; GPU evidence pending |
| MoE -> V2 | GQA cache still grows linearly with layers, context, and batch | A low-rank KV latent plus decoupled RoPE can reduce cached state | GQA -> naive low-rank KV -> educational MLA | materially fewer theoretical cache elements without significant PPL regression; actual throughput waits for cached decoding | theory check available, quality run pending, real cache kernel absent |
| V2 -> V3 | Auxiliary loss may interfere with LM learning; one-token prediction gives limited supervision | Selection bias can control load without changing affinity weights; MTP adds farther supervision | aux vs bias; MTP off vs on | no routing collapse, main LM/PPL no worse, and acceptable added cost | code and design ready; GPU evidence pending |

Write the decision gate before running an experiment. Changing the success criterion after seeing numbers makes almost any result look supportive. A single seed is a debugging and hypothesis-forming run; a formal claim should report means and variation over multiple seeds.

## Why Each Generation Appeared

### DeepSeek LLM

The first public DeepSeek LLM already uses a modern LLaMA-style design: Pre-Norm, RMSNorm, RoPE, SwiGLU, and GQA for the 67B model. Its paper also studies batch size, learning rate, and compute scaling. It is the Dense baseline, not an intentionally outdated Transformer.

### DeepSeekMoE

A Dense FFN activates the same parameters for every token. MoE stores more FFNs but routes each token to only a few. DeepSeekMoE further emphasizes fine-grained experts and shared-expert isolation so common knowledge and routed specialization have different paths.

### DeepSeek-V2

Autoregressive decoding caches K/V states for all previous tokens. MLA compresses cacheable information into a low-rank latent, reconstructs content K/V for attention, and treats the RoPE position path separately.

### DeepSeek-V3

A strong auxiliary balance loss can compete with the language-model objective. V3 instead adjusts expert selection with a bias that does not change affinity weights. It also adds Multi-Token Prediction modules to supervise farther future tokens.

## Three Levels of Evidence

| Level | Meaning |
| --- | --- |
| Paper result | measured by DeepSeek at its data, model, and infrastructure scale |
| TinySeek implementation | code path, shapes, losses, and statistics are runnable |
| TinySeek measured or pending | this repository's own comparison; missing runs stay explicitly pending |

Implementing an equation is not reproducing paper performance. Educational MLA demonstrates the latent path and theoretical cache accounting, but it is not a production cached-decoding kernel.

## Inspect All Four Models

With PyTorch installed:

```bash
python tests/stage_models_test.py
python scripts/inspect_stage_models.py --out out/stage_model_inspection.json
```

The inspection reports logits shape, total and activated parameters, theoretical per-layer KV-cache elements, and available loss fields. For quality and cost comparisons, use [`configs/architecture_lab/`](../configs/architecture_lab/) and the [architecture experiment plan](../experiments/06_architecture_evolution_plan.md).

## Learning Loop

For every generation:

1. Run the previous stage and record shapes.
2. Read the new config fields.
3. Locate the replaced sublayer and new outputs.
4. Run a matched configuration that changes one variable.
5. Record benefit, cost, and failure cases together.

The next code chapter starts with the FFN: how one token reaches two experts, and why total parameters are not the same as activated parameters.

<!-- tinyseek-nav -->

---

Previous: [DeepSeek LM Paper Map](01_deepseek_lm_paper_map.md) | [Tutorial Index](README.md) | Next: [Code First Dense LM](12_code_first_dense_lm.md)
