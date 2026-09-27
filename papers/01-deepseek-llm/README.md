# DeepSeek LLM: ask the training questions first

English | [中文](README_zh.md) | [Paper route](../README.md)

Paper: *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* (2024, [arXiv:2401.02954](https://arxiv.org/abs/2401.02954)). The paper treats scaling behavior, not just a fixed model checkpoint, as the research problem.

## 1. Position

DeepSeek LLM studies batch-size and learning-rate scaling, model/data allocation under compute, and the effect of data quality. It applies those findings to 7B and 67B models, using a LLaMA-like decoder architecture, GQA at the larger scale, 2T tokens, and SFT/DPO for chat models.

## 2. What changed

The paper combines aggressive deduplication and filtering, a BBPE tokenizer, Pre-Norm RMSNorm/SwiGLU/RoPE blocks, GQA, and a multi-step learning-rate schedule. It uses IsoFLOP profiles and non-embedding FLOPs/token rather than parameter count alone. TinySeek turns this into a small recipe sweep and a readable Dense/GQA baseline.

## 3. Method

The data pipeline has deduplication, quality filtering and domain remixing. The architecture is decoder-only next-token prediction with RMSNorm, SwiGLU and RoPE. The 67B model uses 64 query heads and 8 KV heads. AdamW, warmup, multi-step decay, clipping and bf16/fp32 accumulation are part of the recipe. The scaling analysis first studies hyperparameters, then fits model/data allocation and warns that data quality changes the fitted law.

## 4. Paper experiments

![Model specifications](../assets/deepseek-llm/paper-table-2-model-specs.png)

![Scaling curves](../assets/deepseek-llm/paper-fig-3-scaling-curves.png)

![Main results](../assets/deepseek-llm/paper-table-5-main-results.png)

The paper concludes that recipe optimization is necessary for fair scaling comparisons, non-embedding compute is a useful model-scale representation, and better data shifts the preferred allocation toward larger models.

## 5. TinySeek implementation

See [`stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py), [`s02`](../../course/s02_training_recipe/README.md), [`s03`](../../course/s03_gqa/README.md), and [`24_math_to_pytorch.md`](../../docs/24_math_to_pytorch.md). TinySeek does not reproduce HAI-LLM, ZeRO, FlashAttention or 2T-token training.

## 6. TinySeek supplemental evidence

The existing 4090 sweep selected `bs16_lr6e-4` locally. The three-seed GQA comparison reduced theoretical KV elements/token from `384` to `192` while PPL moved `2.017 -> 2.006`.

![TinySeek PPL](../../experiments/architecture_lab_runs/figures/architecture_ppl.svg)

This is **directionally consistent: supplemental check** for the recipe-first method and the GQA cache direction, not a scaling-law or large-model reproduction.

## 7. Claim/evidence table

| Paper claim | Paper evidence | TinySeek evidence | Relation | Boundary |
| --- | --- | --- | --- | --- |
| Study batch size and LR with compute | Figure 3 and scaling section | Four-point sweep | Directionally consistent: supplemental check | No universal law is fitted |
| GQA reduces inference state | Table 2 and architecture section | `384 -> 192` theoretical KV/token | Directionally consistent: supplemental check | No production latency/kernel evidence |
| Data quality changes allocation | Cross-dataset scaling experiments | One TinyStories source | Paper-only | No local data-quality ablation |
| Large benchmark superiority | Table 5 | Toy/mini evaluations | Paper-only | No capability extrapolation |

## 8. Conclusion

DeepSeek LLM establishes the recipe and scaling foundation. TinySeek supports the experimental habit and the local GQA direction, while remaining explicit about scale boundaries.

## 9. Sources and next paper

Paper: [arXiv:2401.02954](https://arxiv.org/abs/2401.02954). Continue to [DeepSeekMoE](../02-deepseek-moe/README.md).
+## Deep reading: what this paper actually establishes

### 1. Fair comparison comes first

The paper is easy to misread as “67B is better than 7B, so scale up.” Section 3 makes the opposite methodological point: if batch size and learning rate are not calibrated first, later model/data comparisons confound optimization mistakes with scaling gains. The authors therefore calibrate hyperparameters, build IsoFLOP profiles, and only then choose the 7B/67B configurations.

### 2. How to read Figure 3

Figure 3 is not a universal law. It is a trend estimate under a finite compute budget, used to narrow the next search space. The transferable lesson is the calibration workflow, not the exact exponent on the plot.

### 3. Why non-embedding FLOPs matters

Parameter count mixes embedding/output parameters with parameters that participate in every token computation. Non-embedding FLOPs/token is a control variable that makes models with different vocabularies more comparable. It prevents a vocabulary-size difference from masquerading as a scaling effect.

### 4. Data quality changes the optimum

The data-scaling experiments imply that high-quality data lets a larger model use extra capacity, while low-quality data can make additional parameters inefficient. The optimal model/data allocation is therefore corpus-dependent. The 89.8% cross-dump deduplication result matters because it changes the effective data distribution, not because the number itself is a universal target.

### 5. Benchmark results are not the scaling proof

Table 5 shows that the resulting models are useful, but cannot by itself prove the scaling fit. The evidence chain is profile, fit, large-model prediction, and prediction error. A high benchmark score and a successful scaling prediction should be evaluated separately.

### 6. Transfer to TinySeek

A local LR/batch sweep answers which recipe works for this data, model, and step budget; it does not estimate compute-optimal scaling. A faithful classroom transfer must hold tokenizer, data mix, optimizer, and token budget fixed while varying one budget axis.
+## Evidence map

| Paper location | Question | Authors' conclusion | Reading boundary |
| --- | --- | --- | --- |
| Section 3.1 / Figure 3 | Do batch and learning rate change with compute? | Trends can narrow the search space | Supports calibration, not a universal law |
| Section 3.2–3.3 | How should model/data scale? | Non-embedding FLOPs and data quality matter | Fit range and corpus limit extrapolation |
| Section 5 / Table 5 | Does the recipe produce useful models? | 7B/67B outperform baselines on many tasks | Does not by itself prove the scaling mechanism |
