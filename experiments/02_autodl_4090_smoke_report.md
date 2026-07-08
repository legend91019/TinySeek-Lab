# AutoDL RTX 4090 Smoke Report

Date: 2026-07-08

This report records the first remote GPU validation run for TinySeek-Lab. These
runs use the toy byte-level dataset, so they are hardware and pipeline checks,
not model-quality claims.

## Hardware

- Provider: AutoDL
- GPU: NVIDIA GeForce RTX 4090
- Driver: 570.124.04
- CUDA runtime: 12.8
- PyTorch: 2.8.0+cu128
- Python: 3.12.3
- Price used for accounting: 2.18 CNY/hour

## Runs

| Run | Params | Activated params | Steps | Peak allocated GB | Peak reserved GB | Val loss | Cost CNY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bs8_lr3e-4 | 2,410,944 | 2,410,944 | 80 | 0.13 | 0.15 | 2.4921 | 0.0008 |
| bs8_lr6e-4 | 2,410,944 | 2,410,944 | 80 | 0.13 | 0.15 | 1.7752 | 0.0009 |
| bs16_lr3e-4 | 2,410,944 | 2,410,944 | 80 | 0.21 | 0.26 | 2.4227 | 0.0008 |
| bs16_lr6e-4 | 2,410,944 | 2,410,944 | 80 | 0.21 | 0.26 | 1.6009 | 0.0008 |
| tiny_dense | 2,410,944 | 2,410,944 | 200 | 0.21 | 0.26 | 0.0757 | 0.0016 |
| tiny_mla | 2,263,488 | 2,263,488 | 200 | 0.21 | 0.26 | 0.1532 | 0.0016 |
| tiny_moe | 16,572,864 | 5,956,032 | 200 | 0.59 | 0.75 | 0.1046 | 0.0059 |
| medium_dense_35m | 33,696,256 | 33,696,256 | 200 | 3.80 | 4.01 | 0.0592 | 0.0096 |
| medium_dense_115m | 113,465,088 | 113,465,088 | 100 | 5.12 | 5.28 | 0.0708 | 0.0088 |
| medium_moe_activated_35m | 235,055,616 | 84,060,672 | 100 | 6.67 | 7.19 | 0.1512 | 0.0101 |

Total measured GPU time: 0.0188 GPU hours.

Total measured training cost: about 0.04 CNY.

## Takeaways

- The selected PyTorch 2.8.0 + CUDA 12.8 image works for the repo.
- The cost telemetry writes valid JSON summaries on CUDA.
- A 4090 has enough room for at least a 115M dense model in the current fp32
  training path.
- The 235M-total-parameter MoE config also fits, with about 84M activated
  parameters and 6.67 GB peak allocated memory.
- The current toy dataset is too small. It is useful for smoke tests, but it
  overfits quickly and should not be used for real model-quality conclusions.

## Next Step

Before burning long GPU hours, add a real text-corpus pipeline and then repeat:

1. Dense 35M training for a real token budget.
2. Dense 115M training for a shorter budget.
3. MoE with matched activated parameters.
4. LR/batch search on the selected model scale.
