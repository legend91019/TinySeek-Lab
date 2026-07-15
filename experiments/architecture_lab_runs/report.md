# TinySeek Architecture Lab GPU Measurements

This report is generated from cost JSON files. PPL is computed only from main `val_lm_loss`; means and standard deviations use repeated seeds of the same config.

- GPU runs: `48`
- Configurations: `16`
- Tracked trainer-process time: `1.5679 h`
- Corresponding estimated cost: `3.4180 CNY`

## Environment and Data

- GPU: `NVIDIA GeForce RTX 4090`; PyTorch: `2.8.0+cu128`; CUDA: `12.8`.
- Rate: `2.18 CNY/h`. The ledger tracks trainer-process time and excludes data preparation, report generation, and idle rental time; estimated cost is not the platform invoice.
- Data: `data/tinystories.jsonl`, `50000` lines, `47726046` bytes, SHA256 `fa16741c6cefdf24c361c6cd75c74f8e6ab500c6bde7adc942bb6afbf8b69814`.
- Per-run configs are in [`configs/`](configs/); raw cost and history ledgers are in [`raw/`](raw/).

## Reproduce

```bash
python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl
python scripts/run_architecture_lab.py --data data/tinystories.jsonl --seeds 42,43,44 --hourly_rate 2.18 --currency CNY
python scripts/generate_architecture_report.py
```

## Decision Summary

| Stage | Multi-seed RTX 4090 observation | Gate and decision |
| --- | --- | --- |
| MHA -> GQA | PPL `2.017 -> 2.006`; theoretical KV/token `384 -> 192`. | Pass: cache elements halve without a PPL regression at this budget; use GQA in the next baseline. |
| coarse -> fine/shared MoE | PPL `2.071/2.081/2.039`; fine/shared are also slower. | Split decision: fine-grained alone fails; shared passes on quality but is about 35% slower, so use shared for quality and retain coarse for throughput. |
| Auxiliary weight | `arch_moe_aux` has the lowest PPL, `2.009`; load CV falls from `0.342` without aux to `0.075` at aux=0.01. | Choose aux=0.01 for this quality/load trade-off; stronger aux adds no quality gain. |
| aux -> bias routing | PPL `2.009 -> 2.024`; load CV `0.075 -> 0.081`. | Fail: bias improves neither main-task quality nor load balance here; retain aux and sweep the bias update rate separately. |
| GQA -> educational MLA | Theoretical KV/token `192 -> 72`, but PPL `2.009 -> 2.194`. | Fail the quality gate: retain the cache-compression hypothesis and sweep latent rank; training VRAM is not decoding-cache evidence. |
| MTP off -> on | PPL `2.207 +/- 0.017` vs `2.196 +/- 0.034`; peak VRAM `0.197 -> 0.246 GB`. | Inconclusive only on the rejected V3-style branch; this does not decide MTP for the GQA+aux branch. |

Branch note: these are controlled comparisons within groups, not one model that stacks every winner. The coarse/fine/shared group and the 8-routed top-2 aux/MLA/V3-style mechanism group use different MoE topologies; the MTP result applies only to the rejected V3-style branch.

## Multi-Seed Aggregate

| Run | Seeds | val LM loss | PPL | tokens/s | peak GB | GPU h | cost CNY | load CV | KV/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arch_dense_gqa` | 3 | 0.6961 +/- 0.0006 | 2.006 +/- 0.001 | 70201 | 0.159 | 0.0159 | 0.0346 | 0.000 | 192 |
| `arch_dense_mha` | 3 | 0.7015 +/- 0.0026 | 2.017 +/- 0.005 | 73678 | 0.161 | 0.0150 | 0.0327 | 0.000 | 384 |
| `arch_moe_aux` | 3 | 0.6979 +/- 0.0055 | 2.009 +/- 0.011 | 30459 | 0.194 | 0.0362 | 0.0790 | 0.075 | 192 |
| `arch_moe_aux_none` | 3 | 0.7030 +/- 0.0039 | 2.020 +/- 0.008 | 31450 | 0.194 | 0.0352 | 0.0767 | 0.342 | 192 |
| `arch_moe_aux_strong` | 3 | 0.7013 +/- 0.0048 | 2.016 +/- 0.010 | 30641 | 0.194 | 0.0361 | 0.0787 | 0.056 | 192 |
| `arch_moe_aux_weak` | 3 | 0.6994 +/- 0.0037 | 2.013 +/- 0.007 | 30184 | 0.194 | 0.0365 | 0.0796 | 0.166 | 192 |
| `arch_moe_bias` | 3 | 0.7052 +/- 0.0049 | 2.024 +/- 0.010 | 30997 | 0.194 | 0.0351 | 0.0765 | 0.081 | 192 |
| `arch_moe_coarse` | 3 | 0.7278 +/- 0.0084 | 2.071 +/- 0.017 | 50975 | 0.188 | 0.0239 | 0.0520 | 0.028 | 192 |
| `arch_moe_fine_grained` | 3 | 0.7328 +/- 0.0047 | 2.081 +/- 0.010 | 32506 | 0.212 | 0.0345 | 0.0751 | 0.057 | 192 |
| `arch_moe_shared` | 3 | 0.7123 +/- 0.0039 | 2.039 +/- 0.008 | 33246 | 0.209 | 0.0337 | 0.0735 | 0.056 | 192 |
| `arch_v2_attention_control` | 3 | 0.6979 +/- 0.0055 | 2.009 +/- 0.011 | 29773 | 0.194 | 0.0363 | 0.0791 | 0.075 | 192 |
| `arch_v2_low_rank_control` | 3 | 0.6979 +/- 0.0055 | 2.009 +/- 0.011 | 29639 | 0.194 | 0.0364 | 0.0793 | 0.075 | 192 |
| `arch_v2_low_rank_kv` | 3 | 0.7839 +/- 0.0005 | 2.190 +/- 0.001 | 30132 | 0.191 | 0.0358 | 0.0781 | 0.067 | 192 |
| `arch_v2_mla` | 3 | 0.7856 +/- 0.0054 | 2.194 +/- 0.012 | 29721 | 0.197 | 0.0364 | 0.0794 | 0.060 | 72 |
| `arch_v3_mtp` | 3 | 0.7865 +/- 0.0156 | 2.196 +/- 0.034 | 27044 | 0.246 | 0.0403 | 0.0878 | 0.063 | 72 |
| `arch_v3_no_mtp` | 3 | 0.7916 +/- 0.0076 | 2.207 +/- 0.017 | 30972 | 0.197 | 0.0354 | 0.0772 | 0.079 | 72 |

## Compute Ledger

| Run | Train tokens | Rough FLOPs |
| --- | ---: | ---: |
| `arch_dense_gqa` | 2,048,000 | 2.781e+13 |
| `arch_dense_mha` | 2,048,000 | 2.963e+13 |
| `arch_moe_aux` | 2,048,000 | 2.245e+13 |
| `arch_moe_aux_none` | 2,048,000 | 2.245e+13 |
| `arch_moe_aux_strong` | 2,048,000 | 2.245e+13 |
| `arch_moe_aux_weak` | 2,048,000 | 2.245e+13 |
| `arch_moe_bias` | 2,048,000 | 2.245e+13 |
| `arch_moe_coarse` | 2,048,000 | 2.783e+13 |
| `arch_moe_fine_grained` | 2,048,000 | 2.789e+13 |
| `arch_moe_shared` | 2,048,000 | 2.788e+13 |
| `arch_v2_attention_control` | 2,048,000 | 2.245e+13 |
| `arch_v2_low_rank_control` | 2,048,000 | 2.245e+13 |
| `arch_v2_low_rank_kv` | 2,048,000 | 2.185e+13 |
| `arch_v2_mla` | 2,048,000 | 2.187e+13 |
| `arch_v3_mtp` | 2,048,000 | 2.944e+13 |
| `arch_v3_no_mtp` | 2,048,000 | 2.187e+13 |

## Individual Runs

| Run | Seed | val LM loss | PPL | tokens/s | peak GB | GPU h | cost CNY |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arch_dense_gqa` | 42 | 0.6953 | 2.004 | 70801 | 0.159 | 0.0157 | 0.0342 |
| `arch_dense_gqa` | 43 | 0.6969 | 2.008 | 69083 | 0.159 | 0.0160 | 0.0349 |
| `arch_dense_gqa` | 44 | 0.6962 | 2.006 | 70719 | 0.159 | 0.0160 | 0.0348 |
| `arch_dense_mha` | 42 | 0.7044 | 2.023 | 73705 | 0.161 | 0.0150 | 0.0327 |
| `arch_dense_mha` | 43 | 0.7020 | 2.018 | 75017 | 0.161 | 0.0150 | 0.0326 |
| `arch_dense_mha` | 44 | 0.6981 | 2.010 | 72311 | 0.161 | 0.0150 | 0.0327 |
| `arch_moe_aux` | 42 | 0.6955 | 2.005 | 30977 | 0.194 | 0.0351 | 0.0766 |
| `arch_moe_aux` | 43 | 0.7054 | 2.025 | 29758 | 0.194 | 0.0382 | 0.0834 |
| `arch_moe_aux` | 44 | 0.6927 | 1.999 | 30641 | 0.194 | 0.0353 | 0.0769 |
| `arch_moe_aux_none` | 42 | 0.6979 | 2.010 | 30994 | 0.194 | 0.0352 | 0.0768 |
| `arch_moe_aux_none` | 43 | 0.7073 | 2.029 | 31227 | 0.194 | 0.0356 | 0.0775 |
| `arch_moe_aux_none` | 44 | 0.7038 | 2.021 | 32128 | 0.194 | 0.0348 | 0.0758 |
| `arch_moe_aux_strong` | 42 | 0.7004 | 2.014 | 28976 | 0.194 | 0.0368 | 0.0803 |
| `arch_moe_aux_strong` | 43 | 0.7076 | 2.029 | 29383 | 0.194 | 0.0375 | 0.0819 |
| `arch_moe_aux_strong` | 44 | 0.6959 | 2.006 | 33565 | 0.194 | 0.0339 | 0.0739 |
| `arch_moe_aux_weak` | 42 | 0.6973 | 2.008 | 28540 | 0.194 | 0.0371 | 0.0809 |
| `arch_moe_aux_weak` | 43 | 0.7046 | 2.023 | 31192 | 0.194 | 0.0365 | 0.0795 |
| `arch_moe_aux_weak` | 44 | 0.6964 | 2.007 | 30819 | 0.194 | 0.0359 | 0.0784 |
| `arch_moe_bias` | 42 | 0.7041 | 2.022 | 34158 | 0.194 | 0.0319 | 0.0696 |
| `arch_moe_bias` | 43 | 0.7117 | 2.037 | 28398 | 0.194 | 0.0380 | 0.0828 |
| `arch_moe_bias` | 44 | 0.6998 | 2.013 | 30436 | 0.194 | 0.0354 | 0.0772 |
| `arch_moe_coarse` | 42 | 0.7171 | 2.049 | 51775 | 0.189 | 0.0234 | 0.0510 |
| `arch_moe_coarse` | 43 | 0.7377 | 2.091 | 49042 | 0.188 | 0.0243 | 0.0529 |
| `arch_moe_coarse` | 44 | 0.7285 | 2.072 | 52109 | 0.188 | 0.0239 | 0.0521 |
| `arch_moe_fine_grained` | 42 | 0.7369 | 2.090 | 29554 | 0.212 | 0.0378 | 0.0824 |
| `arch_moe_fine_grained` | 43 | 0.7353 | 2.086 | 35852 | 0.212 | 0.0313 | 0.0681 |
| `arch_moe_fine_grained` | 44 | 0.7262 | 2.067 | 32113 | 0.212 | 0.0344 | 0.0749 |
| `arch_moe_shared` | 42 | 0.7081 | 2.030 | 30472 | 0.209 | 0.0365 | 0.0795 |
| `arch_moe_shared` | 43 | 0.7114 | 2.037 | 34722 | 0.209 | 0.0322 | 0.0702 |
| `arch_moe_shared` | 44 | 0.7175 | 2.049 | 34543 | 0.209 | 0.0324 | 0.0707 |
| `arch_v2_attention_control` | 42 | 0.6955 | 2.005 | 29764 | 0.194 | 0.0377 | 0.0821 |
| `arch_v2_attention_control` | 43 | 0.7054 | 2.025 | 29802 | 0.194 | 0.0356 | 0.0776 |
| `arch_v2_attention_control` | 44 | 0.6927 | 1.999 | 29752 | 0.194 | 0.0356 | 0.0776 |
| `arch_v2_low_rank_control` | 42 | 0.6955 | 2.005 | 30321 | 0.194 | 0.0363 | 0.0792 |
| `arch_v2_low_rank_control` | 43 | 0.7054 | 2.025 | 27602 | 0.194 | 0.0376 | 0.0820 |
| `arch_v2_low_rank_control` | 44 | 0.6927 | 1.999 | 30995 | 0.194 | 0.0352 | 0.0768 |
| `arch_v2_low_rank_kv` | 42 | 0.7833 | 2.189 | 27304 | 0.191 | 0.0375 | 0.0818 |
| `arch_v2_low_rank_kv` | 43 | 0.7839 | 2.190 | 30528 | 0.191 | 0.0356 | 0.0776 |
| `arch_v2_low_rank_kv` | 44 | 0.7845 | 2.191 | 32565 | 0.191 | 0.0344 | 0.0750 |
| `arch_v2_mla` | 42 | 0.7933 | 2.211 | 30594 | 0.197 | 0.0358 | 0.0780 |
| `arch_v2_mla` | 43 | 0.7821 | 2.186 | 29117 | 0.197 | 0.0366 | 0.0799 |
| `arch_v2_mla` | 44 | 0.7814 | 2.184 | 29451 | 0.197 | 0.0369 | 0.0804 |
| `arch_v3_mtp` | 42 | 0.7722 | 2.165 | 27319 | 0.246 | 0.0410 | 0.0894 |
| `arch_v3_mtp` | 43 | 0.7791 | 2.179 | 26455 | 0.246 | 0.0403 | 0.0878 |
| `arch_v3_mtp` | 44 | 0.8082 | 2.244 | 27358 | 0.246 | 0.0395 | 0.0861 |
| `arch_v3_no_mtp` | 42 | 0.7994 | 2.224 | 29300 | 0.197 | 0.0373 | 0.0813 |
| `arch_v3_no_mtp` | 43 | 0.7941 | 2.212 | 30975 | 0.197 | 0.0348 | 0.0758 |
| `arch_v3_no_mtp` | 44 | 0.7813 | 2.184 | 32641 | 0.197 | 0.0341 | 0.0744 |

## Evidence Boundary

These are TinySeek small-model GPU measurements, not substitutes for DeepSeek paper-scale results. Inspect means, variation, throughput, memory, and routing load before a decision.

## Figures

![Architecture PPL](figures/architecture_ppl.svg)

![Architecture throughput](figures/architecture_throughput.svg)

![Architecture VRAM](figures/architecture_vram.svg)

![MoE load CV](figures/moe_load_cv.svg)
