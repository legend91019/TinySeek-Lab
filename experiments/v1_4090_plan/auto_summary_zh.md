# v1 自动汇总表

这些表由 `scripts/generate_v1_report_assets.py` 从成本 CSV 和 mini-eval JSON 自动生成。

## Core Metrics

| Run | Steps | Params | Activated | Peak GB | GPU h | Cost CNY | Val loss | Eval loss | PPL | Add acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_tiny_base` | 200 | 2.41M | 2.41M | 0.21 | 0.0029 | 0.0063 | 1.1739 | 1.1981 | 3.3137 | 0.00 |
| `v1_dense35` | 200 | 33.70M | 33.70M | 2.64 | 0.0047 | 0.0102 | 0.9962 | 1.0142 | 2.7572 | 0.00 |
| `v1_dense115` | 100 | 113.47M | 113.47M | 3.95 | 0.0076 | 0.0167 | 1.6393 | 1.6582 | 5.2500 | 0.00 |
| `v1_sweep_bs16_lr3e-4` | 1221 | 33.70M | 33.70M | 1.55 | 0.0138 | 0.0302 | 0.6547 | 0.6560 | 1.9270 | 0.00 |
| `v1_sweep_bs16_lr6e-4` | 1221 | 33.70M | 33.70M | 1.55 | 0.0144 | 0.0315 | 0.6633 | 0.6669 | 1.9482 | 0.00 |
| `v1_sweep_bs32_lr3e-4` | 611 | 33.70M | 33.70M | 2.64 | 0.0090 | 0.0197 | 0.6946 | 0.7010 | 2.0157 | 0.00 |
| `v1_sweep_bs32_lr6e-4` | 611 | 33.70M | 33.70M | 2.64 | 0.0110 | 0.0239 | 0.6920 | 0.6978 | 2.0093 | 0.00 |
| `v1_moe_activated35` | 100 | 235.06M | 84.06M | 5.46 | 0.0138 | 0.0302 | 1.6796 | 1.7009 | 5.4791 | 0.00 |
| `v1_tiny_mla` | 200 | 2.26M | 2.26M | 0.21 | 0.0031 | 0.0067 | 1.6156 | 1.6628 | 5.2741 | 0.00 |
| `v1_tiny_sft` | 120 | 2.41M | 2.41M | 0.15 | 0.0019 | 0.0042 | 0.1999 | 2.8388 | 17.0945 | 0.00 |
| `v1_tiny_grpo` | 30 | 2.41M | 2.41M | 0.14 | 0.0044 | 0.0095 | N/A | 2.9111 | 18.3771 | 0.00 |

## 自动生成图表

![PPL comparison](figures/v1_ppl.svg)

![Peak VRAM comparison](figures/v1_peak_vram.svg)

![GPU cost comparison](figures/v1_cost.svg)

![Sweep comparison](figures/v1_sweep_val_loss.svg)

![VRAM versus PPL](figures/v1_vram_vs_ppl.svg)
