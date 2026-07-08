# RTX 4090 v1 正式实验结果

这是 TinySeek-Lab 第一次在付费 RTX 4090 上跑完整 v1 路线。

这次目标不是追求模型质量，而是验证仓库能不能在真实文本数据上完整跑通教程闭环：

```text
TinyStories -> tiny base -> dense 35M/115M -> LR/batch sweep
-> MoE -> MLA -> SFT -> GRPO mini -> mini eval -> 成本汇总
```

## 环境

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-07-08 |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| Driver / CUDA | Driver 580.76.05, CUDA 13.0 |
| PyTorch | 2.8.0+cu128 |
| 数据 | `roneneldan/TinyStories`，50,000 条 JSONL |
| 数据访问说明 | `huggingface.co` 不通；设置 `HF_ENDPOINT=https://hf-mirror.com` 后可下载 |
| 单价 | 2.18 元/小时 |

训练命令：

```bash
python scripts/run_4090_v1.py --execute --skip_data_prepare --hourly_rate 2.18
```

数据准备单独通过 Hugging Face 镜像完成。

## 成本汇总

| Run | Steps | Params | Activated Params | Tokens | 峰值显存 GB | GPU 小时 | 成本 | Val Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `v1_tiny_base` | 200 | 2.41M | 2.41M | 0.41M | 0.21 | 0.0029 | 0.0063 元 | 1.1739 |
| `v1_dense35` | 200 | 33.70M | 33.70M | 1.64M | 2.64 | 0.0047 | 0.0102 元 | 0.9962 |
| `v1_dense115` | 100 | 113.47M | 113.47M | 0.41M | 3.95 | 0.0076 | 0.0167 元 | 1.6393 |
| `v1_sweep_bs16_lr3e-4` | 1221 | 33.70M | 33.70M | 5.00M | 1.55 | 0.0138 | 0.0302 元 | 0.6547 |
| `v1_sweep_bs16_lr6e-4` | 1221 | 33.70M | 33.70M | 5.00M | 1.55 | 0.0144 | 0.0315 元 | 0.6633 |
| `v1_sweep_bs32_lr3e-4` | 611 | 33.70M | 33.70M | 5.01M | 2.64 | 0.0090 | 0.0197 元 | 0.6946 |
| `v1_sweep_bs32_lr6e-4` | 611 | 33.70M | 33.70M | 5.01M | 2.64 | 0.0110 | 0.0239 元 | 0.6920 |
| `v1_moe_activated35` | 100 | 235.06M | 84.06M | 0.41M | 5.46 | 0.0138 | 0.0302 元 | 1.6796 |
| `v1_tiny_mla` | 200 | 2.26M | 2.26M | 0.41M | 0.21 | 0.0031 | 0.0067 元 | 1.6156 |
| `v1_tiny_sft` | 120 | 2.41M | 2.41M | 0.37M | 0.15 | 0.0019 | 0.0042 元 | 0.1999 |
| `v1_tiny_grpo` | 30 | 2.41M | 2.41M | 0 | 0.14 | 0.0044 | 0.0095 元 | N/A |

总计：

- GPU 小时：0.0867
- 估算成本：0.19 元

完整机器可读结果在 [`experiments/v1_4090_plan`](v1_4090_plan/)。

## Mini Eval

| Run | Eval Loss | PPL | 加法准确率 | 格式分 |
| --- | ---: | ---: | ---: | ---: |
| `v1_tiny_base` | 1.1981 | 3.3137 | 0.0 | 1.0 |
| `v1_dense35` | 1.0142 | 2.7572 | 0.0 | 1.0 |
| `v1_dense115` | 1.6582 | 5.2500 | 0.0 | 1.0 |
| `v1_sweep_bs16_lr3e-4` | 0.6560 | 1.9270 | 0.0 | 1.0 |
| `v1_sweep_bs16_lr6e-4` | 0.6669 | 1.9482 | 0.0 | 1.0 |
| `v1_sweep_bs32_lr3e-4` | 0.7010 | 2.0157 | 0.0 | 1.0 |
| `v1_sweep_bs32_lr6e-4` | 0.6978 | 2.0093 | 0.0 | 1.0 |
| `v1_moe_activated35` | 1.7009 | 5.4791 | 0.0 | 1.0 |
| `v1_tiny_mla` | 1.6628 | 5.2741 | 0.0 | 1.0 |
| `v1_tiny_sft` | 2.8388 | 17.0945 | 0.0 | 1.0 |
| `v1_tiny_grpo` | 2.9111 | 18.3771 | 0.0 | 1.0 |

## 结论

1. 仓库完整 v1 流程已经能在 RTX 4090 上跑通。
2. 固定约 5M token 时，`bs16_lr3e-4` 在 validation loss 和 mini-eval perplexity 上最好。
3. 35M dense 短训优于 115M dense 短训，因为 115M 的 token budget 太少。这是一个很好的教学点：模型更大不代表在训练不足时一定更好。
4. MoE 可以轻松放进 4090：总参数 235M，激活参数约 84M，峰值 allocated 显存约 5.46 GB。
5. 教学版 MLA 跑通了，但它不是 DeepSeek-V2 的生产级 MLA 复刻，只适合讲 KV latent 思想。
6. SFT 明显降低 toy SFT validation loss，但会损害 TinyStories 泛化困惑度。这符合预期，因为 SFT 数据很小且偏格式。
7. GRPO mini 有非零 reward，最后 `mean_reward` 到 0.15，但没有解出加法 mini eval。它应该作为算法形状教学，而不是严肃 RL 性能结果。

## 下一步

- 在 GRPO 前加入更强的小型算术 SFT 数据。
- 写一个自动报告生成器，合并 `cost_summary.csv` 和 `eval_*.json`。
- 增加 MoE routing histogram 和 expert-load 汇总。
- 如果要产出更稳定的教程 checkpoint，优先加长 35M dense baseline。
