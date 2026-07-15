# TinySeek 完整 GPU 训练与后训练报告

本表使用真实 RTX 4090 运行生成。语言模型 PPL 与后训练推理指标分开报告。

- 已记录训练/后训练进程时间：`0.8985 h`
- 按 2.18 CNY/h 对应估算费用：`1.9588 CNY`
- 最佳 LR/batch sweep：`formal_sweep_bs16_lr6e-4`

## 环境与数据

- GPU：`NVIDIA GeForce RTX 4090`；PyTorch：`2.8.0+cu128`；CUDA：`12.8`。
- 计价：`2.18 CNY/h`。账本只统计训练/后训练进程，不含数据准备、独立 mini-eval、报告生成和租卡空闲时间；估算费用不是平台账单。
- 数据：`data/tinystories.jsonl`，`50000` 行，`47726046` bytes，SHA256 `fa16741c6cefdf24c361c6cd75c74f8e6ab500c6bde7adc942bb6afbf8b69814`。
- 数据：`data/reasoning_grpo.jsonl`，`1000` 行，`135336` bytes，SHA256 `fb0e41066e678112d99a3a9ed55d50b4fafbc0870c6b3f24ab68dff4b8f06bd7`。
- 数据：`data/reasoning_sft.jsonl`，`5000` 行，`1016508` bytes，SHA256 `2cc3dff35dd9477dd9783c0accf7052f77e79dcc0c45858893fcf07bbbeea164`。
- 完整数据清单：[`dataset_manifests.json`](dataset_manifests.json)。
- 逐 run 配置在 [`configs/`](configs/)，原始成本与 history 在 [`raw/`](raw/)。

## 复现

```bash
python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl
python scripts/run_gpu_completion.py --data data/tinystories.jsonl --hourly_rate 2.18 --currency CNY
python scripts/generate_gpu_completion_report.py
```

## 实测结论

| 问题 | 4090 观察 | 当前决定 |
| --- | --- | --- |
| 小范围 LR/batch 网格 | `formal_sweep_bs16_lr6e-4` 的 validation loss 最低，为 `0.6475`。 | 只把它当作本预算下的 recipe 选择，不外推为 scaling law。 |
| 容量 pilot | 35M/50M-token loss `0.4822`；115M/30M-token `0.4591`；MoE/30M-token `0.4567`。 | token budget 不同，不能据此宣称架构胜负；保留为成本与可运行性 pilot。 |
| 冷启动与 GRPO | base/direct-GRPO/SFT/SFT+GRPO 的留出题得分依次为 `0/5, 0/5, 0/5, 0/5`（每组 n=`5`）；推理格式分分别为 `0.000/0.000/0.600/0.200`。 | 所有答案均错，这组 mini-eval 没有提供算术泛化证据；当前宽松奖励的 GRPO 还破坏了格式。 |
| 分布代价 | TinyStories 前 `100` 条样本 PPL 从 base 的 `1.718` 上升到 SFT 的 `12.670`，SFT+GRPO 为 `12.306`。 | 窄域后训练会换取格式行为并损害原预训练分布；答案、格式和 PPL 必须分开报告。 |

## 训练与成本

`val PPL = exp(val loss)` 使用各 stage 自己的 validation 数据，只能在相同 stage/data 内比较；`sample PPL` 才是 checkpoint 在同一 TinyStories 前 100 条上的分布探针。

| Run | Stage | Steps | Params | Activated | val loss | val PPL | sample PPL | tok/s | peak GB | GPU h | cost CNY |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `formal_base20m` | pretrain | 3256 | 2.41M | 2.41M | 0.5428 | 1.721 | 1.718 | 223296 | 0.376 | 0.0330 | 0.0720 |
| `formal_dense115_30m` | pretrain | 7325 | 113.47M | 113.47M | 0.4591 | 1.583 | N/A | 77436 | 3.948 | 0.1260 | 0.2746 |
| `formal_dense35_50m` | pretrain | 6104 | 33.70M | 33.70M | 0.4822 | 1.620 | N/A | 219298 | 2.636 | 0.0750 | 0.1635 |
| `formal_direct_grpo` | grpo_mini | 300 | 2.41M | 2.41M | N/A | N/A | 1.995 | N/A | 0.172 | 0.1336 | 0.2913 |
| `formal_moe35_30m` | pretrain | 7325 | 235.06M | 84.06M | 0.4567 | 1.579 | N/A | 31526 | 5.464 | 0.3050 | 0.6649 |
| `formal_reasoning_grpo` | grpo_mini | 300 | 2.41M | 2.41M | N/A | N/A | 12.306 | N/A | 0.172 | 0.1350 | 0.2942 |
| `formal_reasoning_sft` | sft | 2000 | 2.41M | 2.41M | 0.0406 | 1.041 | 12.670 | N/A | 0.225 | 0.0175 | 0.0381 |
| `formal_sweep_bs16_lr3e-4` | pretrain | 1221 | 33.70M | 33.70M | 0.6507 | 1.917 | N/A | 106372 | 1.547 | 0.0255 | 0.0556 |
| `formal_sweep_bs16_lr6e-4` | pretrain | 1221 | 33.70M | 33.70M | 0.6475 | 1.911 | N/A | 108009 | 1.547 | 0.0255 | 0.0556 |
| `formal_sweep_bs32_lr3e-4` | pretrain | 611 | 33.70M | 33.70M | 0.6926 | 1.999 | N/A | 213906 | 2.636 | 0.0114 | 0.0249 |
| `formal_sweep_bs32_lr6e-4` | pretrain | 611 | 33.70M | 33.70M | 0.6819 | 1.978 | N/A | 215064 | 2.636 | 0.0111 | 0.0241 |

## 计算量账本

| Run | 训练 tokens | 粗略 FLOPs |
| --- | ---: | ---: |
| `formal_base20m` | 20,004,864 | 2.894e+14 |
| `formal_dense115_30m` | 30,003,200 | 2.043e+16 |
| `formal_dense35_50m` | 50,003,968 | 1.011e+16 |
| `formal_direct_grpo` | N/A | N/A |
| `formal_moe35_30m` | 30,003,200 | 1.513e+16 |
| `formal_reasoning_grpo` | N/A | N/A |
| `formal_reasoning_sft` | 6,144,000 | 8.888e+13 |
| `formal_sweep_bs16_lr3e-4` | 5,001,216 | 1.011e+15 |
| `formal_sweep_bs16_lr6e-4` | 5,001,216 | 1.011e+15 |
| `formal_sweep_bs32_lr3e-4` | 5,005,312 | 1.012e+15 |
| `formal_sweep_bs32_lr6e-4` | 5,005,312 | 1.012e+15 |

## 后训练推理答案与格式

| Checkpoint | Answer accuracy | Format score | GRPO mean reward |
| --- | ---: | ---: | ---: |
| `formal_base20m` | 0.000 | 0.000 | N/A |
| `formal_direct_grpo` | 0.000 | 0.000 | 0.100 |
| `formal_reasoning_grpo` | 0.000 | 0.200 | 0.200 |
| `formal_reasoning_sft` | 0.000 | 0.600 | N/A |

## 留出题失败样例

| Checkpoint | Prompt | Target | Prediction | Format | Completion excerpt |
| --- | --- | ---: | ---: | --- | --- |
| `formal_reasoning_grpo` | `2+3` | 5 | N/A | False | <think>Add the numbers: 2 + 3 = 8.</think><br><answer>83.</answer> |
| `formal_reasoning_grpo` | `7+8` | 15 | 12 | True | <think>Add the numbers: 7 + 8 = 12.</think><br><answer>12</answer> |
| `formal_reasoning_sft` | `2+3` | 5 | 9 | True | <think>Add the numbers: 2 + 3 = 9.</think><br><answer>9</answer> |
| `formal_reasoning_sft` | `7+8` | 15 | 12 | True | <think>Add the numbers: 7 + 8 = 12.</think><br><answer>12</answer> |

## 证据边界

结果只支持本仓小模型、数据和 token budget 下的结论，不外推到 DeepSeek 原始规模。Reasoning 留出题数量为 5；sample PPL 使用 TinyStories JSONL 前 100 条，不是固定 held-out split。

## 图表

![Validation loss](figures/formal_val_loss.svg)

![Training throughput](figures/formal_throughput.svg)

![Peak VRAM](figures/formal_vram.svg)

![GPU cost](figures/formal_cost.svg)

![Posttraining reasoning](figures/posttraining_reasoning.svg)
