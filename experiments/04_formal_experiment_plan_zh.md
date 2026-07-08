# 正式实验计划

这份计划用于下一次付费 GPU 训练。不要在 toy 数据上跑这些正式实验。

## 数据集

用 `scripts/prepare_hf_dataset.py` 接线上数据集，或者直接使用 MiniMind 风格的本地 JSONL。v1 不需要重型数据清洗 pipeline；本仓重点是训练、模型代码和架构路线。

推荐选项：

| 数据集 | 为什么用 | 示例命令 |
| --- | --- | --- |
| `roneneldan/TinyStories` | 适合小模型，loss 曲线明显 | `python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --out data/tinystories.jsonl` |
| `wikitext` / `wikitext-2-raw-v1` | 经典语言建模入门语料 | `python scripts/prepare_hf_dataset.py --dataset_name wikitext --dataset_config wikitext-2-raw-v1 --split train --text_field text --out data/wikitext2.jsonl` |
| MiniMind 风格 JSONL | 更接近中文小模型教程 | 直接使用 `{"text": "..."}` JSONL |

## 实验列表

| ID | 目标 | 配置 | token/example 预算 | 指标 |
| --- | --- | --- | ---: | --- |
| E1 | Dense baseline | `configs/medium_dense_35m.json` | 20M-50M tokens | val loss, ppl, cost |
| E2 | 更大 dense 短训 | `configs/medium_dense_115m.json` | 10M-30M tokens | val loss, ppl, memory |
| E3 | LR/batch sweep | 把 `experiments/01_lr_batch_grid.json` 改到真实数据 | 4 runs x 5M tokens | 最优 LR/batch |
| E4 | MoE 激活参数对照 | `configs/medium_moe_activated_35m.json` | 10M-30M tokens | activated params, memory, val loss |
| E5 | MLA/KV 对照 | `configs/tiny_mla.json` 和 dense control | 5M-10M tokens | KV elements/token, memory |
| E6 | SFT cold start | `configs/tiny_sft.json` | 2k-20k examples | format eval, addition eval |
| E7 | GRPO mini | `configs/tiny_grpo.json` | 100-500 updates | reward curve, pass rate |

## 报告必须包含的列

每份实验表格都应该包括：

- config 文件；
- dataset 和 split；
- token 数或 example 数；
- 总参数量和激活参数量；
- validation loss 或 reward；
- mini-eval 分数；
- 峰值 allocated/reserved 显存；
- GPU 小时；
- 估算成本。

## 停止规则

出现这些情况就提前停止：

- validation loss 连续三次 eval 不下降或变差；
- SFT 已经改进后 reward 仍然长期为 0；
- peak reserved VRAM 超过 GPU 的 90%；
- 生成样本明显坏掉，说明数据格式错了。
