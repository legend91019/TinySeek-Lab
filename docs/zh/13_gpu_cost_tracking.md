# GPU 选择与成本记录

这一章是 TinySeek-Lab 的“硬件账本”。我们不只训练一个小模型，还要教读者怎么把每次实验的成本讲清楚：显存、运行时间、GPU 小时数、租卡单价和总费用。

## 3080 Ti 够不够？

够用，但要看目标。

| GPU | 常见显存 | AutoDL 单价 | 在 TinySeek-Lab 里的定位 |
| --- | ---: | ---: | --- |
| RTX 3080 Ti | 12 GB | 0.98 元/小时 | 便宜验证、dense baseline、小规模 LR/batch sweep、早期调代码 |
| RTX 4090 | 24 GB | 2.18 元/小时 | 完整 v1.0 实验路线、更大的 batch、MoE/MLA、更少显存妥协 |

我的建议很明确：

- 如果你只是想先学习、调通、验证每一阶段，3080 Ti 可以。
- 如果你想把这个仓库做成比较完整、可发布、实验表格也比较像样的 v1.0，建议租 4090。

3080 Ti 的小时单价低，但是 12 GB 显存会更容易限制 batch size、sequence length 和 MoE 实验；4090 单价更高，但 24 GB 显存会让实验路线顺很多。

## 粗略预算

下面是计划估算，不是保证值。真实花费取决于数据量、模型宽度/层数、序列长度、batch size，以及中途调试次数。

| 里程碑 | 预计 GPU 时间 | 4090，2.18 元/小时 | 3080 Ti，0.98 元/小时 |
| --- | ---: | ---: | ---: |
| v0.2 | 12-24 小时 | 26-52 元 | 12-24 元 |
| v0.5 | 35-70 小时 | 76-153 元 | 34-69 元 |
| v1.0 | 60-120 小时 | 131-262 元 | 59-118 元 |

但要注意：3080 Ti 可能因为 batch 更小、实验更碎，实际小时数会增加，所以真实成本差距不会像单价看起来那么大。

## 每次训练自动记账

`trainer/train_pretrain.py` 每次训练结束都会写一个成本摘要：

```bash
python trainer/train_pretrain.py \
  --config configs/tiny_dense.json \
  --data data/toy_pretrain.jsonl \
  --max_steps 2000 \
  --hourly_rate 2.18 \
  --currency CNY
```

输出文件是：

```text
out/<run_name>_cost_summary.json
```

里面会记录：

- GPU 名称、CUDA 状态、CUDA 版本、总显存。
- 峰值 allocated 显存和峰值 reserved 显存。
- 运行秒数和 GPU 小时数。
- 每小时单价、货币和估算费用。
- 估算训练 token 数和粗略训练 FLOPs。
- 模型参数量和激活参数量估计。
- 最后的训练 loss 和验证 loss。

AutoDL 示例：

```bash
# RTX 4090
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --hourly_rate 2.18

# RTX 3080 Ti
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --hourly_rate 0.98
```

## Sweep 也要记账

网格搜索脚本也会把单价传给每个实验：

```bash
python trainer/sweep_pretrain.py \
  --sweep experiments/01_lr_batch_grid.json \
  --hourly_rate 2.18 \
  --currency CNY
```

跑完以后汇总所有实验账本：

```bash
python scripts/summarize_costs.py --input_dir out
```

它会生成：

```text
out/cost_summary.md
out/cost_summary.csv
```

Markdown 表格可以直接放进教程报告，CSV 可以继续用表格软件分析。

## 报告里必须写什么

每个正式实验报告都应该包含：

| 字段 | 为什么重要 |
| --- | --- |
| GPU 型号 | 硬件会影响速度和显存余量 |
| 峰值 allocated 显存 | 说明实际张量显存压力 |
| 峰值 reserved 显存 | 说明 PyTorch allocator 压力 |
| GPU 小时数 | 让实验预算可复现 |
| 总费用 | 让读者看到真实 tradeoff |
| 处理 token 数 | 方便比较训练效率 |
| 近似 FLOPs | 给实验一个粗略计算量尺度 |
| 验证 loss | 把成本和模型质量关联起来 |

TinySeek-Lab 要把成本当成一个真正的实验指标，而不是最后顺手补的一行。
