# 03. 阶段 1：LR 和 Batch Size 搜索

目标：把 DeepSeek LLM 论文里“先研究训练 recipe，再扩大模型”的思路，缩小成可复现的小实验。

## DeepSeek 对应关系

DeepSeek LLM 不是一上来就训练最终大模型，而是先研究 batch size、learning rate、模型规模和数据规模之间的关系。TinySeek 不能复现它的规模，但可以复现这种研究习惯：

1. 固定模型大小。
2. 固定 token budget。
3. 扫 batch size 和 learning rate。
4. 对比 validation loss、稳定性和吞吐。
5. 找到稳定 recipe 后再升级结构。

## Tiny Sweep

运行：

```bash
python trainer/sweep_pretrain.py --sweep experiments/01_lr_batch_grid.json
```

当前 toy sweep 包含：

- batch size：8、16
- learning rate：3e-4、6e-4

真实实验可以扩展为：

```text
batch tokens: 32K, 64K, 128K, 256K
learning rate: 1e-4, 3e-4, 6e-4, 1e-3
warmup: 1%, 2%, 5%
```

## 报告表格

| Run | Batch tokens | LR | Warmup | Val loss | Tokens/sec | Notes |
|---|---:|---:|---:|---:|---:|---|
| bs8_lr3e-4 | TBD | 3e-4 | TBD | TBD | TBD | baseline |

## 你应该学到什么

这一章不追求找到“宇宙最优学习率”。重点是养成习惯：

- 每次只改少数变量。
- 记录失败实验。
- 在加复杂结构之前，先确认训练 recipe 是稳定的。
