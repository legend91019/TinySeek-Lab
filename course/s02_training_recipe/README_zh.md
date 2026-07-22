# s02 训练配方：先固定对照组

中文 | [English](README.md) | [课程目录](../README_zh.md)

## 研究问题

如果 learning rate 或有效 batch size 不合适，架构比较就无法解释。我们要先选出哪一小块训练配方作为对照设置？

## 论文线索

DeepSeek LLM 报告了系统的训练配方研究，包括 batch size 和 learning rate。TinySeek 把这个想法缩成一个小网格；这是方法复现，不是 scaling law 结论。

## 代码改变

这里**故意不改模型**。阅读 [`trainer/train_pretrain.py`](../../trainer/train_pretrain.py) 和 [`trainer/sweep_pretrain.py`](../../trainer/sweep_pretrain.py)：config 加载、一次 optimizer step 的有效 token 数、warmup/cosine schedule、AMP、梯度裁剪、验证和成本账本。

可执行的实验问题在配置里：

- [`experiments/gpu_completion_runs/configs/`](../../experiments/gpu_completion_runs/configs/)
- [`configs/architecture_lab/`](../../configs/architecture_lab/)

## 实验卡片

```text
同一模型 + 同一数据 + 同一 token budget
只改变 batch size 和 learning rate
再用同一验证集重复比较候选
```

正式 4090 sweep 使用 `bs16/bs32 x lr3e-4/lr6e-4`。loss、PPL、吞吐、峰值显存、GPU 小时和成本要一起记录；更低的 loss 不自动等于更好的教学对照。

## 结果与决定

正式报告在这个很小的预算中选择 `formal_sweep_bs16_lr6e-4`（`val loss 0.6475`）。这是本仓配置下的局部配方选择，不是普适 LR 规则。接下来固定配方，再一次只改变一个架构变量。

完整结果见 [`experiments/gpu_completion_runs/report_zh.md`](../../experiments/gpu_completion_runs/report_zh.md)，原有教学章节见 [`docs/zh/03_stage1_lr_batch_search.md`](../../docs/zh/03_stage1_lr_batch_search.md)。

## 代码练习

打印一次 optimizer update 消耗的 token 数。然后解释：为什么改变 `batch_size` 会同时改变梯度噪声和峰值显存；为什么 `grad_accum_steps` 可以保持有效 batch，却仍然改变墙钟时间。

## 下一章

对照组固定后，先问 attention 侧的问题：[s03 GQA](../s03_gqa/README_zh.md) 将验证减少 K/V 状态是否会损失基线语言建模质量。

<!-- tinyseek-nav -->

上一篇：[s01 Dense 基线](../s01_dense_baseline/README_zh.md) | [课程目录](../README_zh.md) | 下一篇：[s03 GQA](../s03_gqa/README_zh.md)
