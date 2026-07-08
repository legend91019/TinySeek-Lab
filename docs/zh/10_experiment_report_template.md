# 10. 实验报告模板

每个章节都应该产出一份小报告。这个项目要教的不只是代码，还有训练研究的方法。

## 实验标题

一句话说明实验。

## DeepSeek 论文锚点

这个实验对应哪篇 DeepSeek 论文的哪个思想？

例子：

- 论文：DeepSeek LLM。
- 思想：扩大模型之前，先做 batch size 和 learning rate 搜索。

## 假设

你预期会发生什么？

例子：

> 在固定 token budget 下，更大的 effective batch 可能需要略高的 learning rate；但 learning rate 过高会导致 validation loss 不稳定。

## 实验设置

- 模型配置：
- 数据：
- token budget：
- 硬件：
- seed：

## Sweep 变量

| Run | 改动变量 |
|---|---|
| baseline | 无 |

## 指标

- train loss。
- validation loss。
- tokens/sec。
- peak memory。
- mini downstream eval。

## 结果

| Run | Train loss | Val loss | Tokens/sec | Peak memory | Notes |
|---|---:|---:|---:|---:|---|

## 结论

下一步应该改什么？

## 失败案例

失败的 run 也要记录。训练研究里，失败记录经常比成功结果更有价值。

<!-- tinyseek-nav -->

---

上一篇: [仓库路线图](09_repository_roadmap.md) | [教程目录](README.md) | 下一篇: [MiniMind 结构说明](11_minimind_structure_notes.md)
