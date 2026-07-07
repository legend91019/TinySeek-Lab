# 07. 阶段 5：SFT 和 Reasoning Cold Start

目标：从 base LM 走向 instruction-following 和更可读的推理。

## DeepSeek 对应关系

DeepSeek-R1 里有两个很重要的对照：

- R1-Zero：从 pretrained base model 直接做 RL。
- R1：先用少量 cold-start reasoning 数据做 SFT，再做 RL。

这里的 cold-start 本质上可以理解为一小段高质量 reasoning SFT。它不主要负责让模型学会世界知识，而是让模型学会比较规整、可读、稳定的推理表达格式。

## 为什么需要 Cold Start

直接 RL 可能会提高 reward，但输出可能很乱：

- 格式不稳定。
- 中英混杂。
- 推理过程可读性差。
- 重复和碎片化表达多。

Cold-start SFT 相当于先告诉模型：

> 做推理题时，答案大概应该这样组织。

然后再用可验证奖励优化正确率。

## TinySeek 计划

1. 先做普通 SFT，把 base model 变成 chat model。
2. 做 reasoning cold-start SFT。
3. 对比 direct rule RL 和 cold-start 后 rule RL。

预期：

- direct RL 可能 reward 上升但表达混乱。
- cold-start + RL 的输出应该更规整。

当前状态：

- `trainer/train_sft.py` 还是占位。
- 数据格式和训练代码将在后续版本补齐。
