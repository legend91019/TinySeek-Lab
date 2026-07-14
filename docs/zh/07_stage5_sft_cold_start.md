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

## TinySeek 已实现的路径

1. 先做普通 SFT，把 base model 变成 chat model。
2. 做 reasoning cold-start SFT。
3. 对比 direct rule RL 和 cold-start 后 rule RL。

预期：

- direct RL 可能 reward 上升但表达混乱。
- cold-start + RL 的输出应该更规整。

代码已经完整接通：

- [`dataset/lm_dataset.py`](../../dataset/lm_dataset.py) 把 prompt token 的 label 设为 `-100`，只训练 response。
- [`trainer/train_sft.py`](../../trainer/train_sft.py) 从 base checkpoint 加载权重并做 SFT。
- [`scripts/prepare_toy_sft_data.py`](../../scripts/prepare_toy_sft_data.py) 生成最小 cold-start 教学数据。

```bash
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --hourly_rate 2.18
```

更细的 masking 与训练目标见[后训练代码细读](19_posttraining_code_walkthrough.md)。当前 4090 v1 证明 SFT 链路能运行并学习 toy 格式，但它损害了 TinyStories PPL；这说明小而窄的 SFT 数据会改变分布，不能把“格式学会了”写成“综合能力提高了”。

<!-- tinyseek-nav -->

---

上一篇: [阶段 4：MLA](06_stage4_mla.md) | [教程目录](README.md) | 下一篇: [阶段 6：GRPO Mini](08_stage6_grpo_mini.md)
