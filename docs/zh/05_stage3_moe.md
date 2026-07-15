# 05. 阶段 3：Tiny DeepSeekMoE

目标：把 Dense FFN 替换成 routed experts，开始研究稀疏激活。

先读完整代码课：[从 Dense LM 到 DeepSeekMoE](21_from_dense_to_deepseek_moe.md)。它会从整模 `forward` 出发解释 router、dispatch、shared experts 和参数计算，本章只负责实验。

## DeepSeek 对应关系

DeepSeekMoE 的核心问题是：如何让不同 expert 形成分工，同时避免 routing collapse。

routing collapse 指的是模型总是把 token 分给少数几个 expert，导致其他 expert 训练不足。MoE 的难点不只是“多放几个 MLP”，而是让路由、负载和训练稳定性一起工作。

## TinySeek 实现

使用 `configs/tiny_moe.json`：

```bash
python trainer/train_pretrain.py --config configs/tiny_moe.json --data data/toy_pretrain.jsonl
```

当前实现包含：

- top-k routing。
- routed experts。
- optional shared experts。
- auxiliary load-balance proxy loss。
- total params 和 activated params 估算。
- 在 `*_history.jsonl` 和 `*_cost_summary.json` 中记录轻量 expert-load 快照。

## 建议实验

1. Dense vs MoE，在 activated params 接近时比较 validation loss。
2. `top_k = 1` vs `top_k = 2`。
3. `moe_aux_loss_weight = 0, 0.001, 0.01, 0.05`。
4. 是否加入 shared expert。

## 需要记录的问题

- expert 使用是否均衡？
- 有没有 routing collapse？
- aux loss 太大时是否伤害 LM loss？
- MoE 是否真的带来更好的参数效率？

expert-load 快照会记录最近一次 forward 里每层 expert 的分配计数。它不是完整
routing trace，但足够发现教程规模实验里的明显 routing collapse。后续可以在
这个基础上继续做 routing histogram。

## 正式对照入口

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data data/tinystories.jsonl --hourly_rate 2.18
```

不要只报告总参数。至少同时报告 activated parameters、expert load、aux loss、validation loss、tokens/s 和峰值显存。3-seed 完整数字与质量/吞吐分支决定见[架构实测报告](../../experiments/architecture_lab_runs/report_zh.md)。

<!-- tinyseek-nav -->

---

上一篇: [阶段 2：Block 升级](04_stage2_block_upgrades.md) | [教程目录](README.md) | 下一篇: [阶段 4：MLA](06_stage4_mla.md)
