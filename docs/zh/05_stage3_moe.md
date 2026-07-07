# 05. 阶段 3：Tiny DeepSeekMoE

目标：把 Dense FFN 替换成 routed experts，开始研究稀疏激活。

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

这章后续要补一个 routing histogram，把每个 expert 收到的 token 数画出来。
