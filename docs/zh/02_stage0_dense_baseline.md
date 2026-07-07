# 02. 阶段 0：Dense Baseline

目标：先训练一个最简单但稳定的 decoder-only 语言模型。

## DeepSeek 对应关系

DeepSeek LLM 的基础结构大体沿着现代 LLaMA-style 路线：

- Pre-Norm Transformer。
- RMSNorm。
- RoPE。
- SwiGLU。
- 在大模型上使用 GQA 降低推理成本。

TinySeek 的第一步也从这类结构开始，不过规模缩到很小。我们先不急着做 MoE、MLA 或 RL，因为如果 Dense baseline 都不稳定，后面所有实验都会变成噪声。

## 实验目标

假设：

> 只要数据格式、学习率、batch size 和 causal mask 没有明显错误，小型 Dense LM 的训练 loss 应该能稳定下降。

最小设置：

- 配置：`configs/tiny_dense.json`
- tokenizer：byte-level tokenizer
- 数据：`data/toy_pretrain.jsonl`
- 目标：next-token prediction

运行：

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 100
```

## 需要观察什么

- train loss 是否下降。
- validation loss 是否跟着下降。
- tokens/sec 是否稳定。
- 生成文本是否逐渐学到训练语料里的主题词。

## 常见问题

如果 loss 不下降，先检查：

- labels 是否右移。
- pad token 是否被 mask 成 `-100`。
- 学习率是否太大。
- 数据是否太短或全是重复内容。

这一章的原则是：先让一条最普通的训练链路跑通。
