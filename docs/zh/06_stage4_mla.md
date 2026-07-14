# 06. 阶段 4：教学版 MLA

目标：理解 DeepSeek-V2 为什么重视 KV cache 压缩。

完整代码课见：[从 DeepSeekMoE 到 DeepSeek-V2](22_from_moe_to_deepseek_v2.md)。那里逐行解释 content/rope 拆分、latent 重建和训练 forward 与 cached decoding 的边界。

## DeepSeek 对应关系

DeepSeek-V2 引入 Multi-head Latent Attention，论文里强调它能把 KV cache 压缩到 latent vector，从而提升推理效率。

TinySeek 里的 `educational_mla` 不是完整复刻 DeepSeek-V2 的生产级 MLA，而是教学版：

1. 把 hidden states 投影到一个低秩 latent。
2. 从 latent 重建 K/V。
3. 用 latent size 估算 KV cache 成本。

## 运行

```bash
python trainer/train_pretrain.py --config configs/tiny_mla.json --data data/toy_pretrain.jsonl
```

## 建议实验

比较：

- `attention_impl = mha`
- `attention_impl = educational_mla`
- `mla_latent_size = 32, 64, 96`

指标：

- validation loss。
- KV-cache elements per token。
- 生成速度。
- 长上下文下显存占用。

## 关键理解

MLA 不是简单的推理时外挂技巧。模型需要在训练中适应这种 K/V 表示方式。TinySeek 第一版先把概念讲清楚，后续再逐步增加 cached generation 和更接近论文的结构。

## 单变量对照

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data data/tinystories.jsonl --hourly_rate 2.18
```

当前能严谨比较的是 validation loss、训练成本和理论 KV 元素；没有 production latent cache 前，不能把训练显存当作 MLA 推理缓存收益。待填结果见[架构演进实验计划](../../experiments/06_architecture_evolution_plan_zh.md)。

<!-- tinyseek-nav -->

---

上一篇: [阶段 3：MoE](05_stage3_moe.md) | [教程目录](README.md) | 下一篇: [阶段 5：SFT](07_stage5_sft_cold_start.md)
