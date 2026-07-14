# 06. Stage 4: Educational MLA

Goal: understand why DeepSeek-V2 cares about KV-cache compression.

Read the complete code lesson: [DeepSeekMoE to DeepSeek-V2](22_from_moe_to_deepseek_v2.md). It explains content/position splitting, latent reconstruction, and the boundary between training forward and cached decoding.

## DeepSeek Anchor

DeepSeek-V2 introduces Multi-head Latent Attention (MLA), described as
compressing the Key-Value cache into a latent vector for efficient inference.

TinySeek's `educational_mla` is not a faithful production MLA implementation.
It is a teaching version:

- project hidden states into a low-rank latent;
- reconstruct K/V from that latent;
- account for latent cache elements instead of full K/V elements.

## Run

```bash
python trainer/train_pretrain.py --config configs/tiny_mla.json --data data/toy_pretrain.jsonl
```

## Experiments

Compare:

- `attention_impl = mha`
- `attention_impl = educational_mla`
- `mla_latent_size = 32, 64, 96`

Metrics:

- Validation loss.
- KV-cache elements per token.
- Generation speed after adding a cached inference path.

## Takeaway

MLA is easiest to understand as an inference-efficiency move that has to be
trained into the model, not merely bolted onto a trained MHA model.

## Matched Comparison

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data data/tinystories.jsonl --hourly_rate 2.18
```

The rigorous current comparisons are validation loss, training cost, and theoretical cache elements. Without a production latent cache, training VRAM is not evidence of decoding-cache savings. Results remain pending in the [architecture plan](../experiments/06_architecture_evolution_plan.md).

<!-- tinyseek-nav -->

---

Previous: [Stage 3: MoE](05_stage3_moe.md) | [Tutorial Index](README.md) | Next: [Stage 5: SFT](07_stage5_sft_cold_start.md)
