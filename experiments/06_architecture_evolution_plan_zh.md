# DeepSeek 架构演进公平实验计划

中文 | [English](06_architecture_evolution_plan.md)

这组实验回答的不是“TinySeek 能否达到 DeepSeek 的能力”，而是更小、更可验证的问题：当数据、token budget 和优化器不变时，某一个结构改动会怎样影响 loss、吞吐、显存和路由行为？

## 先做 CPU 结构检查

```bash
python scripts/inspect_stage_models.py --out out/stage_model_inspection.json
```

它会对四代完整教学模型各做一次 forward，输出 logits shape、总参数、激活参数、loss 字段和单层理论 KV cache 元素数。这一步只验证代码与张量契约，不验证模型质量。

## 公平性约束

每组 A/B 实验固定以下条件：

- 同一份 JSONL 数据和 train/validation 切分；
- 相同 `max_seq_len=128`、batch size、学习率、warmup 和随机种子；
- 相同训练步数，因此 token budget 相同；
- 相同 GPU、PyTorch 版本和精度；
- 只允许表格中列出的目标字段不同。

`tests/architecture_lab_contract_test.py` 会自动检查配置是否违反这些约束。

## 实验矩阵

| ID | 对照 | 只改变 | 论文动机 | TinySeek 状态 |
| --- | --- | --- | --- | --- |
| A1 | MHA vs GQA | `num_kv_heads` | DeepSeek LLM 67B 使用 GQA 降低推理成本 | 待上卡 |
| A2 | aux loss vs bias balance | 路由均衡策略和 aux loss 权重 | V3 避免负载均衡辅助损失干扰 LM 目标 | 待上卡 |
| A3 | MTP off vs on | `mtp_depth` | V3 用额外未来 token 预测增加训练信号 | 待上卡 |
| A4 | GQA control vs educational MLA | `attention_impl` | V2 用低秩 KV latent 压缩缓存 | 待上卡 |

## 运行命令

把 `DATA` 替换成同一份语料，例如 `data/tinystories.jsonl`：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/dense_mha.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/dense_gqa.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/v3_no_mtp.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_mtp.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data "$DATA" --hourly_rate 2.18
```

## 结果表

表中的 `val loss` 固定指 token-weighted `val_lm_loss`，PPL 只由这个主语言模型 loss 计算。`val_objective`、`val_mtp_loss` 和 `val_aux_loss` 另行记录，不能混进 PPL 或直接拿来做 A/B 排名。

| Run | val loss | PPL | tokens/s | peak VRAM | GPU h | 成本 | 结构特有指标 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `arch_dense_mha` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | KV elements/token |
| `arch_dense_gqa` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | KV elements/token |
| `arch_moe_aux` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | expert load, aux loss |
| `arch_moe_bias` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | expert load, selection bias |
| `arch_v3_no_mtp` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | LM loss |
| `arch_v3_mtp` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | LM loss, MTP loss |
| `arch_v2_attention_control` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | theoretical KV elements |
| `arch_v2_mla` | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | 待上卡 | theoretical KV elements |

## 论文结论与本仓结论必须分开

| 来源 | 可以写什么 | 不可以写什么 |
| --- | --- | --- |
| DeepSeek LLM | 论文的小规模网格搜索显示较宽的近优 LR/batch 区域，并观察到 compute 增大时 batch 增、LR 降 | TinySeek 跑四个点就“复现了 scaling law” |
| DeepSeekMoE | 论文提出细粒度专家与 shared expert isolation，并在其规模上报告更好的专家特化和计算效率 | TinySeek 的 Python dispatch 证明了分布式 MoE 吞吐优势 |
| DeepSeek-V2 | 论文相对 DeepSeek 67B 报告训练成本降低 42.5%、KV cache 降低 93.3%、最大吞吐达到 5.76 倍 | 教学版 MLA 只做理论缓存估算就声称取得相同速度 |
| DeepSeek-V3 | 论文采用 auxiliary-loss-free balance 与 MTP，并用消融支持它们 | 结果尚未运行时提前断言 TinySeek loss 一定更好 |

## 如何解释失败

- GQA loss 变差：KV heads 太少可能伤害容量，不能只看缓存。
- MoE 不优于 Dense：token budget 太小、路由未形成分工，或激活参数没有真正匹配。
- bias routing 仍塌缩：更新率可能太小；振荡则可能太大。
- MTP 总 loss 更高：总 loss 包含额外目标，应优先比较主 `lm_loss`、validation PPL 和下游指标。
- MLA 显存没有下降：当前 trainer 没有生产级 latent KV cached decoding；理论缓存减少不等于训练显存下降。

<!-- tinyseek-nav -->

---

[实验报告中心](README_zh.md)
