# DeepSeek 架构演进公平实验计划

中文 | [English](06_architecture_evolution_plan.md)

这组实验回答的不是“TinySeek 能否达到 DeepSeek 的能力”，而是更小、更可验证的问题：当数据、token budget 和优化器不变时，某一个结构改动会怎样影响 loss、吞吐、显存和路由行为？

它也不是把所有新组件一次性打开。下面是教学与分析顺序：

```text
锁定 Dense recipe
-> 验证普通 MoE 到 DeepSeekMoE 的结构消融
-> 测量 KV cache 瓶颈并验证低秩/MLA
-> 画出 aux loss 权衡曲线，再比较 selection bias
-> 最后验证 MTP
```

决策门槛控制的是某个组件能否**晋升并接入累计分支**，不会阻止彼此独立的 matched group 继续运行。因此即使单独的 MLA 或 bias 门槛失败，V3-style 机制组仍可测量 MTP，但结论只能留在该分支内。失败结果同样要保留，因为它决定是调整假设、扩大训练预算，还是继续使用上一条分支。

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

仓库中的单 seed 配置首先用于 pilot。正式结论应至少运行 3 个 seed，报告均值、标准差和全部单次结果；若预算不足，必须把单 seed 结果标为 `pilot`，不能写成稳定结论。

`tests/architecture_lab_contract_test.py` 会自动检查配置是否违反这些约束。

这张矩阵包含的是**彼此独立的组内公平对照**，不是把每组赢家依次叠加成一条单线模型。coarse/fine/shared 容量组与 8-routed、top-2 的 aux/MLA/V3-style 机制组使用不同 expert 拓扑；只能在声明的组内比较，某个机制通过后若要接到另一条已选分支，必须重新做 matched test。

## 实验矩阵

| ID | 对照 | 只改变 | 论文动机 | TinySeek 状态 |
| --- | --- | --- | --- | --- |
| A1 | MHA vs GQA | `num_kv_heads` | DeepSeek LLM 67B 使用 GQA 降低推理成本 | 已实测，3 seeds |
| A2a | 粗粒度 MoE vs 细粒度 MoE vs shared isolation | expert 数、宽度、top-k、shared 划分；匹配 expert FFN 总容量和激活宽度 | DeepSeekMoE 的 fine-grained segmentation 与 shared expert isolation | 已实测，3 seeds |
| A2b | aux 权重 `0/0.001/0.01/0.1` | `moe_aux_loss_weight` | 先测负载改善与主任务代价的权衡 | 已实测，3 seeds |
| A2c | 合理 aux 基线 vs bias balance | 路由均衡策略和 aux loss 权重 | V3 避免负载均衡辅助损失直接干扰 LM 目标 | 已实测，3 seeds |
| A3 | MTP off vs on | `mtp_depth` | V3 用额外未来 token 预测增加训练信号 | 已实测，3 seeds |
| A4a | GQA control vs 朴素低秩 KV | `attention_impl`，RoPE 保持耦合 | 低秩投影本身的质量代价 | 已实测，3 seeds |
| A4b | GQA control vs educational MLA | `attention_impl`，使用解耦 RoPE | V2 用低秩 KV latent 压缩可缓存状态 | 已实测，3 seeds |

## 运行命令

把 `DATA` 替换成同一份语料，例如 `data/tinystories.jsonl`：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/dense_mha.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/dense_gqa.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/moe_coarse.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_fine_grained.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_shared.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_none.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_weak.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_strong.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/v3_no_mtp.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_mtp.json --data "$DATA" --hourly_rate 2.18

python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_control.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_kv.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data "$DATA" --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data "$DATA" --hourly_rate 2.18
```

## 结果表

完整表格由 48 份成本账本自动生成，不手工复制：

- [双语 3-seed 报告、决策与图表](architecture_lab_runs/report_zh.md)
- [聚合 CSV](architecture_lab_runs/aggregate.csv)
- [逐 run CSV](architecture_lab_runs/results.csv)
- [数据 SHA256 清单](architecture_lab_runs/dataset_manifest.json)

门槛摘要：GQA 通过；shared experts 提高质量但降低吞吐；aux=0.01 是当前最佳负载/质量折中；bias routing 与教学版 MLA 未通过门槛。MTP 只在被拒绝的 educational-MLA+bias 分支上结论不确定；选中的 GQA+aux 分支仍需单独做 matched MTP 实验。

## 逐阶段决策门槛

| 阶段 | 升级条件 | 不升级或继续实验的条件 |
| --- | --- | --- |
| Dense -> GQA | 理论 KV 明显下降，多 seed PPL 无显著退化 | 质量退化超过波动；增加 KV heads 或保留 MHA |
| 普通 MoE -> fine-grained/shared | 无塌缩；PPL 稳定改善或在相近质量下容量更大，吞吐代价可接受 | 只在单 seed 有利、负载塌缩或 Python dispatch 成本压倒收益 |
| GQA -> MLA | `192 -> 72` 理论元素成立且 PPL 可接受 | 质量损失大则 sweep latent rank；无 cached decoding 时不发布真实吞吐结论 |
| aux -> bias | bias 负载至少和合理 aux 基线一样健康，主 LM/PPL 不劣 | bias 振荡或塌缩时 sweep update rate；aux 基线更稳则保留 aux |
| MTP off -> on | 主 LM/PPL 或 mini eval 多 seed 改善，新增显存/时间可接受 | 只看 total loss 下降不算；主任务无收益则关闭 MTP |

每阶段报告最后固定写：`观察`、`是否过门槛`、`决定`、`下一轮实验`。当前生成报告已经按实测数据回填；未来新增配置仍必须先标“待验证”，运行后才能下结论。

## 论文结论与本仓结论必须分开

| 来源 | 可以写什么 | 不可以写什么 |
| --- | --- | --- |
| DeepSeek LLM | 论文的小规模网格搜索显示较宽的近优 LR/batch 区域，并观察到 compute 增大时 batch 增、LR 降 | TinySeek 跑四个点就“复现了 scaling law” |
| DeepSeekMoE | 论文提出细粒度专家与 shared expert isolation，并在其规模上报告更好的专家特化和计算效率 | TinySeek 的 Python dispatch 证明了分布式 MoE 吞吐优势 |
| DeepSeek-V2 | 论文相对 DeepSeek 67B 报告训练成本降低 42.5%、KV cache 降低 93.3%、最大吞吐达到 5.76 倍 | 教学版 MLA 只做理论缓存估算就声称取得相同速度 |
| DeepSeek-V3 | 论文采用 auxiliary-loss-free balance 与 MTP，并用消融支持它们 | TinySeek 的小预算失败可以反推论文方法在大规模无效 |

## 如何解释失败

- GQA loss 变差：KV heads 太少可能伤害容量，不能只看缓存。
- MoE 不优于 Dense：token budget 太小、路由未形成分工，或激活参数没有真正匹配。
- bias routing 仍塌缩：更新率可能太小；振荡则可能太大。
- MTP 总 loss 更高：总 loss 包含额外目标，应优先比较主 `lm_loss`、validation PPL 和下游指标。
- MLA 显存没有下降：当前 trainer 没有生产级 latent KV cached decoding；理论缓存减少不等于训练显存下降。

<!-- tinyseek-nav -->

---

[实验报告中心](README_zh.md)
