# 23. 从 DeepSeek-V2 到 DeepSeek-V3：路由 Bias 与 MTP

V3 没有抛弃 V2。主干仍然是 MLA + DeepSeekMoE；本章只加两个训练期升级：auxiliary-loss-free load balancing 和 Multi-Token Prediction（MTP）。

对照代码：

- 上一代：[`stage2_deepseek_v2.py`](../../model/stages/stage2_deepseek_v2.py)
- 新一代：[`stage3_deepseek_v3.py`](../../model/stages/stage3_deepseek_v3.py)
- 正式实现：[`model/tinyseek.py`](../../model/tinyseek.py)
- 训练器：[`trainer/train_pretrain.py`](../../trainer/train_pretrain.py)

## 本章研究卡：先画出冲突，再换机制

V3 的升级分成三个实验，不能直接从 V2 跳到最终配置：

| 步骤 | 实验 | 先观察什么 | 何时继续 |
| --- | --- | --- | --- |
| C0 | aux 权重 `0 / 0.001 / 0.01 / 0.1` | expert load 是否更均衡；主 `val_lm_loss` 是否同时变差 | 找到“负载改善 - 主任务代价”曲线，而不是只挑最好看的一点 |
| C1 | 最合理的 aux 基线 vs selection bias | load spread、PPL、bias 振荡和 tokens/s | bias 至少维持负载健康，且主 LM 不劣于 aux 基线 |
| C2 | MTP off vs on | 主 LM loss/PPL、mini eval、吞吐和显存；不要用 total objective 排名 | 多 seed 主任务有稳定收益，且额外计算可接受 |

aux sweep 配置是 [`moe_aux_none.json`](../../configs/architecture_lab/moe_aux_none.json)、[`moe_aux_weak.json`](../../configs/architecture_lab/moe_aux_weak.json)、[`moe_aux.json`](../../configs/architecture_lab/moe_aux.json) 和 [`moe_aux_strong.json`](../../configs/architecture_lab/moe_aux_strong.json)。

如果 C0 没显示任何主任务代价，本仓就不能声称“小模型实验证明 aux 干扰严重”；只能引用 V3 论文动机，并继续把 C1 当机制对照。如果 C1 或 C2 不过门槛，就不晋升该机制，并保留对应 matched group 的 comparator。这正是实验驱动路线与“照着论文堆组件”的区别。

## 1. V2 路由哪里不够

Stage 1/2 用：

```text
total objective = LM loss + balance auxiliary loss
```

这样确实能惩罚不均衡路由，但 balance loss 和语言建模共享梯度。权重过大时，router 可能为了“平均分 token”牺牲最适合当前 token 的专家。

V3 的目标是：仍然调节专家负载，但尽量不把一个强均衡项塞进主优化目标。

## 2. 两套分数必须分清

`BiasBalancedMoE` 先计算 token-expert affinity：

```python
affinity = torch.sigmoid(self.gate(flat))
```

然后只在**选择专家**时加入 bias：

```python
selection_score = affinity + expert_bias
top_indices = topk(selection_score)
```

真正混合专家输出的权重仍来自原始 affinity：

```python
top_weights = affinity.gather(1, top_indices)
top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
```

因此：

```text
selection_score 决定“去哪个专家”
affinity 决定“选中后占多少权重”
```

如果把 bias 也混进 `top_weights`，就改变了专家输出的相对权重，不再是这里要教学的 auxiliary-loss-free selection bias。

## 3. Bias 怎么更新

每个 forward 记录专家实际接收 token 的次数：

```python
counts = bincount(top_indices)
```

优化器完成参数更新后，训练器调用：

```python
model.update_router_bias()
```

每层 router 做：

```python
target = counts.mean()
direction = sign(target - counts)
expert_bias += update_rate * direction
expert_bias -= expert_bias.mean()
```

- 过载专家：`counts > target`，bias 下降，下个 batch 更难入选。
- 欠载专家：`counts < target`，bias 上升，下个 batch 更容易入选。
- 减去均值：避免所有 bias 一起漂移。

这是教学版离散更新。正式大规模训练还要考虑跨设备统计、更新粒度、group-limited routing 和通信约束。

## 4. 为什么在 optimizer step 后更新

expert bias 是负载控制状态，不通过 `loss.backward()` 学习。训练循环的顺序是：

```text
forward -> LM/MTP/optional aux loss -> backward
-> optimizer step -> update router bias -> next batch
```

把它写进 trainer 能明确展示两种学习机制：

- 模型参数靠梯度下降更新；
- selection bias 靠负载反馈规则更新。

统一模型默认 `router_balance_strategy="aux"`，保证旧配置不变。只有实验配置显式设为 `"bias"` 时才启用 V3 路线。

## 5. MTP 为什么不等于“把 label 多移一位”

普通 causal LM 在位置 `i` 的 hidden state 预测 token `i+1`。MTP 想提供更远的训练信号，但不能让主干直接偷看未来 token。

教学版第一个 MTP module 接收：

- 主干位置 `i` 的 hidden state；
- 已知 token `i+1` 的 embedding；

然后预测 token `i+2`。

```text
main hidden at i + embedding(token i+1)
  -> concat projection
  -> one Transformer block
  -> shared LM head
  -> predict token i+2
```

第二个 MTP depth 再接前一个模块 hidden 与 token `i+2` embedding，预测 token `i+3`。

## 6. MTP 代码和 shape

一个模块先归一化两路输入：

```python
combined = torch.cat(
    (previous_norm(previous_hidden), token_norm(future_token_embed)),
    dim=-1,
)
hidden = block(concat_proj(combined))
```

若主序列长度是 `T`，第一个 depth：

```text
previous_hidden[:, :-1] : [B, T-1, D]
embed(input_ids[:, 1:]) : [B, T-1, D]
MTP hidden              : [B, T-1, D]
MTP logits[:, :-1]      : [B, T-2, V]
targets[:, 2:]          : [B, T-2]
```

这个对齐很重要。多一个或少一个 slice，模型就会预测错位置，loss 仍可能下降，却学的是错误任务。

## 7. 为什么共享 Embedding 和 LM Head

Stage 3 直接使用主模型的：

```python
future_embed = self.embed(future_token_ids)
mtp_logits = self.lm_head(self.norm(mtp_hidden))
```

共享可以减少额外参数，并保证所有预测深度使用同一 token 空间。教学版 MTP 模块内部使用一个 Dense Transformer block，重点保留顺序预测、共享 embedding/head 和 label shift；它没有复刻 V3 的所有工程细节。

## 8. 总 Loss 如何组成

```python
loss = lm_loss + mtp_loss_weight * mtp_loss
```

MoE 的 `aux_loss` 仍单独返回，由 trainer 相加。bias 方案通常把 `moe_aux_loss_weight` 设为 0；也可以用很小的值做 complementary sequence-wise balance 对照。

日志现在分别记录：

- `train_loss`：trainer 最终优化的总值；
- `lm_loss`：主 next-token objective；
- `mtp_loss`：额外未来 token objective；
- `aux_loss`：可选 MoE balance 项；
- `expert_load`：分配比例和 selection bias。

比较 MTP off/on 时，不要只看 total loss，因为 on 版本天然多一个目标。优先看主 `lm_loss`、validation PPL 和 mini eval。

## 9. 完整 Stage 3 Forward

整模顺序是：

```text
Embedding
-> N x (Educational MLA + BiasBalancedMoE)
-> main RMSNorm + shared LM head
-> main next-token loss
-> sequential MTP module(s), training only
-> weighted total loss
```

推理不提供 labels 时不会计算 MTP loss，生成接口仍只使用主 logits。

## 10. 如何验证

```bash
python tests/stage_models_test.py
python tests/unified_v3_contract_test.py
```

测试重点：

- 人工设置 `[20,2,2,2]` 的负载后，Expert 0 bias 必须下降，其余上升；
- MTP 开关不改变主 logits shape；
- `loss == lm_loss + weight * mtp_loss`；
- 老配置仍默认为 aux routing 和 `mtp_depth=0`。

## 11. 公平实验

先跑 aux 权重 sweep，确认问题形状：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_none.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_weak.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux_strong.json --data data/tinystories.jsonl --hourly_rate 2.18
```

再做路由机制对照：

路由对照：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data data/tinystories.jsonl --hourly_rate 2.18
```

MTP 对照：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v3_no_mtp.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v3_mtp.json --data data/tinystories.jsonl --hourly_rate 2.18
```

[3-seed 实测报告](../../experiments/architecture_lab_runs/report_zh.md)显示：当前 bias routing 的 PPL 与 load CV 都不如 aux=0.01；在同一 educational-MLA+bias 分支上，MTP 的平均 PPL 略低，但差异落在 seed 波动内，同时显存与时间增加，因此这条 V3-style 分支不升级。这个实验**不能**回答 MTP 是否有益于已选中的 GQA+aux recipe；那需要新的 matched pair。TinySeek 小预算负结果也不反驳 V3 论文规模结论。

正式结论至少重复多个 seed。若 budget 只够单 seed，就把结果标为 pilot，并优先保留完整 history、expert load 和 cost ledger，供下一轮复查。

## 12. V3 之后为什么进入 R1

到这里，base model 的架构路线完成。接下来研究问题从“block 怎么算”转向“模型如何学会遵循指令和推理”：SFT、cold-start data、GRPO、rejection sampling。也就是说，R1 章节升级的是训练数据流和目标，不是再新增一个 `Stage4Attention`。

<!-- tinyseek-nav -->

---

上一篇: [DeepSeekMoE 到 DeepSeek-V2](22_from_moe_to_deepseek_v2.md) | [教程目录](README.md) | 下一篇: [训练主循环](16_training_loop_from_config_to_checkpoint.md)
