# 21. 从 Dense LM 改到 DeepSeekMoE：完整代码怎么变

前置阅读：[`12_code_first_dense_lm.md`](12_code_first_dense_lm.md)。这一章不重新发明语言模型，只修改 Stage 0 中的 FFN 子层。

对照代码：

- 上一代：[`stage0_deepseek_llm.py`](../../model/stages/stage0_deepseek_llm.py)
- 新一代：[`stage1_deepseek_moe.py`](../../model/stages/stage1_deepseek_moe.py)
- 正式实验实现：[`MoEFFN`](../../model/tinyseek.py)

## 本章研究卡：不是“用了 MoE 就升级”

先把两个问题分开：

1. **Dense -> 稀疏 MoE**：容量与每 token 激活量能否解耦？这一步比较 total/activated parameters、LM loss、tokens/s 和显存。
2. **普通 MoE -> DeepSeekMoE**：在 expert FFN 总容量和激活宽度匹配时，细粒度切分与 shared expert isolation 是否有额外价值？

第二个问题使用三份专门配置：

| 候选 | routed x width | shared x width | 每 token 激活宽度 | expert FFN 总容量 |
| --- | ---: | ---: | ---: | ---: |
| 粗粒度普通 MoE | `2 x 4D`，top-1 | `0` | `4D` | `8D` |
| 细粒度 MoE | `8 x 1D`，top-4 | `0` | `4D` | `8D` |
| shared isolation | `7 x 1D`，top-3 | `1 x 1D` | `4D` | `8D` |

对应 [`moe_coarse.json`](../../configs/architecture_lab/moe_coarse.json)、[`moe_fine_grained.json`](../../configs/architecture_lab/moe_fine_grained.json) 和 [`moe_shared.json`](../../configs/architecture_lab/moe_shared.json)。router 参数会因 expert 数略有差异，因此报告仍要列真实总参数；上表匹配的是主要的 expert FFN 容量和激活宽度。

**升级门槛**：先确认无 routing collapse；再看多 seed 的 validation LM loss/PPL。若细粒度或 shared 版本没有稳定收益，或吞吐代价明显超出质量收益，就保留更简单的 MoE。仅凭“专家数更多”不能宣布创新有效。

## 1. 上一代哪里不够

Dense block 是：

```python
x = x + self.attn(self.attn_norm(x))
x = x + self.ffn(self.ffn_norm(x))
```

其中 `self.ffn` 对每个 token 都运行同一个 SwiGLU。增大 FFN 宽度会同时增加：

- 模型参数容量；
- 每个 token 的计算量；
- optimizer state 和训练显存。

MoE 想把前两件事拆开：模型可以存很多专家参数，但一个 token 只计算 top-k 个专家。

## 2. 论文做了什么

普通 MoE 已经有 router 和 top-k expert。DeepSeekMoE 的重点进一步包括：

- **Fine-grained expert segmentation**：把一个大 FFN 的中间维拆成更小专家，并增加专家数量，让组合更灵活。
- **Shared expert isolation**：始终激活少数 shared experts，承接共通知识，减少 routed experts 重复学习同样内容。

论文中的收益来自大规模训练和 MoE 基础设施。本仓只复现单卡可读的数据流，不实现 expert parallel 和 all-to-all 通信。

## 3. Config 新增了什么

```python
@dataclass
class Stage1Config(Stage0Config):
    num_routed_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    expert_ffn_multiplier: float = 1.0
    moe_aux_loss_weight: float = 0.01
```

假设 `hidden_size=192`：

- Stage 0 Dense FFN 的中间维是 `192 * 4 = 768`。
- Stage 1 每个细粒度 expert 的中间维是 `192 * 1 = 192`。
- 模型可以存 8 个 routed experts，但每个 token 只走 2 个。
- 1 个 shared expert 对所有 token 始终开启。

这不是保证计算量完全相等的公式。正式比较仍需看 activated parameters 和实际 tokens/s。

## 4. Router 逐行怎么写

输入仍是 block hidden states：

```text
x: [B, T, D]
flat: [B*T, D]
```

先给每个 token 计算专家概率：

```python
flat = x.reshape(-1, x.size(-1))
router_probs = softmax(router(flat), dim=-1)
```

若 `B=2, T=128, E=8`：

```text
router_probs: [256, 8]
```

每一行是一个 token 对 8 个专家的分布。然后选 top-k：

```python
top_weights, top_indices = torch.topk(router_probs, k=2, dim=-1)
top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
```

得到：

```text
top_indices: [256, 2]
top_weights: [256, 2]
```

例如某个 token 的结果是：

```text
expert ids = [5, 2]
weights    = [0.72, 0.28]
```

它只运行 Expert 5 和 Expert 2，最后做加权求和。

## 5. Dispatch 循环在做什么

教学实现按 expert 循环：

```python
out = torch.zeros_like(flat)
for expert_id, expert in enumerate(self.routed):
    token_mask = top_indices == expert_id
    token_indices, slots = token_mask.nonzero(as_tuple=True)
    expert_out = expert(flat[token_indices])
    out[token_indices] += expert_out * top_weights[token_indices, slots].unsqueeze(-1)
```

读法是：

1. 找出本 batch 中被分给当前 expert 的 token。
2. 只把这些 token 喂给该 expert。
3. 找到它在 top-k 中对应的权重。
4. 把加权输出累加回原 token 位置。

这个 Python 循环很适合教学，但不是高性能 MoE kernel。专家越多，单卡小 batch 下越可能浪费并行能力。

## 6. Shared expert 放在哪里

```python
shared_out = 0
for expert in self.shared:
    shared_out = shared_out + expert(flat)
output = routed_out + shared_out
```

shared expert 不经过 top-k，所有 token 都走。可以把它理解成一条“共通知识保底通道”；routed experts 再学习更有分化价值的模式。

## 7. 为什么还要 auxiliary loss

如果 router 总选 Expert 0 和 1，其余专家几乎拿不到 token，这叫 routing collapse。Stage 1 记录两种量：

- `importance`：router 给每个专家的平均概率质量；
- `assignment`：top-1 实际分配比例。

```python
balance_proxy = num_experts * torch.sum(importance * assignment)
aux_loss = weight * balance_proxy
```

训练器优化：

```python
loss = lm_loss + aux_loss
```

权重太小可能管不住塌缩，太大又可能牺牲语言建模。这正是 V3 后来改用 selection bias 的原因，详见第 23 章。

## 8. 完整 Block 和整模仍然长什么样

Stage 1 只替换这一行初始化：

```diff
- self.ffn = SwiGLU(...)
+ self.ffn = FineGrainedMoE(config)
```

因此 block 需要额外返回辅助损失：

```python
ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
x = x + ffn_out
return x, aux_loss
```

整模循环收集每层 `aux_loss`，但 logits 和 causal LM loss 的 shape 完全不变：

```text
input_ids [B,T] -> logits [B,T,V]
```

这就是“升级子层，不破坏模型外部接口”。

## 9. 总参数和激活参数

`parameter_count()` 统计所有专家；`activated_parameter_estimate()` 只把 routed expert 参数按 `top_k / num_experts` 计入，再加上始终激活的 attention、shared experts、embedding 和 head。

这个数字仍是估算：router、padding、不同 token 分配和硬件利用率都会影响真实 FLOPs 与速度。

## 10. 如何验证和实验

先跑结构测试：

```bash
python tests/stage_models_test.py
python scripts/inspect_stage_models.py
```

关键不变量：

```text
sum(expert_counts) = num_layers * B * T * top_k
activated_params < total_params
logits shape remains [B, T, V]
```

先跑 DeepSeekMoE 创新链：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_coarse.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_fine_grained.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_shared.json --data data/tinystories.jsonl --hourly_rate 2.18
```

aux 与 bias 是后续 V3 均衡策略对照，不要和 DeepSeekMoE 结构消融混成一组：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/moe_aux.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/moe_bias.json --data data/tinystories.jsonl --hourly_rate 2.18
```

完整控制变量和待填结果见 [`experiments/06_architecture_evolution_plan_zh.md`](../../experiments/06_architecture_evolution_plan_zh.md)。当前旧版 4090 MoE 数字证明链路能运行，但不能代替新的细粒度 expert 公平对照。

结论应按“观察 -> 判断 -> 下一步”写。例如：若细粒度版本负载健康、PPL 稳定更低且吞吐损失可接受，才进入 shared isolation；若结果只在一个 seed 上更好，就增加重复实验，而不是直接升级。

## 你现在应该能回答

- MoE 替换的是 FFN，不是 attention，也不是整个模型。
- total parameters、activated parameters 和真实 tokens/s 是三个不同指标。
- shared expert 始终激活，routed expert 按 token 选择。
- 负载均衡本身会形成一个新的优化目标，因此需要消融。

<!-- tinyseek-nav -->

---

上一篇: [代码优先 Dense LM](12_code_first_dense_lm.md) | [教程目录](README.md) | 下一篇: [DeepSeekMoE 到 DeepSeek-V2](22_from_moe_to_deepseek_v2.md)
