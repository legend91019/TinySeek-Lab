# 22. 从 DeepSeekMoE 到 DeepSeek-V2：MLA 代码怎么写

DeepSeekMoE 解决 FFN 的稀疏激活，attention 仍然要为每个历史 token 保存 K/V。DeepSeek-V2 把两条路线合在一起：block 继续使用 DeepSeekMoE，同时把 attention 换成 MLA。

对照代码：

- 上一代：[`stage1_deepseek_moe.py`](../../model/stages/stage1_deepseek_moe.py)
- 新一代：[`stage2_deepseek_v2.py`](../../model/stages/stage2_deepseek_v2.py)
- 正式实验实现：[`CausalSelfAttention`](../../model/tinyseek.py)

## 本章研究卡：先证明 cache 是问题

在改代码前，先用上一代配置计算每层、每 token 的 KV 元素数：

```text
GQA cache = 2 * num_kv_heads * head_dim
本实验       = 2 * 2 * 48 = 192 elements/token/layer
```

它还会乘以 batch、序列长度和层数。这个公式能证明缓存增长趋势，但不能代替真实 decode profiler。

本章依次检验两个假设：

| 步骤 | 候选 | 想验证什么 | 当前可观测量 |
| --- | --- | --- | --- |
| B0 | GQA control | 上一代基线 | `192` 个理论元素；实测 PPL `2.009 +/- 0.011` |
| B1 | 朴素低秩 KV | 低秩是否能保住语言建模质量 | 实测 PPL `2.190 +/- 0.001`；由于 RoPE 耦合，不宣称 latent-only cache |
| B2 | 解耦 RoPE 的 educational MLA | content latent 与位置分量分开后，应缓存量是否下降 | `72` 个理论元素，实测 PPL `2.194 +/- 0.012`；真实 cached decoding 未实现 |

**决策门槛**：B1/B2 的多 seed validation PPL 不应显著差于 B0；B2 必须在理论账本上明显减少缓存。即使两项都通过，也只能升级“结构研究结论”。要声称真实显存或吞吐收益，还必须实现 cached decoding，并测长上下文下的峰值显存、首 token 后吞吐和延迟。

**实测决定**：B2 通过理论缓存账本，却没有通过质量门槛；B1 说明在当前 rank 下，低秩本身已经带来明显代价。因此保留 GQA，下一轮先 sweep latent rank。完整运行见[架构实测报告](../../experiments/architecture_lab_runs/report_zh.md)。

## 1. 上一代哪里不够

生成第 `t` 个 token 时，attention 需要访问之前所有 token 的 K/V。普通 GQA 每层每个历史 token 需要缓存：

```text
2 * num_kv_heads * head_dim
```

`2` 分别代表 K 和 V。上下文长度、batch、层数增加时，缓存近似线性增长。GQA 减少 `num_kv_heads`，但压缩得太激进可能损害 attention 表达能力。

## 2. 论文 MLA 的核心动机

DeepSeek-V2 使用 low-rank key-value joint compression。直觉是：不要直接缓存完整 K/V，先保存更短的 latent；需要 attention 时，再从 latent 恢复 content K/V。

论文还把与 RoPE 有关的位置分量解耦，因为直接把 RoPE 旋转混进低秩压缩路径，会破坏方便吸收投影矩阵、压缩缓存的推理形式。

论文相对 DeepSeek 67B 报告 KV cache 降低 93.3%、最大生成吞吐达到 5.76 倍。这是论文系统的结果，不是 TinySeek 教学实现的实测结论。

## 3. TinySeek 保留和简化了什么

保留：

- 低秩 `compressed_kv`；
- 从 latent 重建 content K 和 V；
- 单独的 K RoPE 路径；
- Q 的 content/rope 拆分；
- 理论缓存元素计算。

简化：

- 没有 fused kernel；
- 没有真正把 latent 跨 decode step 存入 cache；
- forward 中仍显式重建 K/V；
- 维度和投影比论文更小、更规整。

所以类名是 `EducationalMLA`。它用于理解数据流和做结构消融，不是生产级 MLA。

统一模型为兼容旧版 v1 checkpoint，保留 `mla_decoupled_rope=false` 的旧低秩 K/V 分支；该分支不再声称 latent-only cache。新的公平实验配置显式设置 `mla_decoupled_rope=true` 和 `qk_rope_head_dim=8`，走与本章相同的 content/RoPE 解耦路径。

## 4. Config 如何确定维度

```python
@dataclass
class Stage2Config(Stage1Config):
    kv_lora_rank: int = 64
    qk_rope_head_dim: int = 16
```

每个 attention head 的总维度仍是：

```python
head_dim = hidden_size // num_heads
content_head_dim = head_dim - qk_rope_head_dim
```

例如 `hidden=192, heads=4`：

```text
head_dim = 48
rope_head_dim = 16
content_head_dim = 32
```

Q/K 最后重新拼成 48 维，所以 scaled dot-product attention 的接口不变。

## 5. Q 路径：内容和位置分开

```python
q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
q_content, q_rope = torch.split(
    q, [self.content_head_dim, self.rope_head_dim], dim=-1
)
q_rope = apply_rope(q_rope, self.rope_cos, self.rope_sin)
q = torch.cat((q_content, q_rope), dim=-1)
```

shape 是：

```text
q_content: [B, H, T, content_dim]
q_rope:    [B, H, T, rope_dim]
q:         [B, H, T, head_dim]
```

只有 `q_rope` 旋转；content path 保持低秩内容投影的语义。

## 6. KV 路径：先压缩，再展开

第一步只得到一个短 latent：

```python
compressed_kv = self.kv_down(x)
```

```text
x:             [B, T, D]
compressed_kv: [B, T, R]
```

`R=kv_lora_rank`。再从同一个 latent 重建：

```python
k_content = self.k_content_up(compressed_kv).view(
    batch, seq_len, self.num_kv_heads, self.content_head_dim
).transpose(1, 2)
v = self.v_up(compressed_kv).view(
    batch, seq_len, self.num_kv_heads, self.head_dim
).transpose(1, 2)
k_content = repeat_kv(k_content, self.kv_repeats)
v = repeat_kv(v, self.kv_repeats)
```

两次 `view(...).transpose(1,2)` 把线性层的 packed 输出恢复成
`[B,H_kv,T,dim]`；`repeat_kv` 再把 KV heads 对齐到 query heads。这些 shape
操作只为 attention 接口服务，不改变低秩 latent 本身。

这三层线性映射对应低秩分解：

$$
c_t=W_{down}x_t\in\mathbb{R}^{R},\qquad
k_t^{content}=W_{K,up}c_t,\qquad v_t=W_{V,up}c_t.
$$

`nn.Linear(D,R)` 生成共同 latent，两个 `nn.Linear(R,...)` 重建内容 K 与 V。
$R\ll D$ 使它成为低秩路径，但是否保持质量仍要由实验回答。

这就是 joint compression：K content 和 V 共享一个可缓存 latent，而不是各自保存完整历史张量。

K 的 RoPE 分量单独来自 hidden states：

```python
k_rope = self.k_rope_proj(x).view(
    batch, seq_len, 1, self.rope_head_dim
).transpose(1, 2)
k_rope = apply_rope(k_rope, self.rope_cos, self.rope_sin)
k_rope = k_rope.expand(batch, self.num_heads, seq_len, self.rope_head_dim)
k = torch.cat((k_content, k_rope), dim=-1)
```

`torch.split(q,[content_dim,rope_dim],dim=-1)` 分离内容与位置特征；
`torch.cat(...,dim=-1)` 沿最后一维拼回 `head_dim`；`expand` 则让位置 K 在
heads 间共享逻辑视图。

教学实现让 `k_rope` 在 heads 间共享，以便缓存估算为：

```text
kv_lora_rank + qk_rope_head_dim
```

即每个历史 token 理论保存：

$$C_{MLA/token}=R+d_{rope},$$

而 GQA 保存 $2H_{kv}d_h$ 个 K/V 元素。`kv_cache_elements_per_token()` 只是返回
这个公式，不测实际 CUDA 显存，所以报告称它为理论 cache 元素数。

## 7. 为什么训练时还看得到完整 K/V

训练通常一次处理整段序列。代码把 latent 展开为 K/V 后调用：

```python
scaled_dot_product_attention(q, k, v, is_causal=True)
```

这并不否定 latent cache 的动机。真正的推理优化要在逐 token decoding 时只缓存 latent 和 RoPE 分量，并通过矩阵吸收或专门 kernel 避免反复物化巨大 K/V。

因此必须区分：

- **训练结构正确性**：模型能通过 latent 路径学习 attention。
- **理论缓存大小**：按应缓存的 latent 元素估算。
- **实际推理显存/吞吐**：需要 cached generation 实现后才能测。

## 8. 完整 V2 Block

V2 不是“只剩 MLA”。Stage 2 block 是：

```python
self.attn = EducationalMLA(config)
self.ffn = FineGrainedMoE(config)
```

forward 仍然是两个 pre-norm residual 子层：

```python
x = x + self.attn(self.attn_norm(x))
ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
x = x + ffn_out
```

这体现了 DeepSeek 的演进方式：V2 继承已经验证过的 MoE 路线，再升级 attention。

## 9. 如何验证

```bash
python tests/stage_models_test.py
python scripts/inspect_stage_models.py
```

检查：

- logits 仍是 `[B,T,V]`；
- `q` 和 `k` 拼接后最后一维一致；
- `kv_cache_elements_per_token()` 等于 `rank + rope_dim`；
- V2 block 仍包含 MoE，而不是意外退回 Dense FFN。

## 10. 公平实验怎么做

先测“仅低秩投影”这一步：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_control.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_low_rank_kv.json --data data/tinystories.jsonl --hourly_rate 2.18
```

这组实验主要检查低秩 attention 的质量代价；`mla_decoupled_rope=false`，所以不得把它写成 latent-only cache 收益。

再测带解耦 RoPE 的教学版 MLA：

```bash
python trainer/train_pretrain.py --config configs/architecture_lab/v2_attention_control.json --data data/tinystories.jsonl --hourly_rate 2.18
python trainer/train_pretrain.py --config configs/architecture_lab/v2_mla.json --data data/tinystories.jsonl --hourly_rate 2.18
```

两个配置只改变 `attention_impl`。需要同时记录 validation loss、理论 KV 元素、训练显存和 tokens/s。当前统一 trainer 没有真实 latent cache，所以不能用训练显存直接证明论文的推理缓存收益。

如果 MLA 的 PPL 更差，先 sweep `mla_latent_size`，画出“质量 - 理论 cache”Pareto 曲线；不要只保留一个对 MLA 有利的 rank。若理论 cache 减少但训练 tokens/s 下降，这并不矛盾：当前教学 forward 会重建 K/V，研究问题本来就不是训练 kernel 加速。

结果表见 [`experiments/06_architecture_evolution_plan_zh.md`](../../experiments/06_architecture_evolution_plan_zh.md)。下一章将在 V2 基础上修改 router 的均衡方法，并把 MTP loss 接进整模 forward 和训练循环。

<!-- tinyseek-nav -->

---

上一篇: [Dense 到 DeepSeekMoE](21_from_dense_to_deepseek_moe.md) | [教程目录](README.md) | 下一篇: [DeepSeek-V2 到 DeepSeek-V3](23_from_v2_to_deepseek_v3.md)
