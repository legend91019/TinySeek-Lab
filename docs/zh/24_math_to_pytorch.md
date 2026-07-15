# 数学到 PyTorch：读懂语言模型代码的最小工具箱

这一章解决一个常见断层：你可能看懂了论文公式，也可能认识 `torch` 的函数名，
但还不知道公式里的每一项怎样变成代码里的某个维度、某一行运算。

后续章节统一使用下面的阅读顺序：

```text
公式 -> 符号含义 -> 张量 shape -> 朴素 PyTorch -> 仓库代码 -> API 细节
```

## 1. 本教程的 shape 记号

| 符号 | 含义 | 常见数值 |
| --- | --- | --- |
| $B$ | batch size | 2、16、32 |
| $T$ | sequence length | 128、256 |
| $D$ | hidden size | 192 |
| $H$ | query head 数 | 4 |
| $H_{kv}$ | KV head 数 | 2 或 4 |
| $d_h$ | head dim，$D/H$ | 48 |
| $V$ | vocabulary size | 260 |
| $E$ | routed expert 数 | 8 |

例如：

```text
input_ids [B, T] = [2, 128]
hidden     [B, T, D] = [2, 128, 192]
q          [B, H, T, d_h] = [2, 4, 128, 48]
logits     [B, T, V] = [2, 128, 260]
```

读代码时，先在纸上写 shape。只要一次运算前后的 shape 对不上，数学含义通常也
没有真正对上。

## 2. `nn.Module`、`Parameter` 和 buffer

### `nn.Module`

一个可训练组件通常继承 `nn.Module`：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x
```

`__init__` 注册参数和子模块，`forward` 定义数据怎样流动。调用 `norm(x)` 时，
PyTorch 会进入 `forward(x)`，同时保留 hook、autocast 和 autograd 所需的机制；
不要在训练代码里手动调用 `norm.forward(x)`。

### `nn.Parameter`

`nn.Parameter(torch.ones(D))` 是一个需要优化器更新的张量。把普通 tensor 赋给
module 属性不会自动成为模型参数，而 `Parameter` 会出现在：

```python
list(model.parameters())
model.state_dict()
```

RMSNorm 的 `weight` 对应公式中的可学习缩放 $gamma$。

### `register_buffer`

RoPE 的 cos/sin 表和 V3 路由 bias 使用：

```python
self.register_buffer("rope_cos", cos, persistent=False)
```

buffer 会跟随 `model.to(device)` 移动，但默认不参与梯度更新。`persistent=False`
表示它不写入 checkpoint，因为 RoPE 表可以从配置重新计算。V3 的 `expert_bias`
则是持久 buffer：它不是 optimizer parameter，但要保存当前路由状态。

### `nn.ModuleList`

普通 Python list 不会可靠注册其中的子模块。本仓用：

```python
self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])
```

这样每个 block 的参数都会进入 `state_dict()` 和 optimizer。

## 3. 逐元素运算、归约和广播

RMSNorm 的核心代码是：

```python
x.pow(2).mean(dim=-1, keepdim=True)
```

- `x.pow(2)`：逐元素平方，shape 不变，`[B,T,D] -> [B,T,D]`。
- `.mean(dim=-1)`：沿最后一维 $D$ 求平均。
- `dim=-1`：最后一个维度，不管前面有几个 batch 维都成立。
- `keepdim=True`：保留被归约的轴，结果是 `[B,T,1]`，不是 `[B,T]`。

随后：

```python
scale = torch.rsqrt(mean_square + eps)  # [B,T,1]
y = weight * x * scale                 # [D] * [B,T,D] * [B,T,1]
```

`torch.rsqrt(z)` 等于 $1/\sqrt{z}$。最后一行依赖 broadcasting：

- `[D]` 被看成 `[1,1,D]`；
- `[B,T,1]` 在 hidden 维复制；
- 两者都能和 `[B,T,D]` 逐元素相乘。

广播不会真的复制整块内存，但必须从右向左检查维度：两个维度要么相等，要么
其中一个是 1。

## 4. `reshape`、`view`、`transpose` 和 `contiguous`

Attention 经常把：

```text
[B,T,D] -> [B,T,H,d_h] -> [B,H,T,d_h]
```

对应代码：

```python
q = self.q_proj(x).view(B, T, H, d_h).transpose(1, 2)
```

- `view`：在内存布局允许时，只改变张量的 shape 解释。
- `reshape`：尽量返回 view；不连续时会自动复制，通常更宽容。
- `transpose(1, 2)`：交换第 1、2 维，shape 变了，底层 stride 也变了。
- `contiguous()`：按当前逻辑顺序重新排成连续内存。

所以 attention 输出恢复时写成：

```python
y = y.transpose(1, 2).contiguous().view(B, T, D)
```

如果省略 `contiguous()`，后面的 `view` 可能报错，因为 transpose 后的 tensor
通常不是连续布局。`reshape` 有时会替你复制而不报错，但理解内存布局后再选 API
更可靠。

## 5. `nn.Embedding` 与 `nn.Linear`

### Embedding 是查表

设 embedding 表 $W_e\in\mathbb{R}^{V\times D}$，token id 为 $i$：

$$
e_i = W_e[i].
$$

```python
self.embed = nn.Embedding(V, D)
x = self.embed(input_ids)
```

`input_ids` 必须是整数类型，shape 从 `[B,T]` 变成 `[B,T,D]`。它不是把 token id
当连续数字做线性变换，而是按 id 取矩阵的一行。

### Linear 作用在最后一维

对 $x\in\mathbb{R}^{D_{in}}$：

$$
y=xW^T+b.
$$

`nn.Linear(D_in, D_out)` 的权重 shape 是 `[D_out,D_in]`。输入即使是
`[B,T,D_in]`，它也只变换最后一维，输出 `[B,T,D_out]`，前面的 $B,T$ 被当成
独立样本维。

## 6. 激活、概率与选择

### `F.silu`

$$
\operatorname{SiLU}(x)=x\sigma(x).
$$

SwiGLU 使用：

```python
F.silu(gate(x)) * up(x)
```

两个分支 shape 都是 `[B,T,D_ff]`，星号是逐元素门控，不是矩阵乘法。

### `F.softmax`

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

`F.softmax(router_logits, dim=-1)` 在 expert 维归一化，使每个 token 对所有 expert
的概率和为 1。`dim` 写错会把不同 token 或不同 batch 混在一起。

### `torch.topk`

```python
top_weights, top_indices = torch.topk(probs, k=2, dim=-1)
```

它返回值和下标，shape 都从 `[N,E]` 变成 `[N,2]`。MoE 用下标决定 token 发给
哪些 expert，用值决定 expert 输出权重。

### `gather`

`gather(dim, index)` 按给定下标从某个维度取值：

```python
chosen = affinity.gather(1, top_indices)
```

`affinity [N,E]`、`top_indices [N,K]`，结果是 `[N,K]`。GRPO 中同一 API 从
`log_probs [B,T,V]` 取出真实 token 对应的 log probability。

### `nonzero`、`bincount` 和 `one_hot`

- `mask.nonzero(as_tuple=True)`：返回满足布尔条件的各维下标。
- `torch.bincount(ids, minlength=E)`：统计每个 expert 收到多少次选择。
- `F.one_hot(ids, num_classes=E)`：把 expert id 变成 one-hot 向量。

它们在教学版 MoE 中让路由过程可读，但 Python expert 循环不是高性能分布式
dispatch 实现。

## 7. Attention 的 fused API

论文公式是：

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}+M\right)V.
$$

朴素 PyTorch 可以写成：

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
probs = F.softmax(scores, dim=-1)
y = probs @ v
```

仓库使用：

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

这个 fused API 完成同一数学路径，并可在支持的 GPU 上选择更高效内核。
`is_causal=True` 表示位置 $t$ 不能看到未来位置。它不是“省略了 softmax”，而是
把缩放、mask、softmax、dropout 和乘 $V$ 封装起来。

## 8. Cross Entropy 与 log probability

对正确类别 $y$，单个位置的交叉熵是：

$$
\mathcal{L}=-\log\frac{e^{z_y}}{\sum_j e^{z_j}}
=-\log\operatorname{softmax}(z)_y.
$$

```python
F.cross_entropy(logits, targets)
```

数值上等价于 `log_softmax` 后取正确类别并求负均值，但 fused 实现更稳定。语言
模型先右移成 `[B,T-1,V]` 和 `[B,T-1]`，再展平为 `[B*(T-1),V]`、
`[B*(T-1)]`；`ignore_index=-100`
让 pad 或 SFT prompt token 不进入平均值。

GRPO 需要保留 log probability 本身：

```python
log_probs = F.log_softmax(logits, dim=-1)
token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
```

`unsqueeze(-1)` 把 targets 从 `[B,T]` 变成 `[B,T,1]`，供 `gather` 使用；
`squeeze(-1)` 再移除长度为 1 的轴。

## 9. Autograd 边界

- `loss.backward()`：沿计算图累加每个 parameter 的 `.grad`。
- `tensor.detach()`：返回共享数据但不继续追踪梯度的 tensor，常用于日志。
- `with torch.no_grad()`：块内不构建计算图，适合评测、采样 reference 和手动更新
  V3 expert bias。
- `param.requires_grad_(False)`：冻结 reference model 参数。
- `optimizer.zero_grad(set_to_none=True)`：把梯度设为 `None`，通常比填零更省内存；
  因为 backward 默认累加梯度，所以每次 optimizer step 后必须清理。

## 10. 三个检查题

1. `x [2,128,192]` 做 `mean(dim=-1, keepdim=True)` 后是什么 shape？为什么不能
   随便删掉 `keepdim=True`？
2. `q [2,4,128,48]` 与 `k.transpose(-2,-1) [2,4,48,128]` 相乘后是什么 shape？
3. `router_probs [256,8]` 做 `topk(k=2, dim=-1)` 后，值和下标分别表示什么？

答案分别是 `[2,128,1]`；`[2,4,128,128]`；以及每个 token 选中的两个 expert
概率与 expert id。接下来进入 Dense 章节，把这些工具放进第一个完整模型。

<!-- tinyseek-nav -->

---

上一篇: [四代架构演进总览](20_architecture_evolution_overview.md) | [教程目录](README.md) | 下一篇: [代码优先 Dense LM](12_code_first_dense_lm.md)
