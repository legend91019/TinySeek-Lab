# Math to PyTorch: A Minimal Toolkit for Reading LM Code

This chapter bridges a common gap: recognizing a paper formula is not the same
as knowing how each symbol becomes a tensor dimension and a PyTorch operation.

Every later deep dive follows one order:

```text
formula -> symbols -> tensor shapes -> literal PyTorch -> repository code -> API details
```

## 1. Shape notation

| Symbol | Meaning | Typical value |
| --- | --- | --- |
| $B$ | batch size | 2, 16, 32 |
| $T$ | sequence length | 128, 256 |
| $D$ | hidden size | 192 |
| $H$ | query heads | 4 |
| $H_{kv}$ | key/value heads | 2 or 4 |
| $d_h$ | head dimension, $D/H$ | 48 |
| $V$ | vocabulary size | 260 |
| $E$ | routed experts | 8 |

Example shapes:

```text
input_ids [B,T]
hidden    [B,T,D]
q         [B,H,T,d_h]
logits    [B,T,V]
```

Write shapes down while reading. A shape mismatch usually reveals a conceptual
mismatch before the program runs.

## 2. Modules, parameters, and buffers

An `nn.Module` registers trainable state in `__init__` and defines data flow in
`forward`. Calling `module(x)` enters `forward` while preserving hooks,
autocast, and autograd behavior; training code should not call
`module.forward(x)` directly.

```python
self.weight = nn.Parameter(torch.ones(dim))
```

`nn.Parameter` makes the tensor visible to `model.parameters()`, the optimizer,
and `state_dict()`. RMSNorm's parameter is the learnable scale $\gamma$.

```python
self.register_buffer("rope_cos", cos, persistent=False)
```

A buffer follows `model.to(device)` but is not updated by the optimizer.
`persistent=False` keeps a reproducible RoPE table out of checkpoints. V3's
expert-selection bias is a persistent buffer because it is manually updated
state rather than a gradient-trained parameter.

`nn.ModuleList` registers every child block or expert. A plain Python list does
not reliably expose child parameters to `state_dict()` and the optimizer.

## 3. Elementwise operations, reductions, and broadcasting

For `x [B,T,D]`:

```python
mean_square = x.pow(2).mean(dim=-1, keepdim=True)  # [B,T,1]
scale = torch.rsqrt(mean_square + eps)             # [B,T,1]
y = weight * x * scale                            # [B,T,D]
```

`pow(2)` is elementwise. `mean(dim=-1)` reduces the final dimension.
`keepdim=True` retains a length-one hidden axis. `torch.rsqrt(z)` computes
$1/\sqrt{z}$.

The last line broadcasts `weight [D]` as `[1,1,D]` and `scale [B,T,1]` across
the hidden dimension. Broadcasting compares dimensions right to left; each
pair must match or one side must be 1.

## 4. Reshaping and tensor layout

Attention changes:

```text
[B,T,D] -> [B,T,H,d_h] -> [B,H,T,d_h]
```

with:

```python
q = q_proj(x).view(B, T, H, d_h).transpose(1, 2)
```

- `view` reinterprets shape when the memory layout permits it.
- `reshape` returns a view when possible and otherwise copies.
- `transpose(1, 2)` swaps two dimensions and changes strides.
- `contiguous()` materializes the current logical order as contiguous memory.

That is why the inverse path is:

```python
y = y.transpose(1, 2).contiguous().view(B, T, D)
```

## 5. Embedding and Linear

For embedding table $W_e\in\mathbb{R}^{V\times D}$:

$$e_i=W_e[i].$$

`nn.Embedding(V,D)` maps integer `input_ids [B,T]` to `[B,T,D]` by row lookup;
it does not treat token IDs as continuous values.

For a linear layer:

$$y=xW^T+b.$$

`nn.Linear(D_in,D_out)` stores weight `[D_out,D_in]` and transforms only the
last input dimension. Thus `[B,T,D_in]` becomes `[B,T,D_out]`.

## 6. Activations, probabilities, and selection

SiLU is:

$$\operatorname{SiLU}(x)=x\sigma(x).$$

In SwiGLU, `F.silu(gate(x)) * up(x)` multiplies two `[B,T,D_ff]` tensors
elementwise.

Softmax is:

$$p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.$$

`F.softmax(router_logits, dim=-1)` normalizes over experts for each token.
Choosing the wrong dimension mixes tokens or batches.

```python
top_values, top_indices = torch.topk(probs, k=2, dim=-1)
```

For `[N,E]` router probabilities, both outputs are `[N,2]`. Values become
mixture weights; indices identify selected experts.

`gather(dim,index)` retrieves values at supplied indices. MoE gathers selected
affinities; GRPO gathers the log probability of each observed token.
`nonzero(as_tuple=True)` returns coordinates selected by a boolean mask,
`bincount(..., minlength=E)` measures expert load, and `F.one_hot` converts
expert IDs into indicator vectors.

## 7. Fused attention

The paper formula is:

$$
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}+M\right)V.
$$

A literal implementation is:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
probs = F.softmax(scores, dim=-1)
y = probs @ v
```

TinySeek uses:

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

The fused API performs the same scaling, causal masking, softmax, dropout, and
value aggregation and can select an efficient GPU kernel. It does not remove
softmax from the mathematics.

## 8. Cross entropy and token log probability

For correct class $y$:

$$
\mathcal{L}=-\log\frac{e^{z_y}}{\sum_j e^{z_j}}.
$$

`F.cross_entropy(logits,targets)` is a stable fused log-softmax plus negative
log-likelihood. LM code first shifts to `[B,T-1,V]` and `[B,T-1]`, then flattens
to `[B*(T-1),V]` and `[B*(T-1)]`. `ignore_index=-100` removes padding or SFT
prompt positions from the mean.

GRPO needs the log probabilities themselves:

```python
log_probs = F.log_softmax(logits, dim=-1)
token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
```

`unsqueeze(-1)` creates the index axis expected by `gather`; `squeeze(-1)`
removes it afterward.

## 9. Autograd boundaries

- `loss.backward()` accumulates gradients into parameter `.grad` fields.
- `tensor.detach()` shares data without extending the gradient graph.
- `torch.no_grad()` avoids graph construction during evaluation, reference
  scoring, and manual V3 bias updates.
- `requires_grad_(False)` freezes reference-model parameters.
- `optimizer.zero_grad(set_to_none=True)` clears accumulated gradients with
  lower memory traffic than filling them with zeros.

## 10. Check yourself

1. What is the result of `x [2,128,192].mean(dim=-1, keepdim=True)`?
2. What is the shape of `q [2,4,128,48] @ k.transpose(-2,-1)`?
3. What do the values and indices from `topk([256,8], k=2)` represent?

The answers are `[2,128,1]`, `[2,4,128,128]`, and the two selected expert
probabilities and IDs for each token. The next chapter puts these tools into a
complete Dense LM.

<!-- tinyseek-nav -->

---

Previous: [Architecture Evolution Map](20_architecture_evolution_overview.md) | [Tutorial Index](README.md) | Next: [Code First Dense LM](12_code_first_dense_lm.md)
