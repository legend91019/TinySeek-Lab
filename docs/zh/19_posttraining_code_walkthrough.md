# 19. 后训练代码细读：SFT Masking 和 GRPO Objective

这一章专门补后训练代码。目标不是把 RLHF 讲成工业系统，而是让读者看懂：

- SFT 为什么只训练 response token；
- cold-start 数据在代码里到底是什么；
- GRPO mini 如何采样一组回答、打分、归一化 advantage；
- 为什么当前 GRPO 只能算教学版。

主要文件：

- [`dataset/lm_dataset.py`](../../dataset/lm_dataset.py)
- [`trainer/train_sft.py`](../../trainer/train_sft.py)
- [`trainer/train_grpo.py`](../../trainer/train_grpo.py)
- [`scripts/prepare_toy_sft_data.py`](../../scripts/prepare_toy_sft_data.py)
- [`scripts/prepare_toy_grpo_data.py`](../../scripts/prepare_toy_grpo_data.py)

## SFT 数据格式

SFT JSONL 使用：

```json
{"prompt": "Explain RMSNorm.", "response": "RMSNorm rescales each hidden vector by its root mean square."}
```

`JsonlInstructionDataset` 会把它变成：

```text
### Instruction
Explain RMSNorm.

### Response
RMSNorm rescales each hidden vector by its root mean square.
```

这就是本仓里的 cold-start SFT：先让模型学会“看到 instruction 后，用规整 response
格式回答”。如果 response 写成逐步推理格式，模型学到的就是更可读的推理外形。

## SFT Masking 是关键

核心代码在 `JsonlInstructionDataset.__getitem__`：

```python
prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
response_ids = tokenizer.encode(response_text, add_bos=False, add_eos=True)
ids = (prompt_ids + response_ids)[: self.max_seq_len]
labels = ids.copy()
labels[:prompt_len] = [-100] * prompt_len
```

含义：

```text
input_ids:  [BOS, prompt tokens..., response tokens..., EOS]
labels:     [-100, -100, ..., response tokens..., EOS]
```

`-100` 是 PyTorch cross entropy 的 ignore index。也就是说：

- prompt token 只作为上下文；
- response token 才产生 loss；
- 模型不会被训练成“复读 prompt”；
- 模型会被训练成“在 prompt 后生成 response”。

令 $m_j\in\{0,1\}$ 表示 target token $y_j$ 是否参与监督：

$$
L_{SFT}=-\frac{1}{\sum_t m_{t+1}}
\sum_t m_{t+1}\log p_\theta(y_{t+1}\mid y_{\le t}),
$$

prompt 与 pad target 的 mask 为 0，response target 为 1。边界对齐是：

```text
最后一个 prompt 位置的 logit -> 第一个 response token -> mask 1
response 内部位置的 logit     -> 下一个 response/EOS  -> mask 1
预测 prompt token 的 logit    -> prompt target         -> mask 0
```

代码没有单独把 mask 乘进
loss，而是把对应 target 改成 `-100`，交给 `F.cross_entropy(ignore_index=-100)`
在归约时排除。`ids.copy()` 很重要：若直接写 `labels = ids`，随后修改 labels 也会
破坏输入 token 列表。

这也是“冷启动数据是不是 SFT”的代码答案：在本仓里，是的，它就是一种 SFT。
区别只在于数据内容偏“可读、规整、推理格式”，而不是普通问答。

## SFT Trainer 和 Pretrain Trainer 的区别

`train_sft.py` 和 `train_pretrain.py` 大部分相同：

```python
out = model(input_ids, labels)
loss = out["loss"] + out["aux_loss"]
loss.backward()
optimizer.step()
```

真正的区别在 dataset：

| 阶段 | Dataset | 哪些 token 有 loss |
| --- | --- | --- |
| Pretrain | `JsonlTextDataset` | 所有非 pad 文本 token |
| SFT | `JsonlInstructionDataset` | response token |

所以 SFT 不神秘。它主要是换了样本格式和 label mask。

## GRPO Mini 的数据格式

GRPO JSONL 使用：

```json
{"prompt": "12+9", "answer": "21"}
```

`JsonlPromptDataset` 只保存 prompt 和可验证 answer。这里没有 response，因为
response 要由当前 policy 采样出来。

## GRPO Mini 的一轮训练

`train_grpo.py` 每轮大致做：

```text
prompt -> sample group completions -> rule reward
-> group-normalized advantage -> policy logprob
-> reference KL proxy -> update policy
```

对应代码：

```python
group_sequences = sample_group(policy, tokenizer, prompt, grpo_cfg, model_cfg, device)
rewards = torch.tensor([rule_reward(seq["completion"], answer) for seq in group_sequences])
advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
```

对同一 prompt 的 $G$ 个 completion：

$$
A_i=\frac{r_i-\mu_r}{\sigma_r+\epsilon},\qquad
\mu_r=\frac{1}{G}\sum_i r_i.
$$

`torch.tensor([...],device=device)` 把 Python reward 放到 GPU；`.mean()` 得均值；
`.std(unbiased=False)` 使用总体标准差（分母是 $G$），适合当前完整采样组；
`1e-6` 处理所有 reward 相同时的零方差。归一化后，组内高于平均的回答 advantage
为正，低于平均的为负。

GRPO 的核心直觉是：同一个 prompt 下采样多个回答，不一定需要训练一个 value model；
可以在组内比较“哪些回答比平均水平好”，然后提高这些回答的概率。

## Policy Loss

先看一个 completion 的 log probability 如何计算：

```python
logits = model(ids)["logits"][:, :-1, :]
targets = ids[:, 1:]
logp = F.log_softmax(logits, dim=-1)
logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
completion_logp = logp[:, completion_mask].mean()
```

公式是：

$$
\ell_\theta(y)=\frac{1}{|y|}\sum_{t\in completion}
\log p_\theta(y_t\mid x,y_{<t}).
$$

`log_softmax(...,dim=-1)` 在词表轴产生 `[B,T-1,V]`；`unsqueeze(-1)` 把真实
token id 变成 `[B,T-1,1]`；`gather(-1,...)` 每个位置只取目标 token 的 logp；
`squeeze(-1)` 恢复 `[B,T-1]`。位置 mask 去掉 prompt，只对 completion 求均值。

当前教学版 loss 是：

```python
policy_logp = completion_logprob(policy, ids, prompt_len)
ref_logp = completion_logprob(ref, ids, prompt_len)
kl_proxy = (policy_logp - ref_logp).pow(2)
loss = (-adv * policy_logp) + kl_beta * kl_proxy
```

教学目标写成：

$$
L_i=-A_i\ell_\theta(y_i)+\beta
\left(\ell_\theta(y_i)-\ell_{ref}(y_i)\right)^2.
$$

`adv.detach()` 把 reward/advantage 当常数，不让 autograd 穿过规则 reward；reference
logp 放在 `torch.no_grad()` 中，且 reference 参数已 `requires_grad_(False)`。
`.pow(2)` 是教学用的对称距离 proxy，不是论文 KL 的无偏逐 token 估计。

读法：

- `adv > 0`：这个 completion 比组内平均好，增加它的 log probability。
- `adv < 0`：这个 completion 比组内平均差，降低它的 log probability。
- `kl_proxy`：不要让 policy 离初始化 reference 太远。

这不是完整工业 GRPO，但它保留了教学上最重要的形状：group sampling、rule reward、
relative advantage、reference constraint。

它没有保存采样时的 old-policy log probability，也没有计算 clipped importance ratio；因此这里的 `GRPO Mini` 是受 GRPO 启发的教学目标，而不是论文目标的逐项复刻。

## Rule Reward

当前 reward 面向简单算术：

```python
if pred is not None:
    reward += 0.1
if "answer" in completion.lower() or "final" in completion.lower():
    reward += 0.1
if pred == target:
    return 1.0
```

它有两类信号：

- shaping reward：输出数字、出现 answer/final 这类格式词；
- exact reward：最终整数正确。

正式套件把 cold-start response 统一成：

```text
<think>concise arithmetic trace</think>
<answer>verified integer</answer>
```

更严肃的对照是：

1. pretrained base -> direct GRPO；
2. pretrained base -> structured SFT；
3. structured SFT -> GRPO；
4. 分别比较 tagged-answer accuracy、reasoning-format score 与 PPL。

## 为什么当前 GRPO 不能过度宣传

当前实现有几个刻意简化：

- 任务很小；
- reward 很粗糙；
- 没有大规模采样；
- 没有稳定的 reference/KL 工程；
- 没有 old-policy ratio 和 clipping；
- 只有 300-step toy run，不是大规模或稳定性结果；
- mini eval 也很轻。

所以报告里必须写清楚：它是教学版 GRPO，不是 DeepSeek-R1 级别复现。

## 下一次上卡要补什么数据

已经完成的[正式报告](../../experiments/gpu_completion_runs/report_zh.md)现在包含：

- base、direct GRPO、SFT-only、SFT+GRPO 的 Reasoning Answer / Format；
- 四个 checkpoint 的 PPL / Add / Copy / QA；
- `mean_reward` 曲线；
- GRPO 是否牺牲 PPL；
- 样例 completion 对比。
- 已记录训练/后训练进程时间、估算费用和明确的排除范围。

<!-- tinyseek-nav -->

---

上一篇: [阶段 6：GRPO Mini](08_stage6_grpo_mini.md) | [教程目录](README.md) | 下一篇: [仓库路线图](09_repository_roadmap.md)
