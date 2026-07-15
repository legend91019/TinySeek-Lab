# 19. Post-Training Code Walkthrough: SFT Masking and GRPO Objective

This chapter explains the post-training code. The goal is not to present an
industrial RLHF stack, but to make the reader understand:

- why SFT trains only response tokens;
- what cold-start data means in code;
- how GRPO mini samples a group, scores completions, and normalizes advantages;
- why the current GRPO path is educational.

Main files:

- [`dataset/lm_dataset.py`](../dataset/lm_dataset.py)
- [`trainer/train_sft.py`](../trainer/train_sft.py)
- [`trainer/train_grpo.py`](../trainer/train_grpo.py)
- [`scripts/prepare_toy_sft_data.py`](../scripts/prepare_toy_sft_data.py)
- [`scripts/prepare_toy_grpo_data.py`](../scripts/prepare_toy_grpo_data.py)

## SFT Data Format

SFT JSONL uses:

```json
{"prompt": "Explain RMSNorm.", "response": "RMSNorm rescales each hidden vector by its root mean square."}
```

`JsonlInstructionDataset` formats it as:

```text
### Instruction
Explain RMSNorm.

### Response
RMSNorm rescales each hidden vector by its root mean square.
```

In this repo, cold-start SFT means teaching the model a readable response shape
before rule RL. If responses contain step-by-step reasoning, the model learns
that reasoning format.

## SFT Masking

The key code is in `JsonlInstructionDataset.__getitem__`:

```python
prompt_ids = tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
response_ids = tokenizer.encode(response_text, add_bos=False, add_eos=True)
ids = (prompt_ids + response_ids)[: self.max_seq_len]
labels = ids.copy()
labels[:prompt_len] = [-100] * prompt_len
```

Meaning:

```text
input_ids:  [BOS, prompt tokens..., response tokens..., EOS]
labels:     [-100, -100, ..., response tokens..., EOS]
```

`-100` is PyTorch cross entropy's ignore index. So:

- prompt tokens are context only;
- response tokens produce loss;
- the model is not trained to copy the prompt;
- the model is trained to generate the response after the prompt.

Let $m_j\in\{0,1\}$ mark whether target token $y_j$ is supervised:

$$
L_{SFT}=-\frac{1}{\sum_t m_{t+1}}
\sum_t m_{t+1}\log p_\theta(y_{t+1}\mid y_{\le t}).
$$

Prompt/padding targets have mask 0; response targets have 1. The causal boundary
is:

```text
last prompt logit -> first response token -> mask 1
response logit    -> next response/EOS    -> mask 1
logit predicting a prompt token           -> mask 0
```

Rather than
multiplying an explicit mask, the dataset writes target `-100` and lets
`F.cross_entropy(ignore_index=-100)` exclude those positions from reduction.
`ids.copy()` prevents label masking from mutating the input token list.

So yes: in this repo, cold-start data is implemented as SFT. The special part is
the data content, not a new objective.

## SFT Trainer vs Pretrain Trainer

`train_sft.py` and `train_pretrain.py` share the same core:

```python
out = model(input_ids, labels)
loss = out["loss"] + out["aux_loss"]
loss.backward()
optimizer.step()
```

The real difference is the dataset:

| Stage | Dataset | Tokens with loss |
| --- | --- | --- |
| Pretrain | `JsonlTextDataset` | all non-pad text tokens |
| SFT | `JsonlInstructionDataset` | response tokens |

SFT is mostly sample formatting plus label masking.

## GRPO Mini Data Format

GRPO JSONL uses:

```json
{"prompt": "12+9", "answer": "21"}
```

`JsonlPromptDataset` stores prompts and verifiable answers. There is no response,
because the current policy must sample completions.

## One GRPO Mini Step

`train_grpo.py` does:

```text
prompt -> sample group completions -> rule reward
-> group-normalized advantage -> policy logprob
-> reference KL proxy -> update policy
```

Code:

```python
group_sequences = sample_group(policy, tokenizer, prompt, grpo_cfg, model_cfg, device)
rewards = torch.tensor([rule_reward(seq["completion"], answer) for seq in group_sequences])
advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
```

For $G$ completions of one prompt:

$$
A_i=\frac{r_i-\mu_r}{\sigma_r+\epsilon},\qquad
\mu_r=\frac{1}{G}\sum_i r_i.
$$

`torch.tensor(...,device=device)` moves Python rewards to the GPU. `.mean()`
computes $\mu_r$; `.std(unbiased=False)` uses population variance with divisor
$G$; epsilon handles a group whose rewards are all equal.

The core idea: sample several completions for the same prompt, compare them
inside the group, and increase the probability of above-average completions.

## Policy Loss

First trace completion log probability:

```python
logits = model(ids)["logits"][:, :-1, :]
targets = ids[:, 1:]
logp = F.log_softmax(logits, dim=-1)
logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
completion_logp = logp[:, completion_mask].mean()
```

$$
\ell_\theta(y)=\frac{1}{|y|}\sum_{t\in completion}
\log p_\theta(y_t\mid x,y_{<t}).
$$

`log_softmax` produces `[B,T-1,V]`. `unsqueeze(-1)` creates target indices
`[B,T-1,1]`; `gather` selects one token log probability per position;
`squeeze(-1)` returns `[B,T-1]`. The position mask removes the prompt.

The teaching loss is:

```python
policy_logp = completion_logprob(policy, ids, prompt_len)
ref_logp = completion_logprob(ref, ids, prompt_len)
kl_proxy = (policy_logp - ref_logp).pow(2)
loss = (-adv * policy_logp) + kl_beta * kl_proxy
```

The teaching objective is:

$$
L_i=-A_i\ell_\theta(y_i)+\beta
\left(\ell_\theta(y_i)-\ell_{ref}(y_i)\right)^2.
$$

`adv.detach()` treats rule-derived advantage as a constant. Reference scoring
runs under `torch.no_grad()` with frozen parameters. `.pow(2)` is a symmetric
teaching proxy, not an unbiased token-level estimate of the paper KL term.

Interpretation:

- `adv > 0`: increase this completion's log probability.
- `adv < 0`: decrease this completion's log probability.
- `kl_proxy`: keep the policy near the initialization reference.

This is not full industrial GRPO, but it preserves the shape needed for a first
implementation: group sampling, rule reward, relative advantage, and reference
constraint.

It does not retain old-policy log probabilities from sampling and does not compute a clipped importance ratio. `GRPO Mini` is therefore a GRPO-inspired teaching objective, not a term-by-term reproduction of the paper objective.

## Rule Reward

The current reward is arithmetic-oriented:

```python
if pred is not None:
    reward += 0.1
if "answer" in completion.lower() or "final" in completion.lower():
    reward += 0.1
if pred == target:
    return 1.0
```

It has:

- shaping reward for emitting a number or answer-like wording;
- exact reward for the correct final integer.

The formal suite standardizes cold-start responses as:

```text
<think>concise arithmetic trace</think>
<answer>verified integer</answer>
```

The stronger comparison is:

1. pretrained base -> direct GRPO;
2. pretrained base -> structured SFT;
3. structured SFT -> GRPO;
4. compare tagged-answer accuracy, reasoning-format score, and PPL separately.

## Why Not Overclaim

The current implementation is simplified:

- tiny tasks;
- coarse reward;
- small sampling budget;
- minimal reference/KL engineering;
- no old-policy ratio or clipping;
- only a 300-step toy run, not a large-scale or stability result;
- lightweight mini eval.

Reports should say clearly that this is educational GRPO, not a DeepSeek-R1
reproduction.

## What the Next GPU Run Should Fill

The completed [formal report](../experiments/gpu_completion_runs/report.md) now includes:

- Reasoning Answer / Format for base, direct GRPO, SFT-only, and SFT+GRPO;
- PPL / Add / Copy / QA for all four checkpoints;
- `mean_reward` curve;
- whether GRPO hurts PPL;
- sample completion comparisons.
- tracked training/post-training process time and estimated cost, with exclusions stated.

<!-- tinyseek-nav -->

---

Previous: [Stage 6: GRPO Mini](08_stage6_grpo_mini.md) | [Tutorial Index](README.md) | Next: [Repository Roadmap](09_repository_roadmap.md)
