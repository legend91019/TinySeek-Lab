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

The core idea: sample several completions for the same prompt, compare them
inside the group, and increase the probability of above-average completions.

## Policy Loss

The teaching loss is:

```python
policy_logp = completion_logprob(policy, ids, prompt_len)
ref_logp = completion_logprob(ref, ids, prompt_len)
kl_proxy = (policy_logp - ref_logp).pow(2)
loss = (-adv * policy_logp) + kl_beta * kl_proxy
```

Interpretation:

- `adv > 0`: increase this completion's log probability.
- `adv < 0`: decrease this completion's log probability.
- `kl_proxy`: keep the policy near the initialization reference.

This is not full industrial GRPO, but it preserves the shape needed for a first
implementation: group sampling, rule reward, relative advantage, and reference
constraint.

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

This is enough to teach the algorithm shape. The stronger next step is:

1. run better arithmetic/format cold-start SFT;
2. run GRPO;
3. compare SFT-only with SFT+GRPO on arithmetic pass rate.

## Why Not Overclaim

The current implementation is simplified:

- tiny tasks;
- coarse reward;
- small sampling budget;
- minimal reference/KL engineering;
- no long training result yet;
- lightweight mini eval.

Reports should say clearly that this is educational GRPO, not a DeepSeek-R1
reproduction.

## What the Next GPU Run Should Fill

After the next post-training run, the report should add:

- SFT-only Add / Copy / QA / Format;
- post-GRPO Add / Copy / QA / Format;
- `mean_reward` curve;
- whether GRPO hurts PPL;
- sample completion comparisons;
- total GPU time and cost.

<!-- tinyseek-nav -->

---

Previous: [Stage 6: GRPO Mini](08_stage6_grpo_mini.md) | [Tutorial Index](README.md) | Next: [Repository Roadmap](09_repository_roadmap.md)
