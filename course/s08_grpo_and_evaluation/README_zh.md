# s08 GRPO 与评测：让证据决定故事在哪里停下

中文 | [English](README.md) | [课程目录](../README_zh.md)

## 研究问题

对同一个 prompt 采样多个 completion 后，规则 reward 能否让 SFT 初始化的模型更正确，同时不破坏格式行为和 base 分布？

## 论文线索与范围

DeepSeek-R1 在更大的 pipeline 中使用 GRPO-style reinforcement learning。TinySeek 实现的是教学版 group-relative update：采样一组 completion、计算 reward、归一化成 advantage，再优化采样 token 的 log probability。

它**不是** R1 规模的忠实复现：缺少完整目标、基础设施、数据规模、reward 覆盖和多阶段 pipeline。

## 代码路径

阅读 [`trainer/train_grpo.py`](../../trainer/train_grpo.py) 和 [`docs/zh/19_posttraining_code_walkthrough.md`](../../docs/zh/19_posttraining_code_walkthrough.md)：

```text
prompt -> G 个 completions -> 规则奖励 r_i
-> A_i = (r_i - mean(r)) / (std(r) + eps)
-> gather 采样 token 的 log probability
-> loss = -mean(A_i * log pi(completion_i))
```

代码必须只 gather 真正采样的 completion token，detach advantage，并处理组内 reward 方差接近 0 的情况。

## 实验卡片

比较两条因果路径，而不只是两个最终 checkpoint：

1. `base -> direct GRPO`
2. `base -> cold-start SFT -> GRPO`

同时报告答案准确率、格式分、mean reward 和共同分布 PPL。reward score 必须与留出评测分开；优化 reward 不等于任务已经解决。

## 证据

| Checkpoint | 答案 | 格式 | TinyStories sample PPL |
| --- | ---: | ---: | ---: |
| base | `0/5` | `0.000` | `1.718` |
| direct GRPO | `0/5` | `0.000` | `1.995` |
| cold-start SFT | `0/5` | `0.600` | `12.670` |
| SFT + GRPO | `0/5` | `0.200` | `12.306` |

## 决定

当前宽松 reward 没有带来算术泛化，而且降低了 SFT 的格式分。GRPO 保留为教学机制 demo，不晋级为推理结果。因此课程诚实地停在一个失败的本地 RL 假设上，并给出下一轮研究问题：先改善留出评测、reward 设计和目标忠实度，再扩大训练。

![后训练证据](../../experiments/gpu_completion_runs/figures/posttraining_reasoning.svg)

## 复现与审计

阅读归档报告和原始输出：

- [`experiments/gpu_completion_runs/report_zh.md`](../../experiments/gpu_completion_runs/report_zh.md)
- [`experiments/gpu_completion_runs/eval_formal_reasoning_grpo.json`](../../experiments/gpu_completion_runs/eval_formal_reasoning_grpo.json)
- [`docs/zh/08_stage6_grpo_mini.md`](../../docs/zh/08_stage6_grpo_mini.md)

## 你最终应该学会什么

你现在可以沿着一个稳定 LM 接口，追踪 Dense 训练、配方搜索、GQA、MoE、MLA、V3-style 目标、SFT 和教学版 GRPO。更重要的是，你应该能区分：论文动机、代码实现、实验结果和工程决定。它们彼此相关，但不能互相冒充。

<!-- tinyseek-nav -->

上一篇：[s07 Cold-start SFT](../s07_cold_start_sft/README_zh.md) | [课程目录](../README_zh.md) | 下一篇：[实验报告中心](../../experiments/README_zh.md)
