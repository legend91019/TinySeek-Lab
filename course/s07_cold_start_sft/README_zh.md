# s07 Cold-Start SFT：先教格式，再做 RL

中文 | [English](README.md) | [课程目录](../README_zh.md)

## 瓶颈

预训练 LM 学会预测合理文本，但不会自动稳定地产生可读的推理与答案格式。若直接用宽松 RL reward，模型可能在学会目标接口前先钻奖励函数的空子。

## 论文线索

DeepSeek-R1-Zero 从 pretrained base 直接进入 RL；完整 R1 pipeline 则在第一轮 reasoning-oriented RL 前加入 cold-start reasoning data。本课程里的 cold start 是一个小型监督阶段，不声称 SFT 本身创造了推理能力。

## 代码路径

阅读 [`trainer/train_sft.py`](../../trainer/train_sft.py) 和 [`docs/zh/19_posttraining_code_walkthrough.md`](../../docs/zh/19_posttraining_code_walkthrough.md)。实现核心是 mask：

```text
prompt tokens      -> labels = -100 -> 不计算监督 loss
response tokens    -> labels = token id -> 计算 next-token loss
padding             -> labels = -100
```

`F.cross_entropy(..., ignore_index=-100)` 因此只训练目标回答区域。架构没有改变；我们用更窄的数据分布继续更新同一个 base model。

## 实验卡片

| 对照 | 候选 | 指标 |
| --- | --- | --- |
| `formal_base20m` | `formal_reasoning_sft` | 留出题答案准确率、格式分、TinyStories sample PPL |

评测必须把格式遵循和答案正确分开。格式正确但答案错误，不是推理成功。

## 证据与决定

在 5 道留出加法题上，答案准确率仍是 `0/5`，格式分从 `0.000 -> 0.600`。窄域 SFT 后，TinyStories sample PPL 从 `1.718 -> 12.670`。

决定：SFT 提供了有用的格式冷启动，但没有算术泛化证据，而且明显移动了 base 分布。只把它晋级为教学 RL 的初始化，不把它称为成功的推理模型。

![后训练评测](../../experiments/gpu_completion_runs/figures/posttraining_reasoning.svg)

## 代码练习

拿一条 prompt/response，手写它的 `input_ids` 和 `labels`。验证第一个 response token 由 prompt 的最后位置预测，并确认 prompt 和 pad token 都不贡献 loss。

## 下一章

模型现在偶尔能遵循目标格式。[s08 GRPO + 评测](../s08_grpo_and_evaluation/README_zh.md) 将验证规则 reward 是改善还是破坏这种行为。

<!-- tinyseek-nav -->

上一篇：[s06 V3 路由 + MTP](../s06_v3_routing_mtp/README_zh.md) | [课程目录](../README_zh.md) | 下一篇：[s08 GRPO + 评测](../s08_grpo_and_evaluation/README_zh.md)
