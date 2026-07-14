# 08. 阶段 6：Rule-Based GRPO Mini

目标：把 DeepSeek-R1 风格的 rule-based RL 缩小到可教学、可运行的版本。

## DeepSeek 对应关系

DeepSeek-R1 使用 GRPO，并大量利用可验证任务的规则奖励。最典型的例子是数学题、代码题：答案可以自动判断，不必每一步都靠人工标注。

## Tiny Rewards

第一版可以从这些奖励开始：

- 算术答案是否匹配。
- 最终答案格式是否正确。
- 简单 Python 单测是否通过。
- 输出长度是否合理。
- 是否出现明显重复。

## 最小 GRPO 流程

对每个 prompt：

1. 采样 `G` 个 completion。
2. 用规则奖励给每个 completion 打分。
3. 在同一组里归一化 reward，得到 advantage。
4. 使用 policy ratio、clip 和 KL 项更新模型。

## 建议实验

1. 从 base model 直接 RL。
2. 从 cold-start SFT model 开始 RL。
3. correctness-only reward vs correctness + format reward。
4. group size 4、8、16。

## 关键提醒

RL 不是魔法。小模型上尤其容易出现 reward hacking，所以每个 reward 都要配失败案例分析。

## 代码已经怎样接通

- [`JsonlPromptDataset`](../../dataset/lm_dataset.py) 读取 prompt 和可验证 answer。
- [`sample_group`](../../trainer/train_grpo.py) 对同一 prompt 采样一组 completion。
- `rule_reward` 给最终整数正确性和格式 shaping 打分。
- 组内 reward 标准化后得到 advantage，再结合 reference KL proxy 更新 policy。

```bash
python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --hourly_rate 2.18
```

当前 v1 得到非零 reward，但加法 exact match 仍为 0。正确结论是“教学版 GRPO 形状跑通”，不是“RL 已让小模型学会推理”。完整目标与简化项见[后训练代码细读](19_posttraining_code_walkthrough.md)。

<!-- tinyseek-nav -->

---

上一篇: [阶段 5：SFT](07_stage5_sft_cold_start.md) | [教程目录](README.md) | 下一篇: [仓库路线图](09_repository_roadmap.md)
