# TinySeek 课程主线：用实验重走 DeepSeek LM

中文 | [English](README.md)

这里是 TinySeek-Lab 的**唯一推荐阅读入口**。课程不是把 Transformer 组件逐个列出来，而是沿着研究决策循环前进：

```text
观察瓶颈 -> 找论文线索 -> 写最小代码改动
-> 预注册控制变量实验 -> 读证据 -> 升级或否决
```

这是一条依据公开论文重建的、可复现的研究路径，不声称还原 DeepSeek 内部未公开的真实时间线。论文规模证据和 TinySeek 小模型实测会始终分开标注。

## 主线

| 单元 | 上一代的问题 | 代码改变 | 决定下一步的实验 |
| --- | --- | --- | --- |
| [s01 Dense 基线](s01_dense_baseline/README_zh.md) | 能否写出并训练完整 decoder LM？ | RMSNorm、RoPE、GQA、SwiGLU、残差块和 LM loss | 结构 smoke test 与 Dense 基线 |
| [s02 训练配方](s02_training_recipe/README_zh.md) | 架构比较前，LR 和 batch size 怎么选？ | 不改架构，先让训练循环可测量 | DeepSeek LLM 风格 LR/batch 网格搜索 |
| [s03 GQA](s03_gqa/README_zh.md) | 能否减少 KV 状态而不损害语言建模？ | MHA -> 分组 K/V head | 3-seed MHA 与 GQA 对照 |
| [s04 DeepSeekMoE](s04_deepseek_moe/README_zh.md) | 能否增大容量但不激活全部参数？ | Dense FFN -> routed/shared experts | coarse/fine/shared 与负载均衡消融 |
| [s05 MLA](s05_mla/README_zh.md) | KV cache 能否比 GQA 更小？ | 低秩 KV 路径与解耦 RoPE 路径 | control、低秩和教学版 MLA 对照 |
| [s06 V3 路由 + MTP](s06_v3_routing_mtp/README_zh.md) | 能否不用辅助损失完成均衡？额外目标有用吗？ | routing bias 与 multi-token prediction | aux/bias 和 MTP 开关实验 |
| [s07 Cold-start SFT](s07_cold_start_sft/README_zh.md) | 基座模型如何学会可读的推理格式？ | masked supervised fine-tuning | base 与 cold-start SFT 的格式/答案/PPL 评测 |
| [s08 GRPO + 评测](s08_grpo_and_evaluation/README_zh.md) | 规则 RL 会改善 SFT 学到的行为吗？ | group sampling、reward、归一化 advantage | direct RL 与 SFT -> GRPO 对照，并写清边界 |

## 每个单元都回答五个问题

1. **瓶颈是什么？** 必须来自上一单元的指标或实现限制。
2. **论文贡献是什么？** 公开的动机是什么，TinySeek 保留和简化了什么？
3. **最小代码差异是什么？** 精确到模型文件、配置和张量接口。
4. **什么算赢？** 事先写好决策门槛，同时看质量、速度、显存和不确定性。
5. **最后怎么决定？** 升级、保留分支、否决，还是标记为证据不足。

失败结果不是噪音。精品研究教程应该教你：一个实验如何阻止一个看起来很漂亮的结构被贸然推广。

## 证据地图

- 架构多 seed 证据：[`experiments/architecture_lab_runs/report_zh.md`](../experiments/architecture_lab_runs/report_zh.md) 和[英文报告](../experiments/architecture_lab_runs/report.md)。
- 4090 正式训练与后训练：[`experiments/gpu_completion_runs/report_zh.md`](../experiments/gpu_completion_runs/report_zh.md) 和[英文报告](../experiments/gpu_completion_runs/report.md)。
- 公式 -> 张量 shape -> PyTorch API：[`docs/zh/24_math_to_pytorch.md`](../docs/zh/24_math_to_pytorch.md)。
- 完整代码走读：[`docs/zh/15_code_walkthrough.md`](../docs/zh/15_code_walkthrough.md)。
- 原有章节仍保留在 [`docs/zh/README.md`](../docs/zh/README.md)，作为参考手册。

## 复现契约

课程沿用报告中记录的数据 hash、配置、seed、token budget 和成本账本。先跑最小路径：

```bash
pip install -r requirements.txt
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 20
python scripts/inspect_stage_models.py
```

正式 GPU 命令和环境细节放在实验报告中，正文因此可以保持清楚。

## 范围

只做 LM：不涉及视觉、视频、OCR、多模态、具身或 agent。本仓库是笔者学习 DeepSeek 论文时，为方便理解而完成的作品。
