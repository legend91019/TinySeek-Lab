# 04. 当前进度

更新时间：2026-07-07

## 已完成

- 仓库结构已创建，并参考 MiniMind 的分层方式组织为 `model/`、`dataset/`、`trainer/`、`scripts/`、`configs/`、`experiments/`、`docs/`。
- 已添加英文 README 和中文 README。
- 已添加中文教程入口：项目范围、论文地图、阶段路线图、实验报告模板。
- 已添加 Mermaid 图示：训练路线图、模型升级路线图。
- 已实现 Dense / MoE / educational MLA 三种模型路径。
- 已实现 byte-level tokenizer。
- 已实现 JSONL 文本数据集。
- 已实现预训练脚本。
- 已实现生成脚本。
- 已实现 LR / batch size sweep 入口。
- 已添加 SFT 和 GRPO 的阶段占位脚本。
- 已推送初版到 GitHub。

## 部分完成

- DeepSeek 论文锚点已经写入文档，但还需要继续补更细的章节级引用和实验对应表。
- MoE 已有基础 top-k routing 和辅助负载均衡损失，但还缺专家路由统计和可视化。
- MLA 是教学版，能表达 KV latent 压缩思想，但还不是 DeepSeek-V2 的完整实现。

## 尚未完成

- 真实 BPE tokenizer 训练。
- packed dataset / streaming dataset。
- CSV/WandB/SwanLab 日志。
- loss 曲线自动出图。
- MoE routing histogram。
- SFT 数据集和训练脚本实装。
- reasoning cold-start 数据构造。
- DPO。
- rule-based GRPO mini。
- rejection sampling。
- 蒸馏。
- CI / GitHub Actions。
- 真实 torch smoke test，因为当前本地 Python 环境还没装 `torch`。

## 粗略进度

| 模块 | 进度 |
|---|---:|
| 仓库骨架 | 100% |
| 双语入口 | 60% |
| 图示化 | 35% |
| Dense/MoE/MLA 模型骨架 | 70% |
| 预训练最小链路 | 65% |
| DeepSeek LLM sweep 复现入口 | 50% |
| SFT / Cold Start | 15% |
| GRPO / RL | 10% |
| 实验报告体系 | 40% |

整体来看，当前大约处在 **v0.1 原型完成，v0.2 教程化和实验化开始** 的状态。
