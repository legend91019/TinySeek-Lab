# DeepSeek Phase 1 Tutorial Expansion Design

## Goal
在现有五篇 DeepSeek 教程基础上，新增 V3.2-Exp、V4、DeepSeekMath 三篇双语教程，并把索引分成主架构线与推理训练线。

## Scope
- 新增 papers/06-deepseek-v3-2-sparse-attention：DSA 稀疏注意力与 V3.1 基线对照。
- 新增 papers/07-deepseek-v4：CSA/HCA、mHC、Muon 与百万 token 上下文。
- 新增 papers/08-deepseek-math：数学数据管线、GRPO 与可验证奖励，作为 R1 前置支线。
- 每篇包含中英文 README、论文定位、方法、关键实验/图表、TinySeek 可复用代码或已有证据、claim/evidence 边界、局限和来源。
- 更新 papers/README*.md 与根 README*.md 的路线描述和 Mermaid 图。

## Design decisions
- 保留现有 01–05 编号，新增 06–08，避免破坏现有链接。
- 主架构线：01 DeepSeek LLM → 02 MoE → 03 V2 → 04 V3 → 06 V3.2-Exp → 07 V4。
- 推理训练线：08 DeepSeekMath → 05 R1。R1 仍保留原编号，但不再被描述为架构升级。
- DeepSeek 论文结果是主证据；TinySeek 只作低成本补充，不新增训练或 GPU 实验，不将小模型结果外推到论文规模。
- mHC 和 Muon 不单独成章，作为 V4 内部专题。V3.1-Terminus 作为 V3.2 的基线背景，不单独成章。

## Evidence policy
所有外部数字必须来自官方论文/技术报告或官方仓库；每篇新增文章列出来源台账。无法从现有 TinySeek 代码或结果直接支持的内容标注为“仅论文证���”。

## Non-goals
- 不加入视觉、多模态、OCR、视频、具身、Agent。
- 不新增实验、不改训练器、不声称复现 V3.2/V4 的规模级能力。
