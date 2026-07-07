# 00. 项目范围

TinySeek-Lab 是一个只关注语言模型训练的教程仓库。

## 包含什么

- Decoder-only LM 预训练。
- batch size / learning rate / warmup 等训练 recipe sweep。
- 数据配比实验。
- RMSNorm、RoPE、SwiGLU、GQA。
- DeepSeekMoE 风格 routed experts。
- 教学版 MLA，用来理解 KV cache 压缩。
- SFT、reasoning cold start、DPO、rule-based GRPO mini。
- 后续的 rejection sampling 和 distillation。

## 暂时不包含什么

- 多模态。
- 视觉语言模型。
- 图像/视频生成。
- OCR。
- 具身智能。
- agent/tool-use 主线。

原因很简单：第一版要把语言模型训练路线讲透。如果一开始把 VL、OCR、agent 都揉进来，读者会失去主线。

## 项目产物

第一阶段希望产出：

- 一个可训练的 30M-150M Dense baseline。
- 一个 total params 更大、activated params 更小的 MoE 版本。
- LR / batch size sweep 的可复现实验曲线。
- 每个实验的报告模板。
- 一个小型 reasoning post-training demo。

## 非目标

本项目不宣称复现 DeepSeek 的模型质量。我们复现的是 DeepSeek 论文中的研究问题、实验习惯和架构演进。
