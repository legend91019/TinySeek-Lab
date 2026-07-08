# 01. DeepSeek 语言模型论文地图

本项目只使用 DeepSeek 论文中和语言模型训练直接相关的部分。

## 核心阅读路径

1. **DeepSeek LLM**
   - 对应主题：训练 recipe、scaling law、batch size / learning rate 搜索。
   - TinySeek 对应章节：LR / batch size 小规模网格搜索。

2. **DeepSeekMoE**
   - 对应主题：专家分化、路由、负载均衡、routing collapse。
   - TinySeek 对应章节：Dense FFN -> routed MoE FFN。

3. **DeepSeek-V2**
   - 对应主题：DeepSeekMoE + MLA，经济训练和高效推理。
   - TinySeek 对应章节：教学版 MLA 和 KV-cache 估算。

4. **DeepSeek-V3**
   - 对应主题：MoE + MLA 大规模验证、auxiliary-loss-free balance、multi-token prediction。
   - TinySeek 对应章节：MoE 负载均衡和高级消融。

5. **DeepSeek-R1**
   - 对应主题：R1-Zero 直接 RL、R1 的 cold-start SFT + RL + rejection sampling。
   - TinySeek 对应章节：reasoning cold start vs direct rule RL。

6. **DeepSeek-V3.2 / V4**
   - 对应主题：长上下文稀疏注意力、更大 post-training compute、MoE 架构继续演化。
   - TinySeek 对应章节：后续 sparse attention 和 post-training scaling 笔记。

## 可选语言模型论文

- DeepSeekMath / DeepSeek-Prover：数学推理与可验证数据。
- ESFT：专家特化微调。
- Native Sparse Attention：长上下文高效注意力。
- Inference-Time Scaling for Generalist Reward Modeling：奖励模型和 inference-time compute。
- Engram：conditional memory 作为另一种稀疏轴。

## 明确跳过

- DreamCraft3D。
- DeepSeek-VL / VL2。
- Janus / JanusFlow / Janus-Pro。
- DeepSeek-OCR / OCR-2。

这些论文很有价值，但不属于第一版 LM-only 教程主线。

<!-- tinyseek-nav -->

---

上一篇: [项目范围](00_project_scope.md) | [教程目录](README.md) | 下一篇: [代码优先 Dense LM](12_code_first_dense_lm.md)
