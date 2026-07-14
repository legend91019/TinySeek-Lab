# TinySeek-Lab DeepSeek 架构演进教学改版设计

## 目标

把 TinySeek-Lab 从“组件说明 + 可运行实验仓库”升级为一条初学者可以真正跟写的代码课程：读者先写出 DeepSeek LLM 风格的 Dense 语言模型，再沿论文时间线逐步改造成 DeepSeekMoE、DeepSeek-V2 和 DeepSeek-V3，最后进入 DeepSeek-R1 的 SFT 与 GRPO 训练路线。

本轮只覆盖语言模型。多模态、视觉、视频、OCR、具身和 Agent 不进入主线。

## 目标读者与完成标准

目标读者会 Python 和基础 PyTorch，但没有从零实现过完整 decoder-only LM。

完成课程后，读者应该能够：

1. 从 `input_ids` 开始写出完整 causal LM，而不只是复制若干组件。
2. 解释 DeepSeek LLM、DeepSeekMoE、DeepSeek-V2、DeepSeek-V3 的主要结构差异。
3. 指出每次升级修改了哪个子层、解决什么问题、引入什么代价。
4. 运行 CPU shape/forward 测试，并在 GPU 上完成公平对照实验。
5. 区分论文原始结论、TinySeek 已有实测结果和仍待运行的实验假设。
6. 理解 DeepSeek-R1 不是“模型骨架升级”，而是 base model 之上的训练流程升级。

## 教学主线

主线严格按研究演进组织：

```text
DeepSeek LLM
Dense decoder-only LM
Pre-Norm + RMSNorm + RoPE + SwiGLU；大模型使用 GQA
        |
        v
DeepSeekMoE
Dense FFN -> fine-grained routed experts + shared experts
        |
        v
DeepSeek-V2
DeepSeekMoE + Multi-head Latent Attention
        |
        v
DeepSeek-V3
V2 backbone + auxiliary-loss-free routing balance + MTP objective
        |
        v
DeepSeek-R1
同类 base backbone + cold-start SFT + GRPO + rejection sampling 路线
```

不增加一个虚构的“旧式 Transformer -> modern block”DeepSeek 阶段。DeepSeek LLM 的第一代公开结构本身已经采用 RMSNorm、RoPE 和 SwiGLU。

## 双轨代码结构

### 教学轨

新增 `model/stages/`，每个文件都是可以独立实例化和执行的完整语言模型，而不是只能阅读的伪代码：

- `stage0_deepseek_llm.py`：Dense DeepSeek LLM 风格基线。
- `stage1_deepseek_moe.py`：在 Stage 0 上只替换 FFN，加入细粒度 routed experts 和 shared experts。
- `stage2_deepseek_v2.py`：在 Stage 1 上替换 attention，加入教学版 MLA 和解耦 RoPE 路径。
- `stage3_deepseek_v3.py`：在 Stage 2 上加入无辅助损失路由偏置与 MTP 训练目标。

每个阶段保留统一接口：

```python
model = StageModel(config)
out = model(input_ids, labels=labels)
loss = out["loss"]
logits = out["logits"]
```

Stage 1-3 还可以返回 `aux_loss`、`mtp_loss`、`router_stats` 等阶段特有信息，但不得破坏基本训练接口。

### 实验轨

`model/tinyseek.py` 继续作为统一实验模型。教学章节先讲相邻阶段的完整代码和 diff，再说明这些能力如何通过配置汇入统一模型。正式 sweep、成本记录和 GPU 报告仍使用统一模型，避免维护两套训练系统。

## 各阶段的实现边界

### Stage 0: DeepSeek LLM

保留完整端到端结构：配置、Embedding、RMSNorm、RoPE、MHA/GQA、SwiGLU、Transformer Block、LM Head、权重绑定和右移交叉熵。教程按张量形状贯穿整个 `forward`。

### Stage 1: DeepSeekMoE

先展示 Dense FFN，再替换为 MoE FFN。教学实现必须体现：

- 每个 token 的 router logits 和 top-k 选择。
- routed expert 与 shared expert 的不同职责。
- 细粒度专家相对普通 MoE 的含义：把一个大 FFN 的中间维拆成更小专家，并增加可组合专家数量。
- 总参数与每 token 激活参数的区别。
- auxiliary load-balance loss、expert load 和 routing collapse。

实现保持单卡、原生 PyTorch、易读优先，不实现分布式 expert parallel 或 all-to-all 通信。

### Stage 2: DeepSeek-V2

教学版 MLA 必须明确展示：

- hidden state 先压缩成低秩 KV latent。
- latent 如何重建 content K/V。
- RoPE 位置分量为什么需要与压缩 content path 分开理解。
- 训练 forward 与真正 cached decoding 的区别。
- 每 token 理论缓存元素数量如何估算。

仓库不得把教学版 MLA 宣称为生产级 DeepSeek-V2 复刻。没有真正缓存 latent 的生成路径时，报告只能写“结构与理论缓存估算已实现”。

### Stage 3: DeepSeek-V3

实现两个适合单卡教学的核心升级：

1. Auxiliary-loss-free balance：路由选择使用可更新的 expert bias；bias 只影响选择，不直接改变 affinity 权重。训练器在优化步后根据 expert load 更新 bias。保留可选的小型 sequence-wise balance loss，用于对照论文中极小系数的 complementary loss。
2. Multi-Token Prediction：主 next-token loss 之外增加至少一个未来 token 的预测模块，复用 token embedding 与输出 head，并返回独立的 `mtp_loss`。

不在小仓库复刻 FP8、DualPipe、跨节点通信和硬件内核。这些属于工程扩展阅读，不是单卡模型结构主线。

### DeepSeek-R1

R1 放在训练演进部分：base pretraining -> cold-start SFT -> reasoning RL -> rejection sampling / second SFT 的论文路线。当前仓库保留教学版 GRPO，并明确它与完整 DeepSeek-R1 pipeline 的差距。

## 章节结构

新增一组中英双语代码演进章节：

- 架构演进总览：时间线、每代解决的问题、代码文件和实验入口。
- 从 DeepSeek LLM 到 DeepSeekMoE：逐段代码、shape、路由示例和相邻 diff。
- 从 DeepSeekMoE 到 DeepSeek-V2：MLA 动机、压缩路径、缓存计算和相邻 diff。
- 从 DeepSeek-V2 到 DeepSeek-V3：路由 bias、MTP objective、训练器改动和相邻 diff。

已有 Stage 2-6 短章节改造成实验课：引用详细代码章，补齐假设、控制变量、命令、结果读取方法和失败分析。过时状态必须更新，例如不得继续声称 SFT trainer 是占位实现。

每章固定回答六个问题：

1. 上一代哪里不够？
2. 论文做了什么？
3. TinySeek 为教学保留了什么、简化了什么？
4. 相比上一阶段改了哪些代码？
5. 如何验证实现没有写错？
6. 实验结果支持什么、不支持什么？

## 实验设计

每个阶段固定提供四层验证：

1. CPU shape/forward 测试：输出 shape、finite loss、参数量、阶段特有统计。
2. 公平配置：相同数据、token budget、序列长度和优化器，仅改变目标结构变量。
3. GPU 报告：validation loss/PPL、tokens/s、峰值显存、GPU 小时和人民币成本。
4. 解释边界：论文结论、TinySeek 结果和待验证假设分栏。

新增架构实验配置与汇总脚本，至少覆盖：

- MHA vs GQA。
- Dense FFN vs DeepSeekMoE。
- auxiliary loss routing vs bias-balanced routing。
- MTP off vs MTP on。
- 普通 attention vs educational MLA 的理论 KV cache 对照。

未上卡前，结果报告使用明确的 `待上卡 / Pending GPU run`，不使用虚构数字。已有 RTX 4090 v1 结果继续保留，并标记其代码版本和局限。

## 首页与视觉呈现

README 首屏先回答：这个仓库训练什么、学习路线是什么、已有真实结果是什么、需要多少钱。使用现有 SVG 实验图和 Mermaid 架构图，不添加与教学无关的装饰。

README 增加：

- “四代模型，一条代码主线”的课程表。
- 阶段文件、论文、实验和预期学习成果的映射表。
- 真实 4090 结果图，而不只给报告链接。
- 初学者、研究复现、GPU 实验三种入口。

中英文 README 和核心章节保持一一对应。中文可以解释得更展开，但不得出现英文版缺少整个主题的情况。

## 测试与质量门槛

- 所有新增 Python 文件通过 AST 解析。
- 所有 JSON 配置可解析。
- 安装 PyTorch 的环境运行阶段模型 smoke tests。
- 测试 Stage 0-3 的 logits shape 与 finite loss。
- 测试 MoE expert counts 总和与 top-k 一致。
- 测试 router bias 在负载不均时朝正确方向更新。
- 测试 MTP 开关改变 loss 字段但不改变主 logits shape。
- 检查中英章节配对、内部链接和上一章/下一章导航。
- README 中不得把待运行实验写成已证实结论。

## 非目标

- 不追求 DeepSeek 参数规模或最终能力复现。
- 不实现分布式 MoE 通信、FP8 kernel、DualPipe 或生产推理引擎。
- 不进入 V3.2/V4 的稀疏长上下文代码实现；它们只作为后续阅读方向。
- 不扩展视觉、视频、OCR、具身或 Agent 主线。
