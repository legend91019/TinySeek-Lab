> 本仓库是笔者学习 DeepSeek 论文时，为方便理解而完成的作品。

<div align="center">

# TinySeek-Lab

**用几百 M 以内的小语言模型，重走 DeepSeek 的 LM 研究路线**

中文 | [English](README.md)

</div>

TinySeek-Lab 是一套从代码、训练到实验报告的双语教程。你不会只调用现成模型，而是先写出完整 Dense LM，再沿论文路线逐代改造成 DeepSeekMoE、DeepSeek-V2 和 DeepSeek-V3，最后进入 R1 风格的 SFT 与 GRPO 教学实验。

本仓只做语言模型，不进入多模态、视觉、视频、OCR、具身和 Agent 主线。目标是复现研究问题与实验方法，不是复现 DeepSeek 的参数规模或最终能力。

## 实验驱动，而不是按版本堆组件

每次结构升级都遵循同一个闭环：

```text
上一代基线 -> 可测量瓶颈 -> 研究假设 -> 单变量消融
-> 预先写下的决策门槛 -> 升级 / 保留上一代
```

DeepSeek 论文提供问题、方法和论文规模的证据；TinySeek 提供小模型代码与可运行实验。GPU 数据尚未得到时只写“待验证”，不会把研究假设写成仓库已经证明的结论。完整方法见[四代架构演进总览](docs/zh/20_architecture_evolution_overview.md)，可执行矩阵见[架构演进公平实验](experiments/06_architecture_evolution_plan_zh.md)。

## 四代模型，一条代码主线

| 代际 | 你会写出的完整模型 | 核心变化 | 深入代码课 |
| --- | --- | --- | --- |
| DeepSeek LLM | [`stage0_deepseek_llm.py`](model/stages/stage0_deepseek_llm.py) | Dense、RMSNorm、RoPE、SwiGLU、GQA | [从零写完整 LM](docs/zh/12_code_first_dense_lm.md) |
| DeepSeekMoE | [`stage1_deepseek_moe.py`](model/stages/stage1_deepseek_moe.py) | 细粒度 routed experts + shared experts | [Dense -> MoE](docs/zh/21_from_dense_to_deepseek_moe.md) |
| DeepSeek-V2 | [`stage2_deepseek_v2.py`](model/stages/stage2_deepseek_v2.py) | MoE + 教学版 MLA | [MoE -> V2](docs/zh/22_from_moe_to_deepseek_v2.md) |
| DeepSeek-V3 | [`stage3_deepseek_v3.py`](model/stages/stage3_deepseek_v3.py) | 无辅助损失路由 bias + MTP | [V2 -> V3](docs/zh/23_from_v2_to_deepseek_v3.md) |

先读[架构演进总览](docs/zh/20_architecture_evolution_overview.md)，再按表格从上到下跟写。正式训练使用统一模型 [`model/tinyseek.py`](model/tinyseek.py)，阶段文件负责教学，统一模型负责公平实验。

## 当前成果：正式 4090 实验已完成

TinySeek-Lab 已在一张 RTX 4090 上跑完整套训练与消融：

```text
TinyStories -> tiny base -> dense 35M/115M -> LR/batch sweep
-> MoE -> MLA -> SFT -> GRPO mini -> mini eval -> 成本和图表
```

- [48 次、16 配置、3 seed 架构报告](experiments/architecture_lab_runs/report_zh.md)：`1.5679 GPU h`，约 `3.4180 元`。
- [11 组正式训练与后训练报告](experiments/gpu_completion_runs/report_zh.md)：`0.8985 GPU h`，约 `1.9588 元`。
- 本轮账本合计记录 `2.4664 GPU h` 的训练/后训练进程，对应约 `5.3768 元`；不含数据准备、独立评测、报告生成和租卡空闲时间。
- GQA 在理论 KV/token 从 `384` 降到 `192` 的同时，PPL 没有退化，本预算下通过升级门槛。
- shared expert 路线 PPL 优于 coarse MoE，但吞吐约低 35%，因此保留“质量分支”和“速度分支”，不写成单一赢家。
- 教学版 MLA 把理论 KV/token 从 `192` 降到 `72`，但 PPL 明显变差；bias routing 也没有击败 aux=0.01；两者当前都不升级。
- 在 5 道留出加法题上，SFT 把推理格式分从 `0.0` 提到 `0.6`，但仍是 `0/5`；后续 GRPO 又把格式分降到 `0.2`。这组 mini-eval 没有提供算术泛化证据，宽松奖励还会带来退化。

这些是 TinySeek 小模型上的实测结论，不外推到 DeepSeek 原始规模。

![TinySeek 3-seed 架构 PPL](experiments/architecture_lab_runs/figures/architecture_ppl.svg)

## 三档快速路径

| 路径 | 适合谁 | 命令入口 |
| --- | --- | --- |
| CPU 代码课 | 想先看懂四代模型和 shape | `python scripts/inspect_stage_models.py` |
| 小 GPU 教学 run | 想体验 tiny dense -> SFT -> GRPO | [上卡前最终 Checklist](docs/zh/18_gpu_fill_only_checklist.md) |
| RTX 4090 研究 run | 想复现完整训练与多 seed 架构对照 | [实验报告中心](experiments/README_zh.md) |

快速代码路线：先读[架构演进总览](docs/zh/20_architecture_evolution_overview.md)，跟写四个 stage 文件，再读[训练主循环](docs/zh/16_training_loop_from_config_to_checkpoint.md)，最后按[公平架构实验计划](experiments/06_architecture_evolution_plan_zh.md)上卡。下方完整阅读路线则先补项目范围和论文地图。

## 一图看懂路线

```mermaid
flowchart LR
  A["DeepSeek LLM<br/>Dense"] --> B["DeepSeekMoE<br/>稀疏 FFN"]
  B --> C["DeepSeek-V2<br/>MLA"]
  C --> D["DeepSeek-V3<br/>Bias + MTP"]
  D --> E["DeepSeek-R1<br/>SFT + GRPO"]
```

## 模型升级路线

```mermaid
flowchart TB
  subgraph Dense["Dense 基线"]
    T0["文本 tokens"] --> E0["Embedding"]
    E0 --> B0["Transformer blocks<br/>MHA/GQA + Dense SwiGLU"]
    B0 --> L0["LM head"]
  end

  subgraph MoE["MoE 升级"]
    R["Router"] --> X1["Expert 1"]
    R --> X2["Expert 2"]
    R --> XN["Expert N"]
    X1 --> M["加权求和"]
    X2 --> M
    XN --> M
  end

  subgraph MLA["教学版 MLA"]
    H["Hidden states"] --> Z["低秩 latent KV"]
    Z --> K["重建 K"]
    Z --> V["重建 V"]
  end
```

## 仓库结构

```text
TinySeek-Lab/
  configs/              小模型和实验配置
  dataset/              数据集封装和 byte tokenizer
  docs/                 英文教程章节
  docs/zh/              中文教程章节
  experiments/          sweep 计划和实验模板
  model/stages/         四代完整教学模型
  model/tinyseek.py     正式实验统一模型
  scripts/              数据准备和生成脚本
  trainer/              预训练、SFT、sweep、GRPO 入口
  tests/                smoke tests
```

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

创建 toy 数据：

```bash
python scripts/prepare_toy_data.py --out data/toy_pretrain.jsonl
```

跑一个最小预训练：

```bash
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --max_steps 20
```

从 checkpoint 生成文本：

```bash
python scripts/generate.py --config configs/tiny_dense.json --ckpt out/tiny_dense_last.pt --prompt "DeepSeek is"
```

跑 DeepSeek LLM 启发的 LR / batch size 网格搜索：

```bash
python trainer/sweep_pretrain.py --sweep experiments/01_lr_batch_grid.json
```

训练时记录 AutoDL GPU 成本：

```bash
# RTX 4090：2.18 元/小时
python trainer/train_pretrain.py --config configs/tiny_dense.json --data data/toy_pretrain.jsonl --hourly_rate 2.18

# 汇总所有实验账本
python scripts/summarize_costs.py --input_dir out
```

跑后训练 toy 主线：

```bash
python scripts/prepare_toy_sft_data.py --out data/toy_sft.jsonl
python trainer/train_sft.py --config configs/tiny_sft.json --data data/toy_sft.jsonl --init_ckpt out/tiny_dense_last.pt --hourly_rate 2.18

python scripts/prepare_toy_grpo_data.py --out data/toy_grpo.jsonl
python trainer/train_grpo.py --config configs/tiny_grpo.json --data data/toy_grpo.jsonl --init_ckpt out/tiny_sft_last.pt --hourly_rate 2.18
```

第一次 AutoDL RTX 4090 实测报告见：
[experiments/02_autodl_4090_smoke_report_zh.md](experiments/02_autodl_4090_smoke_report_zh.md)。
v1 预训练 -> SFT -> GRPO 链路实测报告见：
[experiments/03_v1_pipeline_smoke_report_zh.md](experiments/03_v1_pipeline_smoke_report_zh.md)。
最新 3-seed 架构实验见：
[experiments/architecture_lab_runs/report_zh.md](experiments/architecture_lab_runs/report_zh.md)。
正式训练、sweep 与后训练报告见：
[experiments/gpu_completion_runs/report_zh.md](experiments/gpu_completion_runs/report_zh.md)。
早期 v1 结果仍保留在 [experiments/05_4090_v1_results_zh.md](experiments/05_4090_v1_results_zh.md)，用于展示仓库如何从 smoke 逐步升级为正式实验。
代码阅读路线见：[docs/zh/15_code_walkthrough.md](docs/zh/15_code_walkthrough.md)。
本轮正式套件执行前写下的实验计划归档在：
[experiments/04_formal_experiment_plan_zh.md](experiments/04_formal_experiment_plan_zh.md)。

## 中文阅读顺序

完整目录见：[docs/zh/README.md](docs/zh/README.md)

1. [项目范围](docs/zh/00_project_scope.md)
2. [DeepSeek 语言模型论文地图](docs/zh/01_deepseek_lm_paper_map.md)
3. [四代架构演进总览](docs/zh/20_architecture_evolution_overview.md)
4. [从零写 DeepSeek LLM Dense 基线](docs/zh/12_code_first_dense_lm.md)
5. [从 Dense 改到 DeepSeekMoE](docs/zh/21_from_dense_to_deepseek_moe.md)
6. [从 MoE 改到 DeepSeek-V2](docs/zh/22_from_moe_to_deepseek_v2.md)
7. [从 V2 改到 DeepSeek-V3](docs/zh/23_from_v2_to_deepseek_v3.md)
8. [训练主循环：从 Config 到 Checkpoint](docs/zh/16_training_loop_from_config_to_checkpoint.md)
9. [SFT 和 Reasoning Cold Start](docs/zh/07_stage5_sft_cold_start.md)
10. [Rule-Based GRPO Mini](docs/zh/08_stage6_grpo_mini.md)

补充文档：

- [总训练路线图](docs/zh/02_training_roadmap.md)
- [当前进度](docs/zh/04_current_progress.md)

英文原始章节在 [docs/](docs)。

## 当前状态

当前版本已经包含：

- DeepSeek LLM / DeepSeekMoE / V2 / V3 四份完整教学模型。
- 统一模型中的细粒度 expert、aux/bias 路由和 MTP 开关。
- byte-level tokenizer 和 JSONL 文本数据集。
- 预训练脚本。
- 生成脚本。
- LR / batch size sweep 入口。
- SFT、rule-based GRPO mini 和 mini eval。
- GPU 成本、显存、token、粗略 FLOPs 记录。
- RTX 4090 完整编排脚本、48 次多 seed 架构实测、11 组正式训练/后训练、原始结果和自动图表。
- DeepSeek LM 路线相关双语教程文档。
- 16 份架构实验配置，覆盖 MoE 演进、aux 权重、路由 bias、低秩 KV、MLA 与 MTP，并已完成 3 seed 实测与决策回填。

GRPO 当前仍是教学版，用来讲清算法形状；它不是严肃 RL 性能复现。
