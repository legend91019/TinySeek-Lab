# TinySeek-Lab 论文优先教程重构设计

## 背景

TinySeek-Lab 的目标是用小规模、可运行的模型学习 DeepSeek 的语言模型论文。仓库已经积累了完整模型代码、训练脚本、架构消融、4090 实测结果和图表，但当前内容同时按课程单元、代码主题和实验批次组织。读者因此需要在 `course/`、`docs/` 和 `experiments/` 之间拼接一条论文叙事。

本次重构把“DeepSeek 的一篇论文”设为教程的基本叙事单位。实验仍然是证据资产，代码仍然是实现资产，但论文文章负责把论文问题、方法、论文原始实验和 TinySeek 补充证据串成一个连续的学习路径。

## 目标

1. 建立五篇核心论文组成的唯一推荐阅读主线：DeepSeek LLM、DeepSeekMoE、DeepSeek-V2、DeepSeek-V3、DeepSeek-R1。
2. 每篇论文提供一篇相对完整的中文教程，并保留英文镜像入口。
3. 优先讲清 DeepSeek 论文自己的问题、方法、实验和结论；TinySeek 现有实验只作为补充性复核、差异说明或教学反例。
4. 将论文原图/表和 TinySeek 已有结果图嵌入对应论文文章，不要求读者先阅读独立实验报告才能理解结论。
5. 不新增训练或消融实验，不改变现有模型代码、训练代码、配置、原始报告和已记录结果。
6. 保留旧文档和历史实验资产，避免破坏已有链接；但从根 README 和主导航中降级为参考/归档入口。

## 非目标

- 不声称复现 DeepSeek 的参数规模、数据规模、训练基础设施或生产级 kernel。
- 不把 TinySeek 小模型上的结果写成 DeepSeek 论文的替代证据。
- 不拼接不同实验分支的局部胜者来构造一个仓库中不存在的“最强模型”。
- 不在本次重构中补跑实验、扩大 token budget、重做评测或改写模型实现。
- 不删除 `course/`、`docs/`、`experiments/` 中已有资料。

## 信息架构

新增唯一推荐入口：

```text
papers/
  README_zh.md
  README.md
  assets/
    deepseek-llm/
    deepseek-moe/
    deepseek-v2/
    deepseek-v3/
    deepseek-r1/
  01-deepseek-llm/
    README_zh.md
    README.md
  02-deepseek-moe/
    README_zh.md
    README.md
  03-deepseek-v2/
    README_zh.md
    README.md
  04-deepseek-v3/
    README_zh.md
    README.md
  05-deepseek-r1/
    README_zh.md
    README.md
```

根目录 `README_zh.md` 和 `README.md` 的“从这里开始/Start here”只指向 `papers/README_zh.md` 或 `papers/README.md`。`course/README*`、`docs/zh/README.md`、`docs/README.md` 和 `experiments/README*` 保留，但在开头增加迁移说明，明确它们是历史课程、参考手册或证据档案，不再与 `papers/` 并列作为主线。

## 五篇论文的边界与现有内容映射

| 论文文章 | 论文主题 | 文章内覆盖的 TinySeek 内容 | 主要现有来源 |
| --- | --- | --- | --- |
| `01-deepseek-llm` | Dense LM、训练配方、规模/效率问题、现代 Transformer 基线 | Dense baseline、LR/batch recipe、MHA/GQA 对照、基础训练接口 | `course/s01_dense_baseline`、`s02_training_recipe`、`s03_gqa`、`docs/zh/12_code_first_dense_lm.md`、`docs/zh/03_stage1_lr_batch_search.md`、`model/stages/stage0_deepseek_llm.py` |
| `02-deepseek-moe` | fine-grained experts、shared experts、专家专门化和负载均衡 | coarse/fine/shared MoE、激活/总参数、expert load CV、aux loss 分支 | `course/s04_deepseek_moe`、`docs/zh/21_from_dense_to_deepseek_moe.md`、`docs/zh/05_stage3_moe.md`、`model/stages/stage1_deepseek_moe.py`、架构报告中的 `moe_*` 配置 |
| `03-deepseek-v2` | DeepSeekMoE 与 MLA、低秩 KV 压缩、解耦 RoPE 路径 | GQA control、参数匹配 low-rank control、low-rank K/V、教学版 MLA、理论 KV 账本 | `course/s05_mla`、`docs/zh/22_from_moe_to_deepseek_v2.md`、`docs/zh/06_stage4_mla.md`、`model/stages/stage2_deepseek_v2.py`、架构报告中的 `v2_*` 配置 |
| `04-deepseek-v3` | auxiliary-loss-free balancing、routing bias、MTP、训练系统与效率 | aux/bias 对照、MTP/no-MTP 对照、分支依赖和证据不足边界 | `course/s06_v3_routing_mtp`、`docs/zh/23_from_v2_to_deepseek_v3.md`、`model/stages/stage3_deepseek_v3.py`、`model/tinyseek.py`、架构报告中的 `moe_bias`/`v3_*` 配置 |
| `05-deepseek-r1` | reasoning post-training、cold-start、rule-based RL/GRPO 及评测边界 | direct GRPO、cold-start SFT、SFT → GRPO、答案/格式/PPL 分离评测 | `course/s07_cold_start_sft`、`course/s08_grpo_and_evaluation`、`docs/zh/19_posttraining_code_walkthrough.md`、`docs/zh/07_stage5_sft_cold_start.md`、`docs/zh/08_stage6_grpo_mini.md`、`trainer/train_sft.py`、`trainer/train_grpo.py` |

论文来源优先使用公开 arXiv/官方版本：

- DeepSeek LLM: *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*，arXiv:2401.02954。
- DeepSeekMoE: *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*，arXiv:2401.06066。
- DeepSeek-V2: *DeepSeek-V2: A Strong Mixture-of-Experts Language Model*，arXiv:2405.04434。
- DeepSeek-V3: *DeepSeek-V3 Technical Report*，arXiv:2412.19437。
- DeepSeek-R1: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*，arXiv:2501.12948。

## 单篇文章模板

每篇 `README_zh.md` 和 `README.md` 使用同一结构。中文版本是主要教学稿，英文版本保持相同章节顺序和结论边界。

### 1. 论文定位

给出论文标题、作者/年份、原始链接、上一代问题、本论文要回答的问题和一段论文结论摘要。摘要只总结论文原始结论，不混入 TinySeek 结果。

### 2. 从上一代到这一代

用一张架构/训练流程图和一个变更表说明：什么保持不变、什么改变、改变试图解决的瓶颈是什么。对 Dense/MoE、GQA/MLA、aux/bias、SFT/GRPO 等概念使用最小必要公式和张量 shape。

### 3. 方法详解

按论文正文的主要方法组织小节，尽量覆盖模型结构、训练目标、数据/训练配方、并行或推理相关设计。将内容分为“必须理解”和“论文完整性补充”两类；后者仍然讲清，但不阻塞第一次阅读。凡论文有而 TinySeek 未实现的内容明确标注。

### 4. 论文实验与图表

按论文实验问题组织，而不是按图号机械罗列。每个实验写清对照、变量、指标、论文观察和论文结论。关键图/表嵌入正文；图注包含原论文图号/表号、页码或章节、来源 URL、许可/引用说明和“读图重点”。

### 5. TinySeek 对应实现

只映射现有代码、配置和数据流。说明教学实现的简化、未实现的生产级部分和与论文实验不可比的地方。提供已经存在的命令和入口，不引入新实验任务。

### 6. TinySeek 补充证据

将已有 SVG、Markdown 报告表格或 JSON 派生图表放在对应论文实验之后。每条结果明确标记为“论文原始证据”或“TinySeek 补充证据”。

### 7. 论文主张与本仓证据对照表

使用统一表头：

```text
论文主张 | 论文证据 | TinySeek 证据 | 关系 | 可迁移边界
```

关系只能使用以下五类措辞：

- `方向一致：补充性复核`
- `方向不一致：不可直接外推`
- `仅论文证据`
- `仅 TinySeek 教学观察`
- `证据不足/不可比较`

### 8. 本章结论与下一篇

先重述论文在其设置下的结论，再说明 TinySeek 是否观察到相似方向、不同方向或没有对应证据。最后给出进入下一篇论文的因果连接。

### 9. 引用、代码和复现入口

列出论文引用、图表来源、代码路径、现有配置、现有报告和已验证命令。复现入口只引用仓库已有命令和资产。

## 证据和措辞规则

文章中所有实验性陈述必须能归入以下一种：

1. **论文原始结论**：以论文的模型规模、数据、指标和实验设置为准。
2. **TinySeek 相似趋势**：只有在比较对象、方向和指标足够接近时使用，并加上“小规模/教学设置”限定。
3. **TinySeek 不一致**：说明差异，不写成对论文结论的反驳。
4. **TinySeek 独有观察**：不提升为 DeepSeek 结论。
5. **不可比较或证据不足**：明确列出缺失的规模、数据、预算、实现或评测条件。

不能用 TinySeek 的失败结果覆盖论文结论，也不能用论文图表暗示 TinySeek 已经完成同等规模复现。

## 论文图表资产规则

- 只下载公开论文版本中实际引用的关键图/表，不把整篇 PDF 放入仓库。
- 每个资产放在对应的 `papers/assets/<paper>/` 目录。
- 文件名使用 `paper-fig-<number>-<short-name>.<ext>` 或 `paper-table-<number>-<short-name>.<ext>`。
- 图表旁边必须写原始来源、图/表编号、页码/章节和 URL；文章末尾保留完整引用。
- TinySeek 图表不复制到 `papers/assets/`，优先相对链接到已有 `experiments/**/figures/`，避免出现两份结果资产。
- 若 PDF 图表需要裁剪，保留原图比例和可读文字，不修改数据、不重绘成看似原始的论文图。

## 旧目录迁移规则

1. 在 `course/README*`、`docs/zh/README.md`、`docs/README.md`、`experiments/README*` 顶部加迁移说明和 `papers/` 链接。
2. 根 README 的主导航改为论文主线；旧的课程和参考手册列入“历史/参考资料”。
3. 不删除或移动现有源文件、实验原始输出、配置和图表；论文文章使用相对链接引用它们。
4. 旧文档内部链接暂不全面重写，先保证入口层不再产生两条并列主线。
5. 若旧课程单元与新文章内容重复，保留它作为“按实验单元阅读”的历史版本，并在开头说明新主线优先。

## 验收标准

- 从根 README 出发，读者只需沿 `papers/README_zh.md` 和五篇文章即可完成主线阅读。
- 每篇文章都同时包含论文结论、论文关键实验/图表、TinySeek 对应实现和现有补充证据。
- 每篇文章都能找到清晰的“论文主张与本仓证据对照表”。
- 所有 TinySeek 数字都链接到已有报告、配置或图表；不新增未经验证的结果。
- 所有论文图表都有来源、图/表编号和原始链接。
- 根 README、论文索引、旧入口迁移说明和中英文文章之间的链接可用。
- 现有 Python 测试和训练代码不需要因为文档重构而改变；文档重构完成后至少运行 Markdown 链接检查和现有 smoke tests。

## 明确不做的实验工作

本次实施不启动 GPU、不修改实验配置、不补跑 MLA/MTP/bias/GRPO、不更换数据集、不重新生成现有报告。若论文讲到的实验在本仓没有对应结果，只写论文结论和“本仓未覆盖”，不为了填表而制造新实验。
