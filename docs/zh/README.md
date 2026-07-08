# 教程目录

中文 | [English](../README.md)

TinySeek-Lab 的英文教程放在 `docs/`，中文教程放在 `docs/zh/`。两套文档走同一条训练路线，但中文版本会写得更像讲义，解释会更展开一些。

## 主线阅读顺序

1. [项目范围](00_project_scope.md)
2. [DeepSeek 语言模型论文地图](01_deepseek_lm_paper_map.md)
3. [代码优先：从零写出最初的 DeepSeek-style Dense LM](12_code_first_dense_lm.md)
4. [训练主循环：从 Config 到 Checkpoint](16_training_loop_from_config_to_checkpoint.md)
5. [代码导读](15_code_walkthrough.md)
6. [阶段 0：Dense Baseline](02_stage0_dense_baseline.md)
7. [阶段 1：LR 和 Batch Size 搜索](03_stage1_lr_batch_search.md)
8. [阶段 2：MLP 和 Attention 升级](04_stage2_block_upgrades.md)
9. [阶段 3：Tiny DeepSeekMoE](05_stage3_moe.md)
10. [阶段 4：教学版 MLA](06_stage4_mla.md)
11. [阶段 5：SFT 和 Reasoning Cold Start](07_stage5_sft_cold_start.md)
12. [阶段 6：Rule-Based GRPO Mini](08_stage6_grpo_mini.md)
13. [仓库路线图](09_repository_roadmap.md)
14. [实验报告模板](10_experiment_report_template.md)
15. [MiniMind 风格结构说明](11_minimind_structure_notes.md)
16. [GPU 选择与成本记录](13_gpu_cost_tracking.md)
17. [v1 训练执行手册](14_v1_training_runbook.md)

补充文档：

- [总训练路线图](02_training_roadmap.md)
- [当前进度](04_current_progress.md)

## 实验报告

- [RTX 4090 v1 正式结果](../../experiments/05_4090_v1_results_zh.md)
- [v1 自动汇总表和图表](../../experiments/v1_4090_plan/auto_summary_zh.md)

## 一图看懂

```mermaid
flowchart LR
  C["代码<br/>Dense LM"] --> P["预训练<br/>Dense LM"]
  P --> S["搜索<br/>LR + Batch"]
  S --> U["升级<br/>基础 Block"]
  U --> M["MoE<br/>路由专家"]
  M --> A["MLA<br/>KV 压缩"]
  A --> F["SFT<br/>推理冷启动"]
  F --> R["规则 RL<br/>GRPO Mini"]
```
