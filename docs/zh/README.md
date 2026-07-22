# 参考手册

中文 | [English](../README.md)

现在唯一推荐的学习顺序是综合的 [`course/s01-s08`](../../course/README_zh.md) 主线。每个单元会把模型改变、控制变量实验、实测结果和架构决定写在一起。

本目录改为课程按需打开的**参考手册**。旧文件和路径继续保留，因此已有链接和引用不会失效。

## 范围与论文

- [项目范围](00_project_scope.md)
- [DeepSeek 语言模型论文地图](01_deepseek_lm_paper_map.md)
- [四代架构演进总览](20_architecture_evolution_overview.md)
- [仓库路线图](09_repository_roadmap.md)
- [总训练路线图](02_training_roadmap.md)
- [当前进度](04_current_progress.md)

## 代码深读

- [数学到 PyTorch：公式、shape 与 API](24_math_to_pytorch.md)
- [写出完整 Dense LM](12_code_first_dense_lm.md)
- [Dense 到 DeepSeekMoE](21_from_dense_to_deepseek_moe.md)
- [DeepSeekMoE 到 DeepSeek-V2](22_from_moe_to_deepseek_v2.md)
- [DeepSeek-V2 到 DeepSeek-V3](23_from_v2_to_deepseek_v3.md)
- [完整仓库代码导读](15_code_walkthrough.md)
- [训练主循环：config 到 checkpoint](16_training_loop_from_config_to_checkpoint.md)
- [后训练代码细读](19_posttraining_code_walkthrough.md)

## 实验讲义

- [Dense 基线训练](02_stage0_dense_baseline.md)
- [LR 与 batch-size 搜索](03_stage1_lr_batch_search.md)
- [MLP 与 attention 消融](04_stage2_block_upgrades.md)
- [MoE 实验课](05_stage3_moe.md)
- [MLA 实验课](06_stage4_mla.md)
- [SFT 与 reasoning cold start](07_stage5_sft_cold_start.md)
- [规则 GRPO mini](08_stage6_grpo_mini.md)
- [实验报告模板](10_experiment_report_template.md)

## 运行与设计记录

- [MiniMind 风格结构说明](11_minimind_structure_notes.md)
- [GPU 选择与成本记录](13_gpu_cost_tracking.md)
- [v1 训练执行手册](14_v1_training_runbook.md)
- [向 MiniMind 学什么](17_minimind_quality_notes.md)
- [上卡前最终 checklist](18_gpu_fill_only_checklist.md)

## 实测报告

- [3-seed 架构实测与决定](../../experiments/architecture_lab_runs/report_zh.md)
- [正式训练与后训练](../../experiments/gpu_completion_runs/report_zh.md)
- [实验报告中心](../../experiments/README_zh.md)
- [公平架构实验计划与门槛](../../experiments/06_architecture_evolution_plan_zh.md)
