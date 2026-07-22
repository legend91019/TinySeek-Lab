# Reference Library

[中文参考手册](zh/README.md) | English

The recommended learning path is now the integrated [`course/s01-s08`](../course/README.md) track. It keeps the model change, controlled experiment, measured result and architecture decision in one unit.

This directory is the **reference library** opened from that course. The files remain stable so existing links and citations do not break.

## Orientation and Papers

- [Project scope](00_project_scope.md)
- [DeepSeek LM paper map](01_deepseek_lm_paper_map.md)
- [Four-generation architecture overview](20_architecture_evolution_overview.md)
- [Repository roadmap](09_repository_roadmap.md)

## Code Deep Dives

- [Math to PyTorch: formulas, shapes and APIs](24_math_to_pytorch.md)
- [Build the complete Dense LM](12_code_first_dense_lm.md)
- [Dense to DeepSeekMoE](21_from_dense_to_deepseek_moe.md)
- [DeepSeekMoE to DeepSeek-V2](22_from_moe_to_deepseek_v2.md)
- [DeepSeek-V2 to DeepSeek-V3](23_from_v2_to_deepseek_v3.md)
- [Full repository code walkthrough](15_code_walkthrough.md)
- [Training loop: config to checkpoint](16_training_loop_from_config_to_checkpoint.md)
- [Post-training code walkthrough](19_posttraining_code_walkthrough.md)

## Experiment Notes

- [Dense baseline training](02_stage0_dense_baseline.md)
- [LR and batch-size search](03_stage1_lr_batch_search.md)
- [MLP and attention ablations](04_stage2_block_upgrades.md)
- [MoE experiment lab](05_stage3_moe.md)
- [MLA experiment lab](06_stage4_mla.md)
- [SFT and reasoning cold start](07_stage5_sft_cold_start.md)
- [Rule-based GRPO mini](08_stage6_grpo_mini.md)
- [Experiment report template](10_experiment_report_template.md)

## Operations and Design Notes

- [MiniMind-inspired structure notes](11_minimind_structure_notes.md)
- [GPU choice and cost tracking](13_gpu_cost_tracking.md)
- [v1 training runbook](14_v1_training_runbook.md)
- [What TinySeek learns from MiniMind](17_minimind_quality_notes.md)
- [Final GPU checklist](18_gpu_fill_only_checklist.md)

## Measured Reports

- [3-seed architecture measurements and decisions](../experiments/architecture_lab_runs/report.md)
- [Formal training and post-training](../experiments/gpu_completion_runs/report.md)
- [Experiment report hub](../experiments/README.md)
- [Fair architecture experiment plan and gates](../experiments/06_architecture_evolution_plan.md)
