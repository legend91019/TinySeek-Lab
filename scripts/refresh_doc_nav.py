from __future__ import annotations

from pathlib import Path


EN_ORDER = [
    ("00_project_scope.md", "Project Scope"),
    ("01_deepseek_lm_paper_map.md", "DeepSeek LM Paper Map"),
    ("20_architecture_evolution_overview.md", "Architecture Evolution Map"),
    ("24_math_to_pytorch.md", "Math to PyTorch"),
    ("12_code_first_dense_lm.md", "Code First Dense LM"),
    ("21_from_dense_to_deepseek_moe.md", "Dense to DeepSeekMoE"),
    ("22_from_moe_to_deepseek_v2.md", "DeepSeekMoE to DeepSeek-V2"),
    ("23_from_v2_to_deepseek_v3.md", "DeepSeek-V2 to DeepSeek-V3"),
    ("16_training_loop_from_config_to_checkpoint.md", "Training Loop"),
    ("15_code_walkthrough.md", "Code Walkthrough"),
    ("02_stage0_dense_baseline.md", "Stage 0: Dense Baseline"),
    ("03_stage1_lr_batch_search.md", "Stage 1: LR/Batch Search"),
    ("04_stage2_block_upgrades.md", "Stage 2: Block Upgrades"),
    ("05_stage3_moe.md", "Stage 3: MoE"),
    ("06_stage4_mla.md", "Stage 4: MLA"),
    ("07_stage5_sft_cold_start.md", "Stage 5: SFT"),
    ("08_stage6_grpo_mini.md", "Stage 6: GRPO Mini"),
    ("19_posttraining_code_walkthrough.md", "Post-Training Code Walkthrough"),
    ("09_repository_roadmap.md", "Repository Roadmap"),
    ("10_experiment_report_template.md", "Experiment Template"),
    ("11_minimind_structure_notes.md", "MiniMind Structure Notes"),
    ("13_gpu_cost_tracking.md", "GPU Cost Tracking"),
    ("14_v1_training_runbook.md", "v1 Training Runbook"),
    ("17_minimind_quality_notes.md", "MiniMind Quality Notes"),
    ("18_gpu_fill_only_checklist.md", "GPU Checklist"),
]

ZH_ORDER = [
    ("00_project_scope.md", "项目范围"),
    ("01_deepseek_lm_paper_map.md", "DeepSeek 语言模型论文地图"),
    ("20_architecture_evolution_overview.md", "四代架构演进总览"),
    ("24_math_to_pytorch.md", "数学到 PyTorch"),
    ("12_code_first_dense_lm.md", "代码优先 Dense LM"),
    ("21_from_dense_to_deepseek_moe.md", "Dense 到 DeepSeekMoE"),
    ("22_from_moe_to_deepseek_v2.md", "DeepSeekMoE 到 DeepSeek-V2"),
    ("23_from_v2_to_deepseek_v3.md", "DeepSeek-V2 到 DeepSeek-V3"),
    ("16_training_loop_from_config_to_checkpoint.md", "训练主循环"),
    ("15_code_walkthrough.md", "代码导读"),
    ("02_stage0_dense_baseline.md", "阶段 0：Dense Baseline"),
    ("03_stage1_lr_batch_search.md", "阶段 1：LR/Batch 搜索"),
    ("04_stage2_block_upgrades.md", "阶段 2：Block 升级"),
    ("05_stage3_moe.md", "阶段 3：MoE"),
    ("06_stage4_mla.md", "阶段 4：MLA"),
    ("07_stage5_sft_cold_start.md", "阶段 5：SFT"),
    ("08_stage6_grpo_mini.md", "阶段 6：GRPO Mini"),
    ("19_posttraining_code_walkthrough.md", "后训练代码细读"),
    ("09_repository_roadmap.md", "仓库路线图"),
    ("10_experiment_report_template.md", "实验报告模板"),
    ("11_minimind_structure_notes.md", "MiniMind 结构说明"),
    ("13_gpu_cost_tracking.md", "GPU 成本记录"),
    ("14_v1_training_runbook.md", "v1 训练执行手册"),
    ("17_minimind_quality_notes.md", "MiniMind 质量说明"),
    ("18_gpu_fill_only_checklist.md", "上卡前 Checklist"),
]

MARKER = "<!-- tinyseek-nav -->"


def strip_old_nav(text: str) -> str:
    if MARKER not in text:
        return text.rstrip()
    return text.split(MARKER, 1)[0].rstrip()


def nav_block(order: list[tuple[str, str]], idx: int, *, zh: bool) -> str:
    prev_link = None if idx == 0 else order[idx - 1]
    next_link = None if idx == len(order) - 1 else order[idx + 1]
    index_label = "教程目录" if zh else "Tutorial Index"
    prev_label = "上一篇" if zh else "Previous"
    next_label = "下一篇" if zh else "Next"
    parts = []
    if prev_link:
        parts.append(f"{prev_label}: [{prev_link[1]}]({prev_link[0]})")
    parts.append(f"[{index_label}](README.md)")
    if next_link:
        parts.append(f"{next_label}: [{next_link[1]}]({next_link[0]})")
    return f"\n\n{MARKER}\n\n---\n\n" + " | ".join(parts) + "\n"


def refresh(root: Path, order: list[tuple[str, str]], *, zh: bool) -> None:
    for idx, (filename, _) in enumerate(order):
        path = root / filename
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        path.write_text(strip_old_nav(text) + nav_block(order, idx, zh=zh), encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    refresh(repo / "docs", EN_ORDER, zh=False)
    refresh(repo / "docs" / "zh", ZH_ORDER, zh=True)
    print("refreshed chapter navigation")


if __name__ == "__main__":
    main()
