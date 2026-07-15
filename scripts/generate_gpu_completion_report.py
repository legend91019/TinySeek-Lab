from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

try:
    from scripts.generate_architecture_report import bar_svg, billing_metadata, number
except ModuleNotFoundError:
    from generate_architecture_report import bar_svg, billing_metadata, number


ROOT = Path(__file__).resolve().parents[1]
POSTTRAIN_RUNS = {"formal_base20m", "formal_direct_grpo", "formal_reasoning_sft", "formal_reasoning_grpo"}


def optional_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(out_dir: Path, run_dir: Path) -> list[dict]:
    evals = {}
    for path in run_dir.glob("eval_*.json"):
        evals[path.stem.removeprefix("eval_")] = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for path in sorted(out_dir.glob("*_cost_summary.json")):
        cost = json.loads(path.read_text(encoding="utf-8"))
        name = str(cost["run_name"])
        evaluation = evals.get(name, {})
        reasoning = evaluation.get("reasoning", {})
        perplexity = evaluation.get("perplexity", {})
        val_loss = optional_number(cost.get("val_lm_loss", cost.get("val_loss")))
        val_ppl = math.exp(min(val_loss, 50.0)) if val_loss is not None else None
        history_path = Path(str(cost.get("history_path", "")))
        if not history_path.exists():
            rooted_history = ROOT / history_path
            adjacent_history = path.with_name(f"{name}_history.jsonl")
            history_path = rooted_history if rooted_history.exists() else adjacent_history
        rows.append(
            {
                "_source_path": str(path),
                "_history_path": str(history_path),
                "_data_path": str(cost.get("data_path", "")),
                "_evaluation": evaluation,
                "_hardware": cost.get("hardware", {}),
                "_hourly_rate": optional_number(cost.get("hourly_rate")),
                "_currency": cost.get("currency"),
                "run_name": name,
                "stage": cost.get("stage", "pretrain"),
                "step": int(cost.get("step", 0)),
                "model_params": int(cost.get("model_params", 0)),
                "activated_params": int(cost.get("activated_params_estimate", 0)),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "sample_ppl": optional_number(perplexity.get("ppl")),
                "sample_ppl_examples": int(perplexity.get("num_examples", 0)),
                "sample_ppl_tokens": int(perplexity.get("num_tokens", 0)),
                "tokens_per_second": optional_number(cost.get("training_tokens_per_second")),
                "peak_vram_gb": number(cost.get("max_memory_allocated_gb")),
                "gpu_hours": number(cost.get("gpu_hours")),
                "cost": number(cost.get("estimated_cost")),
                "mean_reward": optional_number(cost.get("mean_reward")),
                "reasoning_accuracy": optional_number(reasoning.get("answer_accuracy")),
                "reasoning_format": optional_number(reasoning.get("format_score")),
                "reasoning_examples": int(reasoning.get("num_examples", 0)),
                "estimated_train_tokens": int(cost.get("estimated_train_tokens", 0)) or None,
                "estimated_training_flops": optional_number(cost.get("estimated_training_flops")),
            }
        )
    return rows


def measured_findings(rows: list[dict], zh: bool) -> list[str]:
    by_name = {row["run_name"]: row for row in rows}
    sweep = [row for row in rows if row["run_name"].startswith("formal_sweep_")]
    lines = ["## 实测结论" if zh else "## Measured Findings", ""]
    lines.extend(
        [
            "| 问题 | 4090 观察 | 当前决定 |" if zh else "| Question | RTX 4090 observation | Current decision |",
            "| --- | --- | --- |",
        ]
    )
    if sweep:
        best = min(sweep, key=lambda row: row["val_loss"])
        if zh:
            lines.append(
                f"| 小范围 LR/batch 网格 | `{best['run_name']}` 的 validation loss 最低，为 `{best['val_loss']:.4f}`。 | "
                "只把它当作本预算下的 recipe 选择，不外推为 scaling law。 |"
            )
        else:
            lines.append(
                f"| Small LR/batch grid | `{best['run_name']}` has the lowest validation loss, `{best['val_loss']:.4f}`. | "
                "Use it as the recipe for this budget, not as a scaling law. |"
            )
    dense35 = by_name.get("formal_dense35_50m")
    dense115 = by_name.get("formal_dense115_30m")
    moe = by_name.get("formal_moe35_30m")
    if dense35 and dense115 and moe:
        if zh:
            lines.append(
                f"| 容量 pilot | 35M/50M-token loss `{dense35['val_loss']:.4f}`；115M/30M-token `{dense115['val_loss']:.4f}`；"
                f"MoE/30M-token `{moe['val_loss']:.4f}`。 | token budget 不同，不能据此宣称架构胜负；保留为成本与可运行性 pilot。 |"
            )
        else:
            lines.append(
                f"| Capacity pilots | 35M/50M-token loss `{dense35['val_loss']:.4f}`; 115M/30M-token `{dense115['val_loss']:.4f}`; "
                f"MoE/30M-token `{moe['val_loss']:.4f}`. | Token budgets differ, so this is a cost and feasibility pilot, not an architecture ranking. |"
            )
    base = by_name.get("formal_base20m")
    direct = by_name.get("formal_direct_grpo")
    sft = by_name.get("formal_reasoning_sft")
    grpo = by_name.get("formal_reasoning_grpo")
    if base and direct and sft and grpo:
        checkpoints = (base, direct, sft, grpo)
        score_text = ", ".join(
            f"{round(number(row['reasoning_accuracy']) * row['reasoning_examples'])}/{row['reasoning_examples']}"
            for row in checkpoints
        )
        reasoning_counts = sorted({row["reasoning_examples"] for row in checkpoints if row["reasoning_examples"]})
        reasoning_count_text = ", ".join(str(value) for value in reasoning_counts)
        sample_counts = sorted({row["sample_ppl_examples"] for row in checkpoints if row["sample_ppl_examples"]})
        sample_count_text = ", ".join(str(value) for value in sample_counts)
        all_answers_zero = all(number(row["reasoning_accuracy"]) == 0.0 for row in checkpoints)
        format_degraded = number(grpo["reasoning_format"]) < number(sft["reasoning_format"])
        if zh:
            lines.append(
                f"| 冷启动与 GRPO | base/direct-GRPO/SFT/SFT+GRPO 的留出题得分依次为 `{score_text}`（每组 n=`{reasoning_count_text}`）；推理格式分分别为 "
                f"`{base['reasoning_format']:.3f}/{direct['reasoning_format']:.3f}/{sft['reasoning_format']:.3f}/{grpo['reasoning_format']:.3f}`。 | "
                + ("所有答案均错，这组 mini-eval 没有提供算术泛化证据；" if all_answers_zero else "答案结果必须按 checkpoint 分别解释；")
                + ("当前宽松奖励的 GRPO 还破坏了格式。 |" if format_degraded else "当前 GRPO 没有造成格式分下降。 |")
            )
            lines.append(
                f"| 分布代价 | TinyStories 前 `{sample_count_text}` 条样本 PPL 从 base 的 `{base['sample_ppl']:.3f}` 上升到 SFT 的 `{sft['sample_ppl']:.3f}`，SFT+GRPO 为 `{grpo['sample_ppl']:.3f}`。 | "
                "窄域后训练会换取格式行为并损害原预训练分布；答案、格式和 PPL 必须分开报告。 |"
            )
        else:
            lines.append(
                f"| Cold start and GRPO | Base/direct-GRPO/SFT/SFT+GRPO held-out scores are `{score_text}` (n=`{reasoning_count_text}` each); reasoning-format scores are "
                f"`{base['reasoning_format']:.3f}/{direct['reasoning_format']:.3f}/{sft['reasoning_format']:.3f}/{grpo['reasoning_format']:.3f}`. | "
                + ("All answers are wrong, so this mini-eval provides no arithmetic-generalization evidence; " if all_answers_zero else "Interpret answer results per checkpoint; ")
                + ("the loose GRPO reward also damages format. |" if format_degraded else "GRPO does not lower the format score. |")
            )
            lines.append(
                f"| Distribution cost | PPL on the first `{sample_count_text}` TinyStories rows rises from `{base['sample_ppl']:.3f}` at base to `{sft['sample_ppl']:.3f}` after SFT and `{grpo['sample_ppl']:.3f}` after SFT+GRPO. | "
                "Narrow post-training trades pretraining-distribution quality for format behavior; report answer, format, and PPL separately. |"
            )
    if len(lines) == 3:
        lines.append("| - | 暂无可比较的完整实验组。 | 先完成实验再下结论。 |" if zh else "| - | No complete comparison group is available. | Run the experiment before deciding. |")
    return lines


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [field for field in rows[0] if not field.startswith("_")] if rows else ["run_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def format_metric(value: object, digits: int = 3) -> str:
    numeric = optional_number(value)
    return "N/A" if numeric is None else f"{numeric:.{digits}f}"


def dataset_manifests(rows: list[dict], run_dir: Path) -> list[dict]:
    manifests = []
    archived = {}
    plural_path = run_dir / "dataset_manifests.json"
    singular_path = run_dir / "dataset_manifest.json"
    if plural_path.exists():
        for item in json.loads(plural_path.read_text(encoding="utf-8")):
            archived[str(item.get("path", ""))] = item
    elif singular_path.exists():
        item = json.loads(singular_path.read_text(encoding="utf-8"))
        archived[str(item.get("path", ""))] = item
    seen = set()
    for row in rows:
        declared = str(row.get("_data_path", ""))
        if not declared or declared in seen:
            continue
        seen.add(declared)
        data_path = Path(declared)
        if not data_path.exists():
            data_path = ROOT / data_path
        if not data_path.is_file():
            if declared in archived:
                manifests.append(archived[declared])
            continue
        digest = hashlib.sha256()
        line_count = 0
        with data_path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                line_count += 1
        manifest = {
            "path": declared,
            "lines": line_count,
            "bytes": data_path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
        if declared in {"data/reasoning_sft.jsonl", "data/reasoning_grpo.jsonl"}:
            manifest["generated_by"] = "scripts/prepare_reasoning_data.py --sft_samples 5000 --grpo_samples 1000 --max_operand 99 --seed 7"
        manifests.append(manifest)
    return manifests


def environment_section(rows: list[dict], manifests: list[dict], zh: bool) -> list[str]:
    hardware = rows[0].get("_hardware", {}) if rows else {}
    gpu = hardware.get("gpu_name", "N/A")
    torch_version = hardware.get("torch_version", "N/A")
    cuda_version = hardware.get("cuda_version", "N/A")
    rate, currency = billing_metadata(rows)
    rate_text = format_metric(rate, 2)
    if zh:
        lines = [
            "## 环境与数据",
            "",
            f"- GPU：`{gpu}`；PyTorch：`{torch_version}`；CUDA：`{cuda_version}`。",
            f"- 计价：`{rate_text} {currency}/h`。账本只统计训练/后训练进程，不含数据准备、独立 mini-eval、报告生成和租卡空闲时间；估算费用不是平台账单。",
        ]
        for manifest in manifests:
            lines.append(f"- 数据：`{manifest['path']}`，`{manifest['lines']}` 行，`{manifest['bytes']}` bytes，SHA256 `{manifest['sha256']}`。")
        lines.extend(["- 完整数据清单：[`dataset_manifests.json`](dataset_manifests.json)。", "- 逐 run 配置在 [`configs/`](configs/)，原始成本与 history 在 [`raw/`](raw/)。", "", "## 复现", "", "```bash", "python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl", f"python scripts/run_gpu_completion.py --data data/tinystories.jsonl --hourly_rate {rate:g} --currency {currency}", "python scripts/generate_gpu_completion_report.py", "```"])
        return lines
    lines = [
        "## Environment and Data",
        "",
        f"- GPU: `{gpu}`; PyTorch: `{torch_version}`; CUDA: `{cuda_version}`.",
        f"- Rate: `{rate_text} {currency}/h`. The ledger covers training/post-training processes and excludes data preparation, standalone evaluation, report generation, and idle rental time; estimated cost is not the platform invoice.",
    ]
    for manifest in manifests:
        lines.append(f"- Data: `{manifest['path']}`, `{manifest['lines']}` lines, `{manifest['bytes']}` bytes, SHA256 `{manifest['sha256']}`.")
    lines.extend(["- Full data manifest: [`dataset_manifests.json`](dataset_manifests.json).", "- Per-run configs are in [`configs/`](configs/); raw cost and history ledgers are in [`raw/`](raw/).", "", "## Reproduce", "", "```bash", "python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl", f"python scripts/run_gpu_completion.py --data data/tinystories.jsonl --hourly_rate {rate:g} --currency {currency}", "python scripts/generate_gpu_completion_report.py", "```"])
    return lines


def failure_examples(rows: list[dict], zh: bool) -> list[str]:
    wanted = {"formal_reasoning_sft", "formal_reasoning_grpo"}
    lines = ["## 留出题失败样例" if zh else "## Held-Out Failure Examples", ""]
    lines.extend(["| Checkpoint | Prompt | Target | Prediction | Format | Completion excerpt |", "| --- | --- | ---: | ---: | --- | --- |"])
    found = False
    for row in rows:
        if row["run_name"] not in wanted:
            continue
        examples = row.get("_evaluation", {}).get("reasoning", {}).get("examples", [])
        for example in examples[:2]:
            found = True
            completion = str(example.get("completion", "")).replace("|", "\\|").replace("\n", "<br>")
            prediction = example.get("prediction")
            prediction_text = "N/A" if prediction is None else str(prediction)
            lines.append(f"| `{row['run_name']}` | `{example.get('prompt', '')}` | {example.get('target', 'N/A')} | {prediction_text} | {example.get('format_ok', False)} | {completion} |")
    if not found:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A |")
    return lines


def report_text(rows: list[dict], manifests: list[dict], zh: bool) -> str:
    total_hours = sum(number(row["gpu_hours"]) for row in rows)
    total_cost = sum(number(row["cost"]) for row in rows)
    sweep = [row for row in rows if row["run_name"].startswith("formal_sweep_")]
    best_sweep = min(sweep, key=lambda row: row["val_loss"])["run_name"] if sweep else "N/A"
    rate, currency = billing_metadata(rows)
    if zh:
        title = "TinySeek 完整 GPU 训练与后训练报告"
        intro = "本表使用真实 RTX 4090 运行生成。语言模型 PPL 与后训练推理指标分开报告。"
        summary = f"- 已记录训练/后训练进程时间：`{total_hours:.4f} h`\n- 按 {format_metric(rate, 2)} {currency}/h 对应估算费用：`{total_cost:.4f} {currency}`\n- 最佳 LR/batch sweep：`{best_sweep}`"
        metric_title = "训练与成本"
        reasoning_title = "后训练推理答案与格式"
        reasoning_counts = sorted({row["reasoning_examples"] for row in rows if row["reasoning_examples"]})
        sample_counts = sorted({row["sample_ppl_examples"] for row in rows if row["sample_ppl_examples"]})
        reasoning_label = ", ".join(str(value) for value in reasoning_counts) or "N/A"
        sample_label = ", ".join(str(value) for value in sample_counts) or "N/A"
        boundary = f"结果只支持本仓小模型、数据和 token budget 下的结论，不外推到 DeepSeek 原始规模。Reasoning 留出题数量为 {reasoning_label}；sample PPL 使用 TinyStories JSONL 前 {sample_label} 条，不是固定 held-out split。"
    else:
        title = "TinySeek Complete GPU Training and Post-Training Report"
        intro = "These tables come from real RTX 4090 runs. Language-model PPL and post-training reasoning metrics are reported separately."
        summary = f"- Tracked training/post-training process time: `{total_hours:.4f} h`\n- Corresponding estimate at {format_metric(rate, 2)} {currency}/h: `{total_cost:.4f} {currency}`\n- Best LR/batch sweep: `{best_sweep}`"
        metric_title = "Training and Cost"
        reasoning_title = "Post-Training Reasoning Answer and Format"
        reasoning_counts = sorted({row["reasoning_examples"] for row in rows if row["reasoning_examples"]})
        sample_counts = sorted({row["sample_ppl_examples"] for row in rows if row["sample_ppl_examples"]})
        reasoning_label = ", ".join(str(value) for value in reasoning_counts) or "N/A"
        sample_label = ", ".join(str(value) for value in sample_counts) or "N/A"
        boundary = f"Results support conclusions only for this repository's small models, data, and token budgets; they do not extrapolate to DeepSeek scale. Held-out reasoning counts are {reasoning_label}; sample PPL uses the first {sample_label} TinyStories JSONL rows, not a fixed held-out split."
    lines = [f"# {title}", "", intro, "", summary, ""]
    lines.extend(environment_section(rows, manifests, zh))
    lines.append("")
    lines.extend(measured_findings(rows, zh))
    lines.extend(["", f"## {metric_title}", ""])
    sample_counts = sorted({row["sample_ppl_examples"] for row in rows if row["sample_ppl_examples"]})
    sample_label = "/".join(str(value) for value in sample_counts) or "N/A"
    lines.append(f"`val PPL = exp(val loss)` 使用各 stage 自己的 validation 数据，只能在相同 stage/data 内比较；`sample PPL` 才是 checkpoint 在同一 TinyStories 前 {sample_label} 条上的分布探针。" if zh else f"`val PPL = exp(val loss)` uses each stage's own validation data and is comparable only within the same stage/data; `sample PPL` is the shared distribution probe over the first {sample_label} TinyStories rows.")
    lines.append("")
    lines.extend(
        [
            f"| Run | Stage | Steps | Params | Activated | val loss | val PPL | sample PPL | tok/s | peak GB | GPU h | cost {currency} |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['run_name']}` | {row['stage']} | {row['step']} | {row['model_params'] / 1e6:.2f}M | "
            f"{row['activated_params'] / 1e6:.2f}M | {format_metric(row['val_loss'], 4)} | {format_metric(row['val_ppl'])} | {format_metric(row['sample_ppl'])} | "
            f"{format_metric(row['tokens_per_second'], 0)} | {row['peak_vram_gb']:.3f} | {row['gpu_hours']:.4f} | {row['cost']:.4f} |"
        )
    lines.extend(["", "## 计算量账本" if zh else "## Compute Ledger", "", "| Run | 训练 tokens | 粗略 FLOPs |" if zh else "| Run | Train tokens | Rough FLOPs |", "| --- | ---: | ---: |"])
    for row in rows:
        token_text = "N/A" if row["estimated_train_tokens"] is None else f"{row['estimated_train_tokens']:,}"
        flops_text = "N/A" if row["estimated_training_flops"] is None else f"{row['estimated_training_flops']:.3e}"
        lines.append(f"| `{row['run_name']}` | {token_text} | {flops_text} |")
    lines.extend(["", f"## {reasoning_title}", "", "| Checkpoint | Answer accuracy | Format score | GRPO mean reward |", "| --- | ---: | ---: | ---: |"])
    for row in rows:
        if row["reasoning_accuracy"] or row["reasoning_format"] or row["run_name"] in POSTTRAIN_RUNS:
            lines.append(f"| `{row['run_name']}` | {format_metric(row['reasoning_accuracy'])} | {format_metric(row['reasoning_format'])} | {format_metric(row['mean_reward'])} |")
    lines.extend([""] + failure_examples(rows, zh))
    lines.extend(
        [
            "",
            "## Evidence Boundary" if not zh else "## 证据边界",
            "",
            boundary,
            "",
            "## Figures" if not zh else "## 图表",
            "",
            "![Validation loss](figures/formal_val_loss.svg)",
            "",
            "![Training throughput](figures/formal_throughput.svg)",
            "",
            "![Peak VRAM](figures/formal_vram.svg)",
            "",
            "![GPU cost](figures/formal_cost.svg)",
            "",
            "![Posttraining reasoning](figures/posttraining_reasoning.svg)",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(out_dir: Path, run_dir: Path) -> None:
    rows = load_rows(out_dir, run_dir)
    if not rows:
        raise ValueError("No GPU completion cost summaries found")
    run_dir.mkdir(parents=True, exist_ok=True)
    _, currency = billing_metadata(rows)
    manifests = dataset_manifests(rows, run_dir)
    write_csv(run_dir / "results.csv", rows)
    (run_dir / "report.md").write_text(report_text(rows, manifests, zh=False), encoding="utf-8")
    (run_dir / "report_zh.md").write_text(report_text(rows, manifests, zh=True), encoding="utf-8")
    figures = run_dir / "figures"
    pretrain = [row for row in rows if row["val_loss"] is not None]
    bar_svg("Formal validation loss", [(row["run_name"], row["val_loss"]) for row in pretrain], figures / "formal_val_loss.svg")
    bar_svg("Training throughput", [(row["run_name"], row["tokens_per_second"]) for row in rows if row["tokens_per_second"] is not None], figures / "formal_throughput.svg", " tok/s")
    bar_svg("Peak allocated VRAM", [(row["run_name"], row["peak_vram_gb"]) for row in rows], figures / "formal_vram.svg", " GB")
    bar_svg("Estimated GPU cost", [(row["run_name"], row["cost"]) for row in rows], figures / "formal_cost.svg", f" {currency}")
    reasoning = []
    for row in rows:
        if row["run_name"] in POSTTRAIN_RUNS:
            reasoning.extend([(f"{row['run_name']} answer", number(row["reasoning_accuracy"])), (f"{row['run_name']} format", number(row["reasoning_format"]))])
    bar_svg("Post-training reasoning metrics", reasoning, figures / "posttraining_reasoning.svg")
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        source = Path(str(row.get("_source_path", "")))
        if source.exists():
            destination = raw_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        history = Path(str(row.get("_history_path", "")))
        if not history.exists():
            history = ROOT / history
        if history.is_file():
            destination = raw_dir / history.name
            if history.resolve() != destination.resolve():
                shutil.copy2(history, destination)
    (run_dir / "dataset_manifests.json").write_text(json.dumps(manifests, indent=2) + "\n", encoding="utf-8")
    if manifests:
        (run_dir / "dataset_manifest.json").write_text(json.dumps(manifests[0], indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the complete TinySeek GPU suite report")
    parser.add_argument("--input_dir", default="out/gpu_completion")
    parser.add_argument("--run_dir", default="experiments/gpu_completion_runs")
    args = parser.parse_args()
    build_report(ROOT / args.input_dir, ROOT / args.run_dir)
    print(f"generated GPU completion report in {args.run_dir}")


if __name__ == "__main__":
    main()
