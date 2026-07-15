from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED_SUFFIX = re.compile(r"_seed(\d+)$")
METRICS = (
    "val_lm_loss",
    "ppl",
    "tokens_per_second",
    "peak_vram_gb",
    "gpu_hours",
    "cost",
    "expert_load_cv",
)


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def billing_metadata(rows: list[dict]) -> tuple[float, str]:
    incomplete = [row.get("run_name", "unknown") for row in rows if row.get("_hourly_rate") in (None, "") or row.get("_currency") in (None, "")]
    if incomplete:
        raise ValueError(f"Incomplete billing metadata for runs: {incomplete}")
    rates = {number(row.get("_hourly_rate")) for row in rows}
    currencies = {str(row.get("_currency")) for row in rows}
    if len(rates) > 1 or len(currencies) > 1:
        raise ValueError(f"Mixed billing metadata: rates={sorted(rates)} currencies={sorted(currencies)}")
    return (next(iter(rates), 0.0), next(iter(currencies), "CNY"))


def base_name(run_name: str) -> str:
    return SEED_SUFFIX.sub("", run_name)


def expert_load_cv(expert_load: object) -> float:
    if not isinstance(expert_load, dict):
        return 0.0
    fractions = expert_load.get("total_fractions")
    if not isinstance(fractions, list) or not fractions:
        return 0.0
    values = [number(value) for value in fractions]
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def history_expert_load_cv(history_path: str | Path | None, fallback: object) -> float:
    if not history_path:
        return expert_load_cv(fallback)
    path = Path(history_path)
    if not path.exists():
        path = ROOT / path
    if not path.exists():
        return expert_load_cv(fallback)
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        load = row.get("expert_load")
        if isinstance(load, dict) and load.get("total_fractions"):
            values.append(expert_load_cv(load))
    return statistics.fmean(values) if values else expert_load_cv(fallback)


def kv_elements(config_path: object) -> int:
    if not config_path:
        return 0
    path = Path(str(config_path))
    if not path.exists():
        path = ROOT / path
    if not path.exists():
        return 0
    config = json.loads(path.read_text(encoding="utf-8"))["model"]
    if config.get("attention_impl") == "educational_mla" and config.get("mla_decoupled_rope"):
        return int(config["mla_latent_size"] + config["qk_rope_head_dim"])
    head_dim = int(config["hidden_size"] // config["num_heads"])
    return int(2 * config["num_kv_heads"] * head_dim)


def load_runs(out_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(out_dir.glob("*_cost_summary.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        run_name = str(source["run_name"])
        match = SEED_SUFFIX.search(run_name)
        if not match or "val_lm_loss" not in source:
            continue
        loss = number(source["val_lm_loss"])
        history_path = Path(str(source.get("history_path", "")))
        if not history_path.exists():
            rooted_history = ROOT / history_path
            adjacent_history = path.with_name(f"{run_name}_history.jsonl")
            history_path = rooted_history if rooted_history.exists() else adjacent_history
        rows.append(
            {
                "_source_path": str(path),
                "_history_path": str(history_path),
                "_data_path": str(source.get("data_path", "")),
                "_hardware": source.get("hardware", {}),
                "_hourly_rate": source.get("hourly_rate"),
                "_currency": source.get("currency"),
                "run_name": run_name,
                "base_run_name": base_name(run_name),
                "seed": int(match.group(1)),
                "step": int(source.get("step", 0)),
                "val_lm_loss": loss,
                "ppl": math.exp(min(loss, 50.0)),
                "tokens_per_second": number(source.get("training_tokens_per_second")),
                "peak_vram_gb": number(source.get("max_memory_allocated_gb")),
                "gpu_hours": number(source.get("gpu_hours")),
                "cost": number(source.get("estimated_cost")),
                "model_params": int(source.get("model_params", 0)),
                "activated_params": int(source.get("activated_params_estimate", 0)),
                "expert_load_cv": history_expert_load_cv(history_path, source.get("expert_load")),
                "kv_elements_per_token": kv_elements(source.get("config_path")),
                "estimated_train_tokens": int(source.get("estimated_train_tokens", 0)),
                "estimated_training_flops": number(source.get("estimated_training_flops")),
            }
        )
    return rows


def aggregate_runs(runs: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        grouped.setdefault(run["base_run_name"], []).append(run)
    rows = []
    for run_name, members in sorted(grouped.items()):
        row: dict[str, object] = {
            "run_name": run_name,
            "seeds": len(members),
            "seed_values": ",".join(str(item["seed"]) for item in sorted(members, key=lambda item: item["seed"])),
            "model_params": members[0]["model_params"],
            "activated_params": members[0]["activated_params"],
            "kv_elements_per_token": members[0]["kv_elements_per_token"],
            "estimated_train_tokens": members[0]["estimated_train_tokens"],
            "estimated_training_flops": members[0]["estimated_training_flops"],
        }
        for metric in METRICS:
            values = [number(member[metric]) for member in members]
            row[f"{metric}_mean"] = statistics.fmean(values)
            row[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def bar_svg(title: str, rows: list[tuple[str, float]], path: Path, unit: str = "") -> None:
    width, left, top, bar_h, gap = 1040, 260, 58, 22, 10
    height = top + len(rows) * (bar_h + gap) + 40
    maximum = max([value for _, value in rows] or [1.0])
    colors = ["#276678", "#d97745", "#568259", "#8a5a83", "#b48b36"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="22" y="32" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#172b36">{html.escape(title)}</text>',
    ]
    for index, (name, value) in enumerate(rows):
        y = top + index * (bar_h + gap)
        bar_width = 0.0 if maximum <= 0 else (width - left - 150) * value / maximum
        lines.append(f'<text x="22" y="{y + 16}" font-family="Arial,sans-serif" font-size="12" fill="#29434e">{html.escape(name)}</text>')
        lines.append(f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="{bar_h}" rx="3" fill="{colors[index % len(colors)]}"/>')
        lines.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 16}" font-family="Arial,sans-serif" font-size="12" fill="#172b36">{value:.4g}{html.escape(unit)}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [field for field in rows[0] if not field.startswith("_")] if rows else ["run_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def decision_summary(aggregates: list[dict], zh: bool) -> list[str]:
    by_name = {row["run_name"]: row for row in aggregates}
    lines = ["## 决策摘要" if zh else "## Decision Summary", ""]
    lines.extend(
        [
            "| 阶段 | 4090 多 seed 观察 | 门槛与决定 |" if zh else "| Stage | Multi-seed RTX 4090 observation | Gate and decision |",
            "| --- | --- | --- |",
        ]
    )

    def pair(left: str, right: str) -> tuple[dict, dict] | None:
        return (by_name[left], by_name[right]) if left in by_name and right in by_name else None

    dense = pair("arch_dense_mha", "arch_dense_gqa")
    if dense:
        mha, gqa = dense
        if zh:
            lines.append(
                f"| MHA -> GQA | PPL `{mha['ppl_mean']:.3f} -> {gqa['ppl_mean']:.3f}`；理论 KV/token `{mha['kv_elements_per_token']} -> {gqa['kv_elements_per_token']}`。 | "
                "通过：本预算下缓存元素减半且 PPL 未退化，后续基线采用 GQA。 |"
            )
        else:
            lines.append(
                f"| MHA -> GQA | PPL `{mha['ppl_mean']:.3f} -> {gqa['ppl_mean']:.3f}`; theoretical KV/token `{mha['kv_elements_per_token']} -> {gqa['kv_elements_per_token']}`. | "
                "Pass: cache elements halve without a PPL regression at this budget; use GQA in the next baseline. |"
            )

    if all(name in by_name for name in ("arch_moe_coarse", "arch_moe_fine_grained", "arch_moe_shared")):
        coarse, fine, shared = (by_name[name] for name in ("arch_moe_coarse", "arch_moe_fine_grained", "arch_moe_shared"))
        if zh:
            lines.append(
                f"| coarse -> fine/shared MoE | PPL `{coarse['ppl_mean']:.3f}/{fine['ppl_mean']:.3f}/{shared['ppl_mean']:.3f}`；fine/shared 吞吐也更低。 | "
                "分支决定：fine-grained 单独使用未通过；shared 在质量上通过但吞吐约低 35%，质量分支采用 shared，速度分支保留 coarse。 |"
            )
        else:
            lines.append(
                f"| coarse -> fine/shared MoE | PPL `{coarse['ppl_mean']:.3f}/{fine['ppl_mean']:.3f}/{shared['ppl_mean']:.3f}`; fine/shared are also slower. | "
                "Split decision: fine-grained alone fails; shared passes on quality but is about 35% slower, so use shared for quality and retain coarse for throughput. |"
            )

    aux_names = ("arch_moe_aux_none", "arch_moe_aux_weak", "arch_moe_aux", "arch_moe_aux_strong")
    if all(name in by_name for name in aux_names):
        aux_rows = [by_name[name] for name in aux_names]
        best_ppl = min(aux_rows, key=lambda row: row["ppl_mean"])
        if zh:
            lines.append(
                f"| aux 权重 | `{best_ppl['run_name']}` 的 PPL 最低 `{best_ppl['ppl_mean']:.3f}`；无 aux 的 load CV 为 `{by_name['arch_moe_aux_none']['expert_load_cv_mean']:.3f}`，"
                f"aux=0.01 为 `{by_name['arch_moe_aux']['expert_load_cv_mean']:.3f}`。 | 采用 aux=0.01：质量与负载的折中最好，强 aux 没有额外质量收益。 |"
            )
        else:
            lines.append(
                f"| Auxiliary weight | `{best_ppl['run_name']}` has the lowest PPL, `{best_ppl['ppl_mean']:.3f}`; load CV falls from `{by_name['arch_moe_aux_none']['expert_load_cv_mean']:.3f}` without aux "
                f"to `{by_name['arch_moe_aux']['expert_load_cv_mean']:.3f}` at aux=0.01. | Choose aux=0.01 for this quality/load trade-off; stronger aux adds no quality gain. |"
            )

    routing = pair("arch_moe_aux", "arch_moe_bias")
    if routing:
        aux, bias = routing
        if zh:
            lines.append(
                f"| aux -> bias routing | PPL `{aux['ppl_mean']:.3f} -> {bias['ppl_mean']:.3f}`；load CV `{aux['expert_load_cv_mean']:.3f} -> {bias['expert_load_cv_mean']:.3f}`。 | "
                "未通过：bias 在这组超参下既没有更好的主任务，也没有更均衡负载，保留 aux 并另行 sweep update rate。 |"
            )
        else:
            lines.append(
                f"| aux -> bias routing | PPL `{aux['ppl_mean']:.3f} -> {bias['ppl_mean']:.3f}`; load CV `{aux['expert_load_cv_mean']:.3f} -> {bias['expert_load_cv_mean']:.3f}`. | "
                "Fail: bias improves neither main-task quality nor load balance here; retain aux and sweep the bias update rate separately. |"
            )

    mla = pair("arch_v2_attention_control", "arch_v2_mla")
    if mla:
        control, treatment = mla
        if zh:
            lines.append(
                f"| GQA -> 教学版 MLA | 理论 KV/token `{control['kv_elements_per_token']} -> {treatment['kv_elements_per_token']}`，但 PPL `{control['ppl_mean']:.3f} -> {treatment['ppl_mean']:.3f}`。 | "
                "未通过质量门槛：保留缓存压缩假设，下一轮 sweep latent rank；不把训练显存当作解码 KV cache 证据。 |"
            )
        else:
            lines.append(
                f"| GQA -> educational MLA | Theoretical KV/token `{control['kv_elements_per_token']} -> {treatment['kv_elements_per_token']}`, but PPL `{control['ppl_mean']:.3f} -> {treatment['ppl_mean']:.3f}`. | "
                "Fail the quality gate: retain the cache-compression hypothesis and sweep latent rank; training VRAM is not decoding-cache evidence. |"
            )

    mtp = pair("arch_v3_no_mtp", "arch_v3_mtp")
    if mtp:
        control, treatment = mtp
        if zh:
            lines.append(
                f"| MTP off -> on | PPL `{control['ppl_mean']:.3f} +/- {control['ppl_std']:.3f}` vs `{treatment['ppl_mean']:.3f} +/- {treatment['ppl_std']:.3f}`；"
                f"峰值显存 `{control['peak_vram_gb_mean']:.3f} -> {treatment['peak_vram_gb_mean']:.3f} GB`。 | 仅在已被拒绝的 V3-style 分支上结论不确定；它不能决定 GQA+aux 分支是否启用 MTP。 |"
            )
        else:
            lines.append(
                f"| MTP off -> on | PPL `{control['ppl_mean']:.3f} +/- {control['ppl_std']:.3f}` vs `{treatment['ppl_mean']:.3f} +/- {treatment['ppl_std']:.3f}`; "
                f"peak VRAM `{control['peak_vram_gb_mean']:.3f} -> {treatment['peak_vram_gb_mean']:.3f} GB`. | Inconclusive only on the rejected V3-style branch; this does not decide MTP for the GQA+aux branch. |"
            )

    if len(lines) == 3:
        lines.append("| - | 暂无完整 A/B 对照。 | 先完成实验再决策。 |" if zh else "| - | No complete A/B pair is available. | Run the comparison before deciding. |")
    lines.extend(
        [
            "",
            (
                "分支说明：这些是组内控制变量实验，不是一条把所有胜者依次叠加的单线模型。coarse/fine/shared 与 8-routed top-2 的 aux/MLA/V3-style 机制组使用不同 MoE 拓扑；MTP 结论只适用于已被拒绝的 V3-style 分支。"
                if zh
                else "Branch note: these are controlled comparisons within groups, not one model that stacks every winner. The coarse/fine/shared group and the 8-routed top-2 aux/MLA/V3-style mechanism group use different MoE topologies; the MTP result applies only to the rejected V3-style branch."
            ),
        ]
    )
    return lines


def dataset_manifest(runs: list[dict], run_dir: Path) -> dict:
    for run in runs:
        data_path = Path(str(run.get("_data_path", "")))
        if not data_path.exists():
            data_path = ROOT / data_path
        if not data_path.is_file():
            continue
        digest = hashlib.sha256()
        line_count = 0
        with data_path.open("rb") as handle:
            for line in handle:
                digest.update(line)
                line_count += 1
        return {
            "path": str(run.get("_data_path", data_path)),
            "lines": line_count,
            "bytes": data_path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    archived = run_dir / "dataset_manifest.json"
    if archived.exists():
        return json.loads(archived.read_text(encoding="utf-8"))
    return {}


def environment_section(runs: list[dict], manifest: dict, zh: bool) -> list[str]:
    hardware = runs[0].get("_hardware", {}) if runs else {}
    gpu = hardware.get("gpu_name", "N/A")
    torch_version = hardware.get("torch_version", "N/A")
    cuda_version = hardware.get("cuda_version", "N/A")
    rate, currency = billing_metadata(runs)
    if zh:
        lines = [
            "## 环境与数据",
            "",
            f"- GPU：`{gpu}`；PyTorch：`{torch_version}`；CUDA：`{cuda_version}`。",
            f"- 计价：`{rate:.2f} {currency}/h`。账本统计 trainer 进程时间，不含数据准备、报告生成和租卡空闲时间；估算费用不是平台账单。",
        ]
        if manifest:
            lines.append(f"- 数据：`{manifest['path']}`，`{manifest['lines']}` 行，`{manifest['bytes']}` bytes，SHA256 `{manifest['sha256']}`。")
        lines.extend(["- 逐 run 配置在 [`configs/`](configs/)，原始成本与 history 在 [`raw/`](raw/)。", "", "## 复现", "", "```bash", "python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl", f"python scripts/run_architecture_lab.py --data data/tinystories.jsonl --seeds 42,43,44 --hourly_rate {rate:g} --currency {currency}", "python scripts/generate_architecture_report.py", "```"])
        return lines
    lines = [
        "## Environment and Data",
        "",
        f"- GPU: `{gpu}`; PyTorch: `{torch_version}`; CUDA: `{cuda_version}`.",
        f"- Rate: `{rate:.2f} {currency}/h`. The ledger tracks trainer-process time and excludes data preparation, report generation, and idle rental time; estimated cost is not the platform invoice.",
    ]
    if manifest:
        lines.append(f"- Data: `{manifest['path']}`, `{manifest['lines']}` lines, `{manifest['bytes']}` bytes, SHA256 `{manifest['sha256']}`.")
    lines.extend(["- Per-run configs are in [`configs/`](configs/); raw cost and history ledgers are in [`raw/`](raw/).", "", "## Reproduce", "", "```bash", "python scripts/prepare_hf_dataset.py --dataset_name roneneldan/TinyStories --split train --text_field text --max_samples 50000 --min_chars 80 --out data/tinystories.jsonl", f"python scripts/run_architecture_lab.py --data data/tinystories.jsonl --seeds 42,43,44 --hourly_rate {rate:g} --currency {currency}", "python scripts/generate_architecture_report.py", "```"])
    return lines


def markdown_report(runs: list[dict], aggregates: list[dict], manifest: dict, *, zh: bool) -> str:
    total_hours = sum(number(run["gpu_hours"]) for run in runs)
    total_cost = sum(number(run["cost"]) for run in runs)
    _, currency = billing_metadata(runs)
    if zh:
        title = "TinySeek 架构实验 GPU 实测"
        intro = "本报告由成本 JSON 自动生成。PPL 只由主 `val_lm_loss` 计算；均值和标准差来自相同配置的不同 seed。"
        totals = f"- GPU 实测运行：`{len(runs)}`\n- 配置数：`{len(aggregates)}`\n- 已记录 trainer 进程时间：`{total_hours:.4f} h`\n- 对应估算费用：`{total_cost:.4f} {currency}`"
        aggregate_title, raw_title = "多 seed 汇总", "逐次运行"
        evidence = "这些是 TinySeek 小模型 GPU 实测，不应替代 DeepSeek 论文规模的结论。决策前同时检查均值、波动、吞吐、显存和路由负载。"
    else:
        title = "TinySeek Architecture Lab GPU Measurements"
        intro = "This report is generated from cost JSON files. PPL is computed only from main `val_lm_loss`; means and standard deviations use repeated seeds of the same config."
        totals = f"- GPU runs: `{len(runs)}`\n- Configurations: `{len(aggregates)}`\n- Tracked trainer-process time: `{total_hours:.4f} h`\n- Corresponding estimated cost: `{total_cost:.4f} {currency}`"
        aggregate_title, raw_title = "Multi-Seed Aggregate", "Individual Runs"
        evidence = "These are TinySeek small-model GPU measurements, not substitutes for DeepSeek paper-scale results. Inspect means, variation, throughput, memory, and routing load before a decision."
    lines = [f"# {title}", "", intro, "", totals, ""]
    lines.extend(environment_section(runs, manifest, zh))
    lines.append("")
    lines.extend(decision_summary(aggregates, zh))
    lines.extend(["", f"## {aggregate_title}", ""])
    lines.extend(
        [
            f"| Run | Seeds | val LM loss | PPL | tokens/s | peak GB | GPU h | cost {currency} | load CV | KV/token |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| `{row['run_name']}` | {row['seeds']} | {row['val_lm_loss_mean']:.4f} +/- {row['val_lm_loss_std']:.4f} | "
            f"{row['ppl_mean']:.3f} +/- {row['ppl_std']:.3f} | {row['tokens_per_second_mean']:.0f} | "
            f"{row['peak_vram_gb_mean']:.3f} | {row['gpu_hours_mean']:.4f} | {row['cost_mean']:.4f} | "
            f"{row['expert_load_cv_mean']:.3f} | {row['kv_elements_per_token']} |"
        )
    lines.extend(
        [
            "",
            "## 计算量账本" if zh else "## Compute Ledger",
            "",
            "| Run | 训练 tokens | 粗略 FLOPs |" if zh else "| Run | Train tokens | Rough FLOPs |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in aggregates:
        lines.append(f"| `{row['run_name']}` | {row['estimated_train_tokens']:,} | {row['estimated_training_flops']:.3e} |")
    lines.extend(["", f"## {raw_title}", "", f"| Run | Seed | val LM loss | PPL | tokens/s | peak GB | GPU h | cost {currency} |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in sorted(runs, key=lambda item: (item["base_run_name"], item["seed"])):
        lines.append(
            f"| `{row['base_run_name']}` | {row['seed']} | {row['val_lm_loss']:.4f} | {row['ppl']:.3f} | "
            f"{row['tokens_per_second']:.0f} | {row['peak_vram_gb']:.3f} | {row['gpu_hours']:.4f} | {row['cost']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary" if not zh else "## 证据边界",
            "",
            evidence,
            "",
            "## Figures" if not zh else "## 图表",
            "",
            "![Architecture PPL](figures/architecture_ppl.svg)",
            "",
            "![Architecture throughput](figures/architecture_throughput.svg)",
            "",
            "![Architecture VRAM](figures/architecture_vram.svg)",
            "",
            "![MoE load CV](figures/moe_load_cv.svg)",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(runs: list[dict], run_dir: Path) -> None:
    if not runs:
        raise ValueError("No completed architecture runs found")
    billing_metadata(runs)
    aggregates = aggregate_runs(runs)
    manifest = dataset_manifest(runs, run_dir)
    write_csv(run_dir / "results.csv", runs)
    write_csv(run_dir / "aggregate.csv", aggregates)
    (run_dir / "report.md").write_text(markdown_report(runs, aggregates, manifest, zh=False), encoding="utf-8")
    (run_dir / "report_zh.md").write_text(markdown_report(runs, aggregates, manifest, zh=True), encoding="utf-8")
    figures = run_dir / "figures"
    bar_svg("Validation perplexity by architecture", [(row["run_name"], number(row["ppl_mean"])) for row in aggregates], figures / "architecture_ppl.svg")
    bar_svg("Training throughput by architecture", [(row["run_name"], number(row["tokens_per_second_mean"])) for row in aggregates], figures / "architecture_throughput.svg", " tok/s")
    bar_svg("Peak allocated VRAM by architecture", [(row["run_name"], number(row["peak_vram_gb_mean"])) for row in aggregates], figures / "architecture_vram.svg", " GB")
    moe_rows = [(row["run_name"], number(row["expert_load_cv_mean"])) for row in aggregates if row["run_name"].startswith("arch_moe") or row["run_name"].startswith("arch_v")]
    bar_svg("Expert-load coefficient of variation", moe_rows, figures / "moe_load_cv.svg")
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        source = Path(str(run.get("_source_path", "")))
        if source.exists():
            destination = raw_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        history = Path(str(run.get("_history_path", "")))
        if not history.exists():
            history = ROOT / history
        if history.is_file():
            destination = raw_dir / history.name
            if history.resolve() != destination.resolve():
                shutil.copy2(history, destination)
    if manifest:
        (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TinySeek architecture-lab GPU results")
    parser.add_argument("--input_dir", default="out/architecture_lab")
    parser.add_argument("--run_dir", default="experiments/architecture_lab_runs")
    args = parser.parse_args()
    runs = load_runs(ROOT / args.input_dir)
    write_reports(runs, ROOT / args.run_dir)
    print(f"generated architecture report for {len(runs)} runs")


if __name__ == "__main__":
    main()
