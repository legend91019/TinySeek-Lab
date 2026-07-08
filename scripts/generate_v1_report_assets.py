from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


RUN_ORDER = [
    "v1_tiny_base",
    "v1_dense35",
    "v1_dense115",
    "v1_sweep_bs16_lr3e-4",
    "v1_sweep_bs16_lr6e-4",
    "v1_sweep_bs32_lr3e-4",
    "v1_sweep_bs32_lr6e-4",
    "v1_moe_activated35",
    "v1_tiny_mla",
    "v1_tiny_sft",
    "v1_tiny_grpo",
]


def read_costs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    order = {name: idx for idx, name in enumerate(RUN_ORDER)}
    return sorted(rows, key=lambda row: order.get(row["run_name"], len(order)))


def read_evals(run_dir: Path) -> dict[str, dict]:
    evals = {}
    for path in sorted(run_dir.glob("eval_*.json")):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        run_name = path.stem.removeprefix("eval_")
        evals[run_name] = row
    return evals


def to_float(value: object, default: float = 0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_millions(value: object) -> str:
    return f"{to_float(value) / 1_000_000:.2f}M"


def merged_rows(cost_rows: list[dict], evals: dict[str, dict]) -> list[dict]:
    rows = []
    for cost in cost_rows:
        run_name = cost["run_name"]
        ev = evals.get(run_name, {})
        ppl = ev.get("perplexity", {}).get("ppl", "")
        eval_loss = ev.get("perplexity", {}).get("loss", "")
        add_acc = ev.get("addition", {}).get("accuracy", "")
        fmt_score = ev.get("format", {}).get("format_score", "")
        row = dict(cost)
        row.update(
            {
                "eval_loss": eval_loss,
                "ppl": ppl,
                "addition_accuracy": add_acc,
                "format_score": fmt_score,
            }
        )
        rows.append(row)
    return rows


def bar_svg(title: str, rows: list[tuple[str, float]], out: Path, *, unit: str = "", lower_is_better: bool = False) -> None:
    width = 980
    left = 210
    top = 64
    bar_h = 24
    gap = 12
    right = 44
    max_value = max([value for _, value in rows] or [1.0])
    height = top + len(rows) * (bar_h + gap) + 52
    palette = ["#2f6f9f", "#d95f59", "#5f8f3f", "#9b6fb3", "#d18b2c", "#4c7c7b"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2933">{html.escape(title)}</text>',
    ]
    if lower_is_better:
        lines.append('<text x="24" y="54" font-family="Arial, sans-serif" font-size="12" fill="#52606d">Lower is better</text>')
    for idx, (name, value) in enumerate(rows):
        y = top + idx * (bar_h + gap)
        bar_w = 0 if max_value <= 0 else (width - left - right - 90) * value / max_value
        color = palette[idx % len(palette)]
        lines.extend(
            [
                f'<text x="24" y="{y + 17}" font-family="Arial, sans-serif" font-size="13" fill="#243b53">{html.escape(name)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{color}"/>',
                f'<text x="{left + bar_w + 8:.1f}" y="{y + 17}" font-family="Arial, sans-serif" font-size="13" fill="#102a43">{value:.4g}{html.escape(unit)}</text>',
            ]
        )
    lines.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scatter_svg(title: str, rows: list[tuple[str, float, float]], out: Path) -> None:
    width = 760
    height = 440
    left = 72
    right = 36
    top = 58
    bottom = 70
    xs = [x for _, x, _ in rows]
    ys = [y for _, _, y in rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x *= 0.9
        max_x *= 1.1
    if min_y == max_y:
        min_y *= 0.9
        max_y *= 1.1

    def sx(x: float) -> float:
        return left + (x - min_x) / (max_x - min_x) * (width - left - right)

    def sy(y: float) -> float:
        return height - bottom - (y - min_y) / (max_y - min_y) * (height - top - bottom)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2933">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9fb3c8"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#9fb3c8"/>',
        f'<text x="{width / 2 - 70}" y="{height - 22}" font-family="Arial, sans-serif" font-size="13" fill="#334e68">Peak allocated VRAM (GB)</text>',
        f'<text x="18" y="{top - 12}" font-family="Arial, sans-serif" font-size="13" fill="#334e68">PPL</text>',
    ]
    for idx in range(5):
        x = min_x + (max_x - min_x) * idx / 4
        px = sx(x)
        lines.extend(
            [
                f'<line x1="{px:.1f}" y1="{height - bottom}" x2="{px:.1f}" y2="{height - bottom + 5}" stroke="#9fb3c8"/>',
                f'<text x="{px - 14:.1f}" y="{height - bottom + 22}" font-family="Arial, sans-serif" font-size="11" fill="#52606d">{x:.1f}</text>',
            ]
        )
    for idx in range(5):
        y = min_y + (max_y - min_y) * idx / 4
        py = sy(y)
        lines.extend(
            [
                f'<line x1="{left - 5}" y1="{py:.1f}" x2="{left}" y2="{py:.1f}" stroke="#9fb3c8"/>',
                f'<text x="28" y="{py + 4:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#52606d">{y:.1f}</text>',
            ]
        )
    palette = ["#2f6f9f", "#d95f59", "#5f8f3f", "#9b6fb3", "#d18b2c", "#4c7c7b"]
    for idx, (name, x, y) in enumerate(rows):
        px = sx(x)
        py = sy(y)
        color = palette[idx % len(palette)]
        lines.extend(
            [
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" fill="{color}"/>',
                f'<text x="{px + 9:.1f}" y="{py - 8:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#102a43">{html.escape(name)}</text>',
            ]
        )
    lines.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(rows: list[dict], out: Path, *, zh: bool) -> None:
    title = "v1 自动汇总表" if zh else "v1 Auto Summary Tables"
    note = (
        "这些表由 `scripts/generate_v1_report_assets.py` 从成本 CSV 和 mini-eval JSON 自动生成。"
        if zh
        else "These tables are generated by `scripts/generate_v1_report_assets.py` from the cost CSV and mini-eval JSON files."
    )
    metric_header = (
        "| Run | Steps | Params | Activated | Peak GB | GPU h | Cost CNY | Val loss | Eval loss | PPL | Add acc |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [f"# {title}", "", note, "", "## Core Metrics", "", metric_header]
    for row in rows:
        lines.append(
            "| `{run}` | {steps} | {params} | {activated} | {peak:.2f} | {hours:.4f} | {cost:.4f} | {val} | {eval_loss} | {ppl} | {add} |".format(
                run=row["run_name"],
                steps=row["step"],
                params=fmt_millions(row["model_params"]),
                activated=fmt_millions(row["activated_params_estimate"]),
                peak=to_float(row["max_memory_allocated_gb"]),
                hours=to_float(row["gpu_hours"]),
                cost=to_float(row["estimated_cost"]),
                val=f"{to_float(row['val_loss']):.4f}" if row.get("val_loss") else "N/A",
                eval_loss=f"{to_float(row['eval_loss']):.4f}" if row.get("eval_loss") != "" else "N/A",
                ppl=f"{to_float(row['ppl']):.4f}" if row.get("ppl") != "" else "N/A",
                add=f"{to_float(row['addition_accuracy']):.2f}" if row.get("addition_accuracy") != "" else "N/A",
            )
        )
    lines.extend(
        [
            "",
            "## Generated Figures" if not zh else "## 自动生成图表",
            "",
            "![PPL comparison](figures/v1_ppl.svg)",
            "",
            "![Peak VRAM comparison](figures/v1_peak_vram.svg)",
            "",
            "![GPU cost comparison](figures/v1_cost.svg)",
            "",
            "![Sweep comparison](figures/v1_sweep_val_loss.svg)",
            "",
            "![VRAM versus PPL](figures/v1_vram_vs_ppl.svg)",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TinySeek-Lab v1 report tables and SVG figures")
    parser.add_argument("--run_dir", default="experiments/v1_4090_plan")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = merged_rows(read_costs(run_dir / "cost_summary.csv"), read_evals(run_dir))
    figures = run_dir / "figures"

    ppl_rows = [(row["run_name"], to_float(row["ppl"])) for row in rows if row.get("ppl") != ""]
    vram_rows = [(row["run_name"], to_float(row["max_memory_allocated_gb"])) for row in rows]
    cost_rows = [(row["run_name"], to_float(row["estimated_cost"])) for row in rows]
    sweep_rows = [(row["run_name"].replace("v1_sweep_", ""), to_float(row["val_loss"])) for row in rows if row["run_name"].startswith("v1_sweep")]
    scatter_rows = [(row["run_name"], to_float(row["max_memory_allocated_gb"]), to_float(row["ppl"])) for row in rows if row.get("ppl") != ""]

    bar_svg("Mini-eval perplexity by run", ppl_rows, figures / "v1_ppl.svg", lower_is_better=True)
    bar_svg("Peak allocated VRAM by run", vram_rows, figures / "v1_peak_vram.svg", unit=" GB")
    bar_svg("Estimated GPU cost by run", cost_rows, figures / "v1_cost.svg", unit=" CNY")
    bar_svg("LR / batch sweep validation loss", sweep_rows, figures / "v1_sweep_val_loss.svg", lower_is_better=True)
    scatter_svg("VRAM vs mini-eval PPL", scatter_rows, figures / "v1_vram_vs_ppl.svg")

    write_table(rows, run_dir / "auto_summary.md", zh=False)
    write_table(rows, run_dir / "auto_summary_zh.md", zh=True)
    print(f"generated figures and tables in {run_dir}")


if __name__ == "__main__":
    main()
