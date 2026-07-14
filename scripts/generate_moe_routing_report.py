from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_expert_load(row: dict) -> dict | None:
    load = row.get("expert_load")
    if isinstance(load, dict) and load.get("total_counts"):
        return load
    return None


def collect_loads(input_dir: Path) -> list[tuple[str, dict, Path]]:
    rows = []
    for path in sorted(input_dir.glob("*_cost_summary.json")):
        row = load_json(path)
        load = find_expert_load(row)
        if load:
            rows.append((row.get("run_name", path.stem), load, path))
    return rows


def bar_svg(title: str, counts: list[int], out: Path) -> None:
    width = 820
    height = 300
    left = 56
    right = 32
    top = 56
    bottom = 46
    max_count = max(counts or [1])
    n = max(1, len(counts))
    gap = 10
    bar_w = max(14, (width - left - right - gap * (n - 1)) / n)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2933">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#9fb3c8"/>',
    ]
    palette = ["#2f6f9f", "#d95f59", "#5f8f3f", "#9b6fb3", "#d18b2c", "#4c7c7b"]
    for idx, count in enumerate(counts):
        x = left + idx * (bar_w + gap)
        bar_h = 0 if max_count <= 0 else (height - top - bottom - 20) * count / max_count
        y = height - bottom - bar_h
        color = palette[idx % len(palette)]
        lines.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="3" fill="{color}"/>',
                f'<text x="{x + bar_w / 2 - 9:.1f}" y="{height - bottom + 20}" font-family="Arial, sans-serif" font-size="12" fill="#334e68">E{idx}</text>',
                f'<text x="{x + bar_w / 2 - 14:.1f}" y="{y - 7:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#102a43">{count}</text>',
            ]
        )
    lines.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(loads: list[tuple[str, dict, Path]], out: Path, figures_dir: Path) -> None:
    lines = [
        "# MoE Routing Report",
        "",
        "This report is generated from `expert_load` snapshots in `*_cost_summary.json`.",
        "The snapshot is lightweight: it records the latest observed expert assignment counts, not a full training trace.",
        "",
    ]
    if not loads:
        lines.extend(
            [
                "No expert-load data found yet.",
                "",
                "Run a MoE config with the current trainer, then regenerate this report:",
                "",
                "```bash",
                "python scripts/generate_moe_routing_report.py --input_dir out --out experiments/moe_routing_report.md",
                "```",
            ]
        )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(["| Run | Experts | Top-k | Max fraction | Min fraction | Source |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for run_name, load, source in loads:
        counts = [int(v) for v in load["total_counts"]]
        total = max(1, sum(counts))
        fractions = [v / total for v in counts]
        figure = figures_dir / f"{run_name}_expert_load.svg"
        bar_svg(f"{run_name} expert load", counts, figure)
        rel_figure = figure.relative_to(out.parent).as_posix()
        lines.append(
            f"| `{run_name}` | {len(counts)} | {load.get('top_k', 'N/A')} | {max(fractions):.3f} | {min(fractions):.3f} | `{source}` |"
        )
        lines.extend(["", f"![{run_name} expert load]({rel_figure})", ""])

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MoE expert-load report from TinySeek cost summaries")
    parser.add_argument("--input_dir", default="out")
    parser.add_argument("--out", default="experiments/moe_routing_report.md")
    parser.add_argument("--figures_dir", default="experiments/moe_figures")
    args = parser.parse_args()

    out = Path(args.out)
    figures_dir = Path(args.figures_dir)
    loads = collect_loads(Path(args.input_dir))
    write_report(loads, out, figures_dir)
    print(f"moe_runs={len(loads)} report={out}")


if __name__ == "__main__":
    main()
