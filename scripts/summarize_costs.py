from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        row = json.load(f)
    hardware = row.get("hardware", {})
    return {
        "file": str(path),
        "run_name": row.get("run_name", ""),
        "gpu_name": hardware.get("gpu_name", "cpu"),
        "step": row.get("step", 0),
        "gpu_hours": row.get("gpu_hours", 0.0),
        "hourly_rate": row.get("hourly_rate", 0.0),
        "currency": row.get("currency", "CNY"),
        "estimated_cost": row.get("estimated_cost", 0.0),
        "max_memory_allocated_gb": row.get("max_memory_allocated_gb", 0.0),
        "max_memory_reserved_gb": row.get("max_memory_reserved_gb", 0.0),
        "model_params": row.get("model_params", 0),
        "activated_params_estimate": row.get("activated_params_estimate", 0),
        "estimated_train_tokens": row.get("estimated_train_tokens", 0),
        "estimated_training_flops": row.get("estimated_training_flops", 0),
        "val_loss": row.get("val_loss", ""),
    }


def write_markdown(rows: list[dict], out: Path) -> None:
    total_hours = sum(float(row["gpu_hours"]) for row in rows)
    total_cost = sum(float(row["estimated_cost"]) for row in rows)
    currency = rows[0]["currency"] if rows else "CNY"
    lines = [
        "# GPU Cost Summary",
        "",
        f"- Runs: {len(rows)}",
        f"- Total GPU hours: {total_hours:.4f}",
        f"- Total estimated cost: {total_cost:.2f} {currency}",
        "",
        "| Run | GPU | Steps | GPU hours | Cost | Peak allocated GB | Peak reserved GB | Val loss |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {gpu_name} | {step} | {gpu_hours:.4f} | {estimated_cost:.2f} {currency} | {max_memory_allocated_gb:.2f} | {max_memory_reserved_gb:.2f} | {val_loss} |".format(
                **row
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_name"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize TinySeek GPU cost summaries")
    parser.add_argument("--input_dir", default="out")
    parser.add_argument("--pattern", default="*_cost_summary.json")
    parser.add_argument("--markdown_out", default="out/cost_summary.md")
    parser.add_argument("--csv_out", default="out/cost_summary.csv")
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob(args.pattern))
    rows = [read_summary(path) for path in paths]
    write_markdown(rows, Path(args.markdown_out))
    write_csv(rows, Path(args.csv_out))
    print(f"loaded={len(rows)} markdown={args.markdown_out} csv={args.csv_out}")


if __name__ == "__main__":
    main()
