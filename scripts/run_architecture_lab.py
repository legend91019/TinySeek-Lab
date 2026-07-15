from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def materialize_config(base_path: Path, seed: int, config_dir: Path, out_dir: Path) -> Path:
    row = json.loads(base_path.read_text(encoding="utf-8"))
    row["run_name"] = f"{row['run_name']}_seed{seed}"
    row["train"]["seed"] = seed
    row["train"]["out_dir"] = portable_path(out_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{base_path.stem}_seed{seed}.json"
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def is_complete(out_dir: Path, run_name: str, expected_steps: int | None = None) -> bool:
    path = out_dir / f"{run_name}_cost_summary.json"
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return expected_steps is None or int(row.get("step", -1)) >= expected_steps


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matched TinySeek architecture lab with resumable multi-seed configs")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--config_dir", default="configs/architecture_lab")
    parser.add_argument("--run_dir", default="experiments/architecture_lab_runs")
    parser.add_argument("--out_dir", default="out/architecture_lab")
    parser.add_argument("--only", default="", help="Optional comma-separated config stems")
    parser.add_argument("--hourly_rate", type=float, default=2.18)
    parser.add_argument("--currency", default="CNY")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    selected = {part.strip() for part in args.only.split(",") if part.strip()}
    base_paths = sorted((ROOT / args.config_dir).glob("*.json"))
    if selected:
        base_paths = [path for path in base_paths if path.stem in selected]
    if not base_paths:
        raise SystemExit("No architecture configs selected")

    run_dir = ROOT / args.run_dir
    generated_dir = run_dir / "configs"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    planned = len(base_paths) * len(seeds)

    for seed in seeds:
        for base_path in base_paths:
            generated = materialize_config(base_path, seed, generated_dir, out_dir)
            row = json.loads(generated.read_text(encoding="utf-8"))
            run_name = row["run_name"]
            expected_steps = int(row["train"]["max_steps"])
            if not args.no_resume and is_complete(out_dir, run_name, expected_steps):
                completed += 1
                print(f"skip complete: {run_name}")
                continue
            command = [
                sys.executable,
                "trainer/train_pretrain.py",
                "--config",
                str(generated.relative_to(ROOT)),
                "--data",
                args.data,
                "--hourly_rate",
                str(args.hourly_rate),
                "--currency",
                args.currency,
            ]
            print("run:", " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
                completed += 1

    print(f"architecture lab complete={completed}/{planned} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
