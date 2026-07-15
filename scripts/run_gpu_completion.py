from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, row: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def steps_for_tokens(token_budget: int, config: dict) -> int:
    train = config["train"]
    tokens_per_step = int(train["batch_size"]) * int(train.get("grad_accum_steps", 1)) * int(config["model"]["max_seq_len"])
    return math.ceil(token_budget / tokens_per_step)


def formalize(base_path: Path, run_name: str, token_budget: int, config_dir: Path, out_dir: Path, **train_overrides: object) -> dict:
    config = load(base_path)
    config["run_name"] = run_name
    config["train"].update(train_overrides)
    config["train"]["dtype"] = "bfloat16"
    config["train"]["out_dir"] = portable_path(out_dir)
    steps = steps_for_tokens(token_budget, config)
    config["train"]["max_steps"] = steps
    config["train"]["warmup_steps"] = max(20, min(200, steps // 20))
    config["train"]["eval_interval"] = max(100, steps // 10)
    config["train"]["save_interval"] = max(500, steps // 2)
    path = write(config_dir / f"{run_name}.json", config)
    return {"run_name": run_name, "token_budget": token_budget, "config_path": str(path)}


def build_pretrain_configs(config_dir: Path, out_dir: Path) -> list[dict]:
    rows = [
        formalize(ROOT / "configs/medium_dense_35m.json", "formal_dense35_50m", 50_000_000, config_dir, out_dir),
        formalize(ROOT / "configs/medium_dense_115m.json", "formal_dense115_30m", 30_000_000, config_dir, out_dir),
        formalize(ROOT / "configs/medium_moe_activated_35m.json", "formal_moe35_30m", 30_000_000, config_dir, out_dir),
    ]
    for batch_size in (16, 32):
        for learning_rate in (0.0003, 0.0006):
            suffix = f"bs{batch_size}_lr{learning_rate:.0e}".replace("e-0", "e-")
            rows.append(
                formalize(
                    ROOT / "configs/medium_dense_35m.json",
                    f"formal_sweep_{suffix}",
                    5_000_000,
                    config_dir,
                    out_dir,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            )
    rows.append(
        formalize(
            ROOT / "configs/tiny_sft.json",
            "formal_base20m",
            20_000_000,
            config_dir,
            out_dir,
            batch_size=32,
            grad_accum_steps=1,
            learning_rate=0.0006,
            weight_decay=0.1,
        )
    )
    return rows


def build_posttrain_configs(config_dir: Path, out_dir: Path) -> dict[str, Path]:
    pretrain = build_pretrain_configs(config_dir, out_dir)
    base_path = Path(next(row["config_path"] for row in pretrain if row["run_name"] == "formal_base20m"))
    base = load(base_path)
    sft = load(ROOT / "configs/tiny_sft.json")
    sft["run_name"] = "formal_reasoning_sft"
    sft["model"] = base["model"]
    sft["train"].update(
        {
            "batch_size": 16,
            "grad_accum_steps": 1,
            "max_steps": 2000,
            "eval_interval": 200,
            "save_interval": 500,
            "dtype": "bfloat16",
            "out_dir": portable_path(out_dir),
        }
    )
    grpo = load(ROOT / "configs/tiny_grpo.json")
    grpo["run_name"] = "formal_reasoning_grpo"
    grpo["model"] = base["model"]
    grpo["train"].update(
        {
            "max_steps": 300,
            "eval_interval": 25,
            "save_interval": 100,
            "dtype": "bfloat16",
            "out_dir": portable_path(out_dir),
        }
    )
    grpo["grpo"].update({"max_new_tokens": 64, "group_size": 4})
    direct_grpo = json.loads(json.dumps(grpo))
    direct_grpo["run_name"] = "formal_direct_grpo"
    return {
        "base": base_path,
        "sft": write(config_dir / "formal_reasoning_sft.json", sft),
        "grpo": write(config_dir / "formal_reasoning_grpo.json", grpo),
        "direct_grpo": write(config_dir / "formal_direct_grpo.json", direct_grpo),
    }


def complete(out_dir: Path, run_name: str, expected_steps: int) -> bool:
    path = out_dir / f"{run_name}_cost_summary.json"
    if not path.exists():
        return False
    try:
        return int(load(path).get("step", -1)) >= expected_steps
    except (OSError, json.JSONDecodeError):
        return False


def execute(command: list[str], dry_run: bool) -> None:
    print("run:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete TinySeek 4090 training, post-training, evaluation, and cost suite")
    parser.add_argument("--data", default="data/tinystories.jsonl")
    parser.add_argument("--run_dir", default="experiments/gpu_completion_runs")
    parser.add_argument("--out_dir", default="out/gpu_completion")
    parser.add_argument("--stages", default="pretrain,posttrain,eval,summary")
    parser.add_argument("--hourly_rate", type=float, default=2.18)
    parser.add_argument("--currency", default="CNY")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    stages = {part.strip() for part in args.stages.split(",") if part.strip()}
    run_dir = ROOT / args.run_dir
    config_dir = run_dir / "configs"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pretrain_rows = build_pretrain_configs(config_dir, out_dir)
    posttrain = build_posttrain_configs(config_dir, out_dir)

    if "pretrain" in stages or "posttrain" in stages:
        selected = pretrain_rows if "pretrain" in stages else [row for row in pretrain_rows if row["run_name"] == "formal_base20m"]
        for row in selected:
            config = load(Path(row["config_path"]))
            expected_steps = int(config["train"]["max_steps"])
            if not args.no_resume and complete(out_dir, row["run_name"], expected_steps):
                print(f"skip complete: {row['run_name']}")
                continue
            execute(
                [sys.executable, "trainer/train_pretrain.py", "--config", str(Path(row["config_path"]).relative_to(ROOT)), "--data", args.data, "--hourly_rate", str(args.hourly_rate), "--currency", args.currency],
                args.dry_run,
            )

    sft_data = "data/reasoning_sft.jsonl"
    grpo_data = "data/reasoning_grpo.jsonl"
    base_ckpt = out_dir / "formal_base20m_last.pt"
    sft_ckpt = out_dir / "formal_reasoning_sft_last.pt"
    grpo_ckpt = out_dir / "formal_reasoning_grpo_last.pt"
    direct_grpo_ckpt = out_dir / "formal_direct_grpo_last.pt"
    if "posttrain" in stages:
        execute([sys.executable, "scripts/prepare_reasoning_data.py", "--sft_out", sft_data, "--grpo_out", grpo_data], args.dry_run)
        direct_cfg = load(posttrain["direct_grpo"])
        if args.no_resume or not complete(out_dir, direct_cfg["run_name"], int(direct_cfg["train"]["max_steps"])):
            execute([sys.executable, "trainer/train_grpo.py", "--config", str(posttrain["direct_grpo"].relative_to(ROOT)), "--data", grpo_data, "--init_ckpt", str(base_ckpt.relative_to(ROOT)), "--hourly_rate", str(args.hourly_rate), "--currency", args.currency], args.dry_run)
        sft_cfg = load(posttrain["sft"])
        if args.no_resume or not complete(out_dir, sft_cfg["run_name"], int(sft_cfg["train"]["max_steps"])):
            execute([sys.executable, "trainer/train_sft.py", "--config", str(posttrain["sft"].relative_to(ROOT)), "--data", sft_data, "--init_ckpt", str(base_ckpt.relative_to(ROOT)), "--hourly_rate", str(args.hourly_rate), "--currency", args.currency], args.dry_run)
        grpo_cfg = load(posttrain["grpo"])
        if args.no_resume or not complete(out_dir, grpo_cfg["run_name"], int(grpo_cfg["train"]["max_steps"])):
            execute([sys.executable, "trainer/train_grpo.py", "--config", str(posttrain["grpo"].relative_to(ROOT)), "--data", grpo_data, "--init_ckpt", str(sft_ckpt.relative_to(ROOT)), "--hourly_rate", str(args.hourly_rate), "--currency", args.currency], args.dry_run)

    if "eval" in stages:
        eval_specs = [
            ("formal_base20m", posttrain["base"], base_ckpt),
            ("formal_direct_grpo", posttrain["direct_grpo"], direct_grpo_ckpt),
            ("formal_reasoning_sft", posttrain["sft"], sft_ckpt),
            ("formal_reasoning_grpo", posttrain["grpo"], grpo_ckpt),
        ]
        for name, config_path, checkpoint in eval_specs:
            output = run_dir / f"eval_{name}.json"
            if output.exists() and not args.no_resume:
                print(f"skip complete: eval_{name}")
                continue
            execute([sys.executable, "eval/mini_eval.py", "--config", str(config_path.relative_to(ROOT)), "--ckpt", str(checkpoint.relative_to(ROOT)), "--data", args.data, "--out", str(output.relative_to(ROOT)), "--max_new_tokens", "64"], args.dry_run)

    if "summary" in stages:
        execute([sys.executable, "scripts/summarize_costs.py", "--input_dir", str(out_dir.relative_to(ROOT)), "--markdown_out", str((run_dir / "cost_summary.md").relative_to(ROOT)), "--csv_out", str((run_dir / "cost_summary.csv").relative_to(ROOT))], args.dry_run)
        execute([sys.executable, "scripts/generate_gpu_completion_report.py", "--input_dir", str(out_dir.relative_to(ROOT)), "--run_dir", str(run_dir.relative_to(ROOT))], args.dry_run)


if __name__ == "__main__":
    main()
