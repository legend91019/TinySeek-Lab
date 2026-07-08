from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PlannedCommand:
    name: str
    argv: list[str]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def deep_update(row: dict, dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cur = row
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def steps_for_tokens(token_budget: int, batch_size: int, grad_accum_steps: int, max_seq_len: int) -> int:
    tokens_per_step = max(1, batch_size * grad_accum_steps * max_seq_len)
    return max(1, math.ceil(token_budget / tokens_per_step))


def py_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def train_pretrain_cmd(config: str, data: str, run_name: str, hourly_rate: float, currency: str, max_steps: int | None = None) -> list[str]:
    cmd = py_cmd(
        "trainer/train_pretrain.py",
        "--config",
        config,
        "--data",
        data,
        "--run_name",
        run_name,
        "--hourly_rate",
        str(hourly_rate),
        "--currency",
        currency,
    )
    if max_steps is not None:
        cmd.extend(["--max_steps", str(max_steps)])
    return cmd


def mini_eval_cmd(config: str, ckpt: str, data: str, out: str) -> list[str]:
    return py_cmd("eval/mini_eval.py", "--config", config, "--ckpt", ckpt, "--data", data, "--out", out)


def add_prepare_data(plan: list[PlannedCommand], args: argparse.Namespace) -> None:
    if args.skip_data_prepare:
        return
    cmd = py_cmd(
        "scripts/prepare_hf_dataset.py",
        "--dataset_name",
        args.dataset_name,
        "--split",
        args.split,
        "--text_field",
        args.text_field,
        "--max_samples",
        str(args.max_samples),
        "--min_chars",
        str(args.min_chars),
        "--out",
        args.data,
    )
    if args.dataset_config:
        cmd.extend(["--dataset_config", args.dataset_config])
    if args.streaming:
        cmd.append("--streaming")
    plan.append(PlannedCommand("prepare_pretrain_data", cmd))


def add_pretrain_run(plan: list[PlannedCommand], config: str, run_name: str, args: argparse.Namespace, max_steps: int | None = None) -> None:
    plan.append(PlannedCommand(f"train_{run_name}", train_pretrain_cmd(config, args.data, run_name, args.hourly_rate, args.currency, max_steps=max_steps)))
    plan.append(PlannedCommand(f"eval_{run_name}", mini_eval_cmd(config, f"out/{run_name}_last.pt", args.data, str(Path(args.run_dir) / f"eval_{run_name}.json"))))


def add_sweep(plan: list[PlannedCommand], args: argparse.Namespace) -> None:
    base = load_json(ROOT / "configs/medium_dense_35m.json")
    sweep_specs = [
        ("v1_sweep_bs16_lr3e-4", 16, 0.0003),
        ("v1_sweep_bs16_lr6e-4", 16, 0.0006),
        ("v1_sweep_bs32_lr3e-4", 32, 0.0003),
        ("v1_sweep_bs32_lr6e-4", 32, 0.0006),
    ]
    for run_name, batch_size, learning_rate in sweep_specs:
        cfg = json.loads(json.dumps(base))
        cfg["run_name"] = run_name
        deep_update(cfg, "train.batch_size", batch_size)
        deep_update(cfg, "train.learning_rate", learning_rate)
        steps = steps_for_tokens(args.sweep_tokens, batch_size, int(cfg["train"].get("grad_accum_steps", 1)), int(cfg["model"]["max_seq_len"]))
        deep_update(cfg, "train.max_steps", steps)
        deep_update(cfg, "train.eval_interval", max(25, steps // 4))
        deep_update(cfg, "train.save_interval", max(25, steps // 2))
        cfg_path = Path(args.run_dir) / "configs" / f"{run_name}.json"
        if args.execute or args.write_manifest:
            save_json(ROOT / cfg_path, cfg)
        add_pretrain_run(plan, str(cfg_path), run_name, args, max_steps=steps)


def build_plan(args: argparse.Namespace) -> list[PlannedCommand]:
    stages = set(part.strip() for part in args.stages.split(",") if part.strip())
    if "all" in stages:
        stages = {"data", "tiny_base", "dense", "sweep", "moe", "mla", "posttrain", "summary"}

    plan: list[PlannedCommand] = []
    if "data" in stages:
        add_prepare_data(plan, args)

    if "tiny_base" in stages or "posttrain" in stages or "mla" in stages:
        add_pretrain_run(plan, "configs/tiny_dense.json", "v1_tiny_base", args, max_steps=args.tiny_base_steps)

    if "dense" in stages:
        add_pretrain_run(plan, "configs/medium_dense_35m.json", "v1_dense35", args)
        add_pretrain_run(plan, "configs/medium_dense_115m.json", "v1_dense115", args)

    if "sweep" in stages:
        add_sweep(plan, args)

    if "moe" in stages:
        add_pretrain_run(plan, "configs/medium_moe_activated_35m.json", "v1_moe_activated35", args)

    if "mla" in stages:
        add_pretrain_run(plan, "configs/tiny_mla.json", "v1_tiny_mla", args, max_steps=args.mla_steps)

    if "posttrain" in stages:
        sft_data = "data/v1_toy_sft.jsonl"
        grpo_data = "data/v1_toy_grpo.jsonl"
        plan.append(PlannedCommand("prepare_sft_data", py_cmd("scripts/prepare_toy_sft_data.py", "--out", sft_data)))
        plan.append(
            PlannedCommand(
                "train_v1_tiny_sft",
                py_cmd(
                    "trainer/train_sft.py",
                    "--config",
                    "configs/tiny_sft.json",
                    "--data",
                    sft_data,
                    "--init_ckpt",
                    "out/v1_tiny_base_last.pt",
                    "--run_name",
                    "v1_tiny_sft",
                    "--hourly_rate",
                    str(args.hourly_rate),
                    "--currency",
                    args.currency,
                ),
            )
        )
        plan.append(PlannedCommand("eval_v1_tiny_sft", mini_eval_cmd("configs/tiny_sft.json", "out/v1_tiny_sft_last.pt", args.data, str(Path(args.run_dir) / "eval_v1_tiny_sft.json"))))
        plan.append(PlannedCommand("prepare_grpo_data", py_cmd("scripts/prepare_toy_grpo_data.py", "--out", grpo_data)))
        plan.append(
            PlannedCommand(
                "train_v1_tiny_grpo",
                py_cmd(
                    "trainer/train_grpo.py",
                    "--config",
                    "configs/tiny_grpo.json",
                    "--data",
                    grpo_data,
                    "--init_ckpt",
                    "out/v1_tiny_sft_last.pt",
                    "--run_name",
                    "v1_tiny_grpo",
                    "--hourly_rate",
                    str(args.hourly_rate),
                    "--currency",
                    args.currency,
                ),
            )
        )
        plan.append(PlannedCommand("eval_v1_tiny_grpo", mini_eval_cmd("configs/tiny_grpo.json", "out/v1_tiny_grpo_last.pt", args.data, str(Path(args.run_dir) / "eval_v1_tiny_grpo.json"))))

    if "summary" in stages:
        plan.append(
            PlannedCommand(
                "summarize_costs",
                py_cmd(
                    "scripts/summarize_costs.py",
                    "--input_dir",
                    "out",
                    "--markdown_out",
                    str(Path(args.run_dir) / "cost_summary.md"),
                    "--csv_out",
                    str(Path(args.run_dir) / "cost_summary.csv"),
                ),
            )
        )
    return plan


def write_manifest(plan: list[PlannedCommand], args: argparse.Namespace) -> None:
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RTX 4090 v1 Run Manifest",
        "",
        f"- Execute mode: {args.execute}",
        f"- Dataset path: `{args.data}`",
        f"- Hourly rate: {args.hourly_rate} {args.currency}/hour",
        f"- Sweep token target per run: {args.sweep_tokens}",
        "",
        "| Step | Name | Command |",
        "| ---: | --- | --- |",
    ]
    for idx, item in enumerate(plan, start=1):
        lines.append(f"| {idx} | `{item.name}` | `{' '.join(item.argv)}` |")
    (run_dir / "COMMANDS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute the TinySeek-Lab RTX 4090 v1 experiment path")
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Without this flag the script only prints the plan.")
    parser.add_argument("--write_manifest", action="store_true", help="Write COMMANDS.md and generated sweep configs during a dry run.")
    parser.add_argument("--stages", default="all", help="Comma-separated stages: all,data,tiny_base,dense,sweep,moe,mla,posttrain,summary")
    parser.add_argument("--run_dir", default="experiments/v1_4090_plan")
    parser.add_argument("--data", default="data/tinystories.jsonl")
    parser.add_argument("--dataset_name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--max_samples", type=int, default=50000)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--skip_data_prepare", action="store_true")
    parser.add_argument("--hourly_rate", type=float, default=2.18)
    parser.add_argument("--currency", default="CNY")
    parser.add_argument("--sweep_tokens", type=int, default=5_000_000)
    parser.add_argument("--tiny_base_steps", type=int, default=200)
    parser.add_argument("--mla_steps", type=int, default=200)
    args = parser.parse_args()

    plan = build_plan(args)
    if args.execute or args.write_manifest:
        write_manifest(plan, args)
        manifest = f"{args.run_dir}/COMMANDS.md"
    else:
        manifest = "not written"
    print(f"planned_steps={len(plan)} manifest={manifest} execute={args.execute}")

    for idx, item in enumerate(plan, start=1):
        printable = " ".join(item.argv)
        print(f"\n[{idx}/{len(plan)}] {item.name}\n{printable}")
        if args.execute:
            subprocess.run(item.argv, cwd=ROOT, check=True)

    if not args.execute:
        print("\nDry run only. Add --execute on the 4090 machine to start training.")
        print("Add --write_manifest if you want COMMANDS.md and generated sweep configs written during dry run.")


if __name__ == "__main__":
    main()
