from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trainer.utils import deep_update, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pretraining hyperparameter sweep")
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--hourly_rate", type=float, default=0.0, help="GPU rental price per hour, for example AutoDL RTX 4090=2.18 or RTX 3080 Ti=0.98.")
    parser.add_argument("--currency", default="CNY")
    args = parser.parse_args()

    sweep = load_config(args.sweep)
    base = load_config(sweep["base_config"])
    results = []
    for run in sweep["runs"]:
        cfg = copy.deepcopy(base)
        cfg["run_name"] = run["name"]
        for key, value in run.items():
            if key == "name":
                continue
            deep_update(cfg, key, value)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            tmp_cfg = f.name
        cmd = [
            sys.executable,
            "trainer/train_pretrain.py",
            "--config",
            tmp_cfg,
            "--data",
            sweep["data"],
            "--max_steps",
            str(sweep.get("max_steps", cfg["train"]["max_steps"])),
            "--run_name",
            run["name"],
            "--hourly_rate",
            str(args.hourly_rate),
            "--currency",
            args.currency,
        ]
        print("Running:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        results.append({"name": run["name"], "returncode": completed.returncode})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
