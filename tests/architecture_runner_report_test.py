from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row), encoding="utf-8")


def test_materialize_and_resume_contract() -> None:
    from scripts.run_architecture_lab import is_complete, materialize_config

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = {
            "run_name": "arch_dense_mha",
            "model": {"vocab_size": 260},
            "train": {"seed": 42, "out_dir": "out", "max_steps": 1000},
        }
        base_path = root / "dense_mha.json"
        write_json(base_path, base)
        generated = materialize_config(base_path, 43, root / "configs", root / "out")
        row = json.loads(generated.read_text(encoding="utf-8"))
        assert row["run_name"] == "arch_dense_mha_seed43"
        assert row["train"]["seed"] == 43
        assert row["train"]["out_dir"] == str(root / "out")
        assert base == json.loads(base_path.read_text(encoding="utf-8"))
        assert not is_complete(root / "out", row["run_name"])
        write_json(root / "out" / f"{row['run_name']}_cost_summary.json", {"step": 999})
        assert not is_complete(root / "out", row["run_name"], expected_steps=1000)
        write_json(root / "out" / f"{row['run_name']}_cost_summary.json", {"step": 1000})
        assert is_complete(root / "out", row["run_name"], expected_steps=1000)


def test_report_aggregates_main_lm_loss() -> None:
    from scripts.generate_architecture_report import aggregate_runs, history_expert_load_cv, load_runs, write_reports

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "out"
        run_dir = root / "report"
        data_path = root / "data.jsonl"
        data_path.write_text('{"text":"one"}\n{"text":"two"}\n', encoding="utf-8")
        for seed, loss, tokens_per_second in ((42, 1.0, 1000.0), (43, 1.2, 1200.0)):
            adjacent_history = out_dir / f"arch_dense_mha_seed{seed}_history.jsonl"
            adjacent_history.parent.mkdir(parents=True, exist_ok=True)
            adjacent_history.write_text(
                json.dumps({"expert_load": {"total_fractions": [0.75, 0.25]}}) + "\n",
                encoding="utf-8",
            )
            write_json(
                out_dir / f"arch_dense_mha_seed{seed}_cost_summary.json",
                {
                    "run_name": f"arch_dense_mha_seed{seed}",
                    "seed": seed,
                    "step": 1000,
                    "val_lm_loss": loss,
                    "val_objective": loss + 0.5,
                    "training_tokens_per_second": tokens_per_second,
                    "max_memory_allocated_gb": 2.0 + seed / 100,
                    "gpu_hours": 0.1,
                    "estimated_cost": 0.218,
                    "hourly_rate": 2.18,
                    "currency": "CNY",
                    "model_params": 100,
                    "activated_params_estimate": 100,
                    "expert_load": {},
                    "history_path": f"/remote/out/arch_dense_mha_seed{seed}_history.jsonl",
                    "data_path": str(data_path),
                },
            )
        runs = load_runs(out_dir)
        aggregate = aggregate_runs(runs)[0]
        assert aggregate["run_name"] == "arch_dense_mha"
        assert aggregate["seeds"] == 2
        assert math.isclose(aggregate["val_lm_loss_mean"], 1.1)
        assert math.isclose(aggregate["ppl_mean"], (math.exp(1.0) + math.exp(1.2)) / 2)
        assert math.isclose(aggregate["tokens_per_second_mean"], 1100.0)
        assert math.isclose(aggregate["expert_load_cv_mean"], 0.5)
        write_reports(runs, run_dir)
        assert (run_dir / "results.csv").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "report_zh.md").exists()
        assert (run_dir / "figures" / "architecture_ppl.svg").exists()
        assert (run_dir / "raw" / "arch_dense_mha_seed42_cost_summary.json").exists()
        assert json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))["lines"] == 2
        assert "GPU measurements" in (run_dir / "report.md").read_text(encoding="utf-8")
        assert "Tracked trainer-process time" in (run_dir / "report.md").read_text(encoding="utf-8")
        assert "GPU 实测" in (run_dir / "report_zh.md").read_text(encoding="utf-8")
        assert "Decision Summary" in (run_dir / "report.md").read_text(encoding="utf-8")
        assert "决策摘要" in (run_dir / "report_zh.md").read_text(encoding="utf-8")
        assert "Environment and Data" in (run_dir / "report.md").read_text(encoding="utf-8")
        assert "Reproduce" in (run_dir / "report.md").read_text(encoding="utf-8")
        data_path.unlink()
        rerun = load_runs(run_dir / "raw")
        write_reports(rerun, run_dir)
        assert math.isclose(aggregate_runs(rerun)[0]["expert_load_cv_mean"], 0.5)
        assert "SHA256" in (run_dir / "report.md").read_text(encoding="utf-8")

        history = root / "history.jsonl"
        history.write_text(
            '\n'.join(
                [
                    json.dumps({"expert_load": {"total_fractions": [0.5, 0.5]}}),
                    json.dumps({"expert_load": {"total_fractions": [0.75, 0.25]}}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        assert math.isclose(history_expert_load_cv(history, {}), 0.25)


def test_generated_configs_keep_repo_output_paths_portable() -> None:
    from scripts.run_architecture_lab import materialize_config

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_path = root / "dense.json"
        write_json(
            base_path,
            {"run_name": "arch_dense", "model": {}, "train": {"seed": 42, "max_steps": 1}},
        )
        generated = materialize_config(base_path, 42, root / "configs", ROOT / "out" / "architecture_lab")
        row = json.loads(generated.read_text(encoding="utf-8"))
        assert row["train"]["out_dir"] == "out/architecture_lab"


def test_mtp_decision_stays_conditional_on_the_tested_branch() -> None:
    from scripts.generate_architecture_report import decision_summary

    rows = [
        {
            "run_name": "arch_v3_no_mtp",
            "ppl_mean": 2.2,
            "ppl_std": 0.02,
            "peak_vram_gb_mean": 0.2,
        },
        {
            "run_name": "arch_v3_mtp",
            "ppl_mean": 2.19,
            "ppl_std": 0.03,
            "peak_vram_gb_mean": 0.25,
        },
    ]
    en = "\n".join(decision_summary(rows, zh=False))
    zh = "\n".join(decision_summary(rows, zh=True))
    assert "rejected V3-style branch" in en
    assert "已被拒绝的 V3-style 分支" in zh


def test_billing_metadata_rejects_mixed_ledgers() -> None:
    from scripts.generate_architecture_report import billing_metadata

    try:
        billing_metadata(
            [
                {"_hourly_rate": 2.18, "_currency": "CNY"},
                {"_hourly_rate": 1.0, "_currency": "USD"},
            ]
        )
    except ValueError as exc:
        assert "Mixed billing metadata" in str(exc)
    else:
        raise AssertionError("mixed billing metadata must be rejected")

    try:
        billing_metadata(
            [
                {"_hourly_rate": 2.18, "_currency": "CNY"},
                {"_hourly_rate": None, "_currency": None},
            ]
        )
    except ValueError as exc:
        assert "Incomplete billing metadata" in str(exc)
    else:
        raise AssertionError("incomplete billing metadata must be rejected")


if __name__ == "__main__":
    test_materialize_and_resume_contract()
    test_report_aggregates_main_lm_loss()
    test_generated_configs_keep_repo_output_paths_portable()
    test_mtp_decision_stays_conditional_on_the_tested_branch()
    test_billing_metadata_rejects_mixed_ledgers()
    print("architecture runner/report tests ok")
