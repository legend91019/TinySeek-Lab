from __future__ import annotations

import json
import math
import sys
import tempfile
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row), encoding="utf-8")


def test_gpu_completion_report_combines_cost_and_reasoning_eval() -> None:
    from scripts.generate_gpu_completion_report import build_report

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "out"
        run_dir = root / "run"
        data_path = root / "data.jsonl"
        data_path.write_text('{"text":"one"}\n', encoding="utf-8")
        history_path = out_dir / "formal_base20m_history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text('{"step":10}\n', encoding="utf-8")
        write_json(
            out_dir / "formal_base20m_cost_summary.json",
            {
                "run_name": "formal_base20m",
                "step": 10,
                "val_lm_loss": 1.0,
                "training_tokens_per_second": 1000,
                "max_memory_allocated_gb": 2.0,
                "gpu_hours": 0.1,
                "estimated_cost": 0.218,
                "model_params": 100,
                "activated_params_estimate": 100,
                "data_path": str(data_path),
                "history_path": "/remote/formal_base20m_history.jsonl",
                "hourly_rate": 2.18,
                "currency": "CNY",
            },
        )
        write_json(
            run_dir / "eval_formal_base20m.json",
            {"reasoning": {"answer_accuracy": 0.2, "format_score": 0.4}, "perplexity": {"ppl": 3.0}},
        )
        write_json(
            out_dir / "formal_dense35_50m_cost_summary.json",
            {
                "run_name": "formal_dense35_50m",
                "step": 10,
                "val_lm_loss": 0.5,
                "training_tokens_per_second": 2000,
                "max_memory_allocated_gb": 3.0,
                "gpu_hours": 0.2,
                "estimated_cost": 0.436,
                "hourly_rate": 2.18,
                "currency": "CNY",
                "model_params": 200,
                "activated_params_estimate": 200,
                "data_path": str(data_path),
            },
        )
        write_json(
            out_dir / "formal_direct_grpo_cost_summary.json",
            {
                "run_name": "formal_direct_grpo",
                "stage": "grpo_mini",
                "step": 2,
                "max_memory_allocated_gb": 1.0,
                "gpu_hours": 0.01,
                "estimated_cost": 0.0218,
                "hourly_rate": 2.18,
                "currency": "CNY",
                "model_params": 100,
                "activated_params_estimate": 100,
                "data_path": str(data_path),
            },
        )
        build_report(out_dir, run_dir)
        assert (run_dir / "report.md").exists()
        assert (run_dir / "report_zh.md").exists()
        assert (run_dir / "results.csv").exists()
        assert (run_dir / "figures" / "posttraining_reasoning.svg").exists()
        assert (run_dir / "raw" / "formal_base20m_cost_summary.json").exists()
        assert (run_dir / "raw" / "formal_base20m_history.jsonl").exists()
        assert json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))["lines"] == 1
        assert "0.200" in (run_dir / "report.md").read_text(encoding="utf-8")
        assert "推理答案" in (run_dir / "report_zh.md").read_text(encoding="utf-8")
        with (run_dir / "results.csv").open(encoding="utf-8", newline="") as handle:
            rows = {row["run_name"]: row for row in csv.DictReader(handle)}
        assert math.isclose(float(rows["formal_dense35_50m"]["val_ppl"]), math.exp(0.5))
        assert rows["formal_dense35_50m"]["sample_ppl"] == ""
        assert float(rows["formal_base20m"]["sample_ppl"]) == 3.0
        assert "Measured Findings" in (run_dir / "report.md").read_text(encoding="utf-8")
        assert "实测结论" in (run_dir / "report_zh.md").read_text(encoding="utf-8")
        assert rows["formal_direct_grpo"]["val_loss"] == ""
        assert rows["formal_direct_grpo"]["tokens_per_second"] == ""
        report = (run_dir / "report.md").read_text(encoding="utf-8")
        assert "Environment and Data" in report
        assert "Reproduce" in report
        assert "excludes data preparation, standalone evaluation" in report
        assert "| N/A |" in report
        assert "dataset_manifests.json" in report
        data_path.unlink()
        build_report(run_dir / "raw", run_dir)
        manifests = json.loads((run_dir / "dataset_manifests.json").read_text(encoding="utf-8"))
        assert manifests[0]["path"] == str(data_path)
        fresh_dir = root / "fresh"
        build_report(run_dir / "raw", fresh_dir)
        assert (fresh_dir / "raw" / "formal_base20m_history.jsonl").exists()


if __name__ == "__main__":
    test_gpu_completion_report_combines_cost_and_reasoning_eval()
    print("GPU completion report tests ok")
