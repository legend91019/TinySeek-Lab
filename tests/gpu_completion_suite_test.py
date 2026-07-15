from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def test_completion_suite_materializes_declared_token_budgets() -> None:
    from scripts.run_gpu_completion import build_pretrain_configs

    with tempfile.TemporaryDirectory() as tmp:
        config_dir = Path(tmp) / "configs"
        out_dir = Path(tmp) / "out"
        rows = build_pretrain_configs(config_dir, out_dir)
        assert len(rows) == 8
        by_name = {row["run_name"]: row for row in rows}
        assert {"formal_dense35_50m", "formal_dense115_30m", "formal_moe35_30m", "formal_base20m"} <= set(by_name)
        for row in rows:
            config = json.loads(Path(row["config_path"]).read_text(encoding="utf-8"))
            tokens_per_step = (
                config["train"]["batch_size"]
                * config["train"].get("grad_accum_steps", 1)
                * config["model"]["max_seq_len"]
            )
            actual = tokens_per_step * config["train"]["max_steps"]
            assert actual >= row["token_budget"]
            assert actual < row["token_budget"] + tokens_per_step
            assert config["train"]["dtype"] == "bfloat16"
            assert config["train"]["eval_interval"] >= 100


def test_posttrain_configs_share_the_base_architecture() -> None:
    from scripts.run_gpu_completion import build_posttrain_configs

    with tempfile.TemporaryDirectory() as tmp:
        rows = build_posttrain_configs(Path(tmp) / "configs", Path(tmp) / "out")
        base = json.loads(Path(rows["base"]).read_text(encoding="utf-8"))
        sft = json.loads(Path(rows["sft"]).read_text(encoding="utf-8"))
        grpo = json.loads(Path(rows["grpo"]).read_text(encoding="utf-8"))
        direct_grpo = json.loads(Path(rows["direct_grpo"]).read_text(encoding="utf-8"))
        assert base["model"] == sft["model"] == grpo["model"] == direct_grpo["model"]
        assert sft["train"]["max_steps"] == 2000
        assert grpo["train"]["max_steps"] == 300
        assert direct_grpo["run_name"] == "formal_direct_grpo"
        assert grpo["grpo"]["max_new_tokens"] == 64


def test_completion_suite_generates_the_final_report() -> None:
    source = (ROOT / "scripts" / "run_gpu_completion.py").read_text(encoding="utf-8")
    assert '"scripts/generate_gpu_completion_report.py"' in source


def test_formal_configs_keep_repo_output_paths_portable() -> None:
    from scripts.run_gpu_completion import build_pretrain_configs

    with tempfile.TemporaryDirectory() as tmp:
        rows = build_pretrain_configs(Path(tmp) / "configs", ROOT / "out" / "gpu_completion")
        config = json.loads(Path(rows[0]["config_path"]).read_text(encoding="utf-8"))
        assert config["train"]["out_dir"] == "out/gpu_completion"


if __name__ == "__main__":
    test_completion_suite_materializes_declared_token_budgets()
    test_posttrain_configs_share_the_base_architecture()
    test_completion_suite_generates_the_final_report()
    test_formal_configs_keep_repo_output_paths_portable()
    print("GPU completion suite tests ok")
