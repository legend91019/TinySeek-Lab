from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = [
    "20_architecture_evolution_overview.md",
    "21_from_dense_to_deepseek_moe.md",
    "22_from_moe_to_deepseek_v2.md",
    "23_from_v2_to_deepseek_v3.md",
]


def test_bilingual_architecture_chapters() -> None:
    for name in CHAPTERS:
        en_path = ROOT / "docs" / name
        zh_path = ROOT / "docs" / "zh" / name
        assert en_path.exists(), f"Missing English chapter: {name}"
        assert zh_path.exists(), f"Missing Chinese chapter: {name}"
        for path in (en_path, zh_path):
            text = path.read_text(encoding="utf-8")
            assert "model/stages" in text
            assert "configs/architecture_lab" in text
            assert "<!-- tinyseek-nav -->" in text


def test_old_chapters_are_current() -> None:
    zh_sft = (ROOT / "docs" / "zh" / "07_stage5_sft_cold_start.md").read_text(encoding="utf-8")
    en_sft = (ROOT / "docs" / "07_stage5_sft_cold_start.md").read_text(encoding="utf-8")
    assert "还是占位" not in zh_sft
    assert "placeholder" not in en_sft.lower()
    assert "trainer/train_sft.py" in zh_sft
    assert "trainer/train_sft.py" in en_sft


if __name__ == "__main__":
    test_bilingual_architecture_chapters()
    test_old_chapters_are_current()
    print("docs contract ok")
