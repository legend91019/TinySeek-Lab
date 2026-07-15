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

    zh_overview = (ROOT / "docs" / "zh" / CHAPTERS[0]).read_text(encoding="utf-8")
    en_overview = (ROOT / "docs" / CHAPTERS[0]).read_text(encoding="utf-8")
    for phrase in ("可测量瓶颈", "研究假设", "决策门槛", "证据状态"):
        assert phrase in zh_overview
    for phrase in ("Measurable bottleneck", "Research hypothesis", "Decision gate", "Evidence status"):
        assert phrase in en_overview


def test_old_chapters_are_current() -> None:
    zh_sft = (ROOT / "docs" / "zh" / "07_stage5_sft_cold_start.md").read_text(encoding="utf-8")
    en_sft = (ROOT / "docs" / "07_stage5_sft_cold_start.md").read_text(encoding="utf-8")
    assert "还是占位" not in zh_sft
    assert "placeholder" not in en_sft.lower()
    assert "trainer/train_sft.py" in zh_sft
    assert "trainer/train_sft.py" in en_sft


def test_readme_and_indexes_expose_the_course() -> None:
    for path in (ROOT / "README.md", ROOT / "README_zh.md"):
        text = path.read_text(encoding="utf-8")
        assert "stage0_deepseek_llm.py" in text
        assert "stage3_deepseek_v3.py" in text
        assert "architecture_ppl.svg" in text
        assert "06_architecture_evolution_plan" in text
    zh_readme = (ROOT / "README_zh.md").read_text(encoding="utf-8")
    en_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "本仓库是笔者学习 DeepSeek 论文时，为方便理解而完成的作品" in zh_readme
    assert "created while studying the DeepSeek papers" in en_readme
    for path in (ROOT / "docs" / "README.md", ROOT / "docs" / "zh" / "README.md"):
        text = path.read_text(encoding="utf-8")
        positions = [text.index(name) for name in CHAPTERS]
        assert positions == sorted(positions)
    nav_source = (ROOT / "scripts" / "refresh_doc_nav.py").read_text(encoding="utf-8")
    for name in CHAPTERS:
        assert nav_source.count(name) == 2
    for path in (ROOT / "experiments" / "README.md", ROOT / "experiments" / "README_zh.md"):
        assert "06_architecture_evolution_plan" in path.read_text(encoding="utf-8")


if __name__ == "__main__":
    test_bilingual_architecture_chapters()
    test_old_chapters_are_current()
    test_readme_and_indexes_expose_the_course()
    print("docs contract ok")
