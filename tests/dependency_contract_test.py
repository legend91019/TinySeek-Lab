from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_datasets_major_version_is_bounded() -> None:
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "datasets>=2.19,<5" in requirements


if __name__ == "__main__":
    test_datasets_major_version_is_bounded()
    print("dependency contract ok")
