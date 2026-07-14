from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def test_local_markdown_links() -> None:
    broken = []
    for markdown_path in ROOT.rglob("*.md"):
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in LINK_RE.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            target_path = (markdown_path.parent / unquote(target)).resolve()
            if not target_path.exists():
                broken.append(f"{markdown_path.relative_to(ROOT)} -> {raw_target}")
    assert not broken, "Broken local Markdown links:\n" + "\n".join(broken)


if __name__ == "__main__":
    test_local_markdown_links()
    print("markdown links ok")
