from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trainer.utils import save_jsonl


TEXT_SUFFIXES = {".txt", ".md", ".jsonl"}


def iter_texts(input_dir: Path):
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            yield path, normalize_text(text)


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def chunk_text(text: str, max_chars: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = end
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a JSONL pretraining corpus from local text files")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out", default="data/corpus_pretrain.jsonl")
    parser.add_argument("--max_chars", type=int, default=1800)
    parser.add_argument("--min_chars", type=int, default=80)
    args = parser.parse_args()

    rows = []
    input_dir = Path(args.input_dir)
    for path, text in iter_texts(input_dir):
        for chunk in chunk_text(text, args.max_chars):
            if len(chunk) >= args.min_chars:
                rows.append({"text": chunk, "source": str(path)})
    save_jsonl(args.out, rows)
    print(f"wrote {len(rows)} chunks to {args.out}")


if __name__ == "__main__":
    main()
