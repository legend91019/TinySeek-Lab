from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from trainer.utils import save_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a HuggingFace dataset split to TinySeek JSONL")
    parser.add_argument("--dataset_name", required=True, help="For example: wikitext, roneneldan/TinyStories, or a MiniMind-style dataset repo")
    parser.add_argument("--dataset_config", default=None, help="Optional dataset config, for example wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--out", default="data/hf_pretrain.jsonl")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing optional dependency: pip install datasets") from exc

    if args.dataset_config:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.split, streaming=args.streaming)
    else:
        ds = load_dataset(args.dataset_name, split=args.split, streaming=args.streaming)

    rows = []
    for row in ds:
        text = row.get(args.text_field)
        if text is None:
            available = ", ".join(row.keys())
            raise SystemExit(f"Field '{args.text_field}' not found. Available fields: {available}")
        text = str(text).strip()
        if len(text) < args.min_chars:
            continue
        rows.append({"text": text, "source": args.dataset_name})
        if len(rows) >= args.max_samples:
            break

    save_jsonl(args.out, rows)
    print(f"wrote {len(rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
