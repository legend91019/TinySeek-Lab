from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .byte_tokenizer import ByteTokenizer


class JsonlTextDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: ByteTokenizer, max_seq_len: int):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text")
                if text:
                    self.samples.append(text)
        if not self.samples:
            raise ValueError(f"No text samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenizer.encode(self.samples[idx])[: self.max_seq_len]
        if len(ids) < self.max_seq_len:
            ids = ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(ids))
        x = torch.tensor(ids, dtype=torch.long)
        y = x.clone()
        y[y == self.tokenizer.pad_id] = -100
        return x, y
