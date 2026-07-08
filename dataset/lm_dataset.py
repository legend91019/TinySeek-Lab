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


class JsonlInstructionDataset(Dataset):
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
                prompt = row.get("prompt")
                response = row.get("response")
                if prompt is None:
                    instruction = row.get("instruction", "")
                    input_text = row.get("input", "")
                    prompt = instruction if not input_text else f"{instruction}\n{input_text}"
                if response is None:
                    response = row.get("output")
                if prompt and response:
                    self.samples.append((prompt, response))
        if not self.samples:
            raise ValueError(f"No instruction samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt, response = self.samples[idx]
        prompt_text = format_prompt(prompt)
        response_text = response.strip() + "\n"
        prompt_ids = self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        response_ids = self.tokenizer.encode(response_text, add_bos=False, add_eos=True)
        if len(prompt_ids) >= self.max_seq_len:
            prompt_ids = prompt_ids[: max(1, self.max_seq_len // 2)]
        response_room = max(1, self.max_seq_len - len(prompt_ids))
        response_ids = response_ids[:response_room]
        ids = (prompt_ids + response_ids)[: self.max_seq_len]
        labels = ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        if len(ids) < self.max_seq_len:
            pad = self.max_seq_len - len(ids)
            ids = ids + [self.tokenizer.pad_id] * pad
            labels = labels + [-100] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class JsonlPromptDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.samples = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or row.get("instruction")
                answer = row.get("answer") or row.get("target") or row.get("output")
                if prompt and answer is not None:
                    self.samples.append({"prompt": prompt, "answer": str(answer)})
        if not self.samples:
            raise ValueError(f"No prompt samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def format_prompt(prompt: str) -> str:
    return f"### Instruction\n{prompt.strip()}\n\n### Response\n"
