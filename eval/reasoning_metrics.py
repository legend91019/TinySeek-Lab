from __future__ import annotations

import re


THINK_ANSWER_PATTERN = re.compile(
    r"<think>\s*(?P<think>.+?)\s*</think>\s*<answer>\s*(?P<answer>-?\d+)\s*</answer>",
    re.DOTALL | re.IGNORECASE,
)
ANSWER_PATTERN = re.compile(r"<answer>\s*(?P<answer>-?\d+)\s*</answer>", re.IGNORECASE)


def reasoning_match(text: str) -> re.Match[str] | None:
    return THINK_ANSWER_PATTERN.search(text)


def extract_tagged_answer(text: str) -> int | None:
    match = ANSWER_PATTERN.search(text)
    return int(match.group("answer")) if match else None


def has_reasoning_format(text: str) -> bool:
    match = reasoning_match(text)
    return bool(match and match.group("think").strip())
