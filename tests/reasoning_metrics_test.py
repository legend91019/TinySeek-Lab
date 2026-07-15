from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def test_reasoning_tag_parser_requires_ordered_nonempty_tags() -> None:
    from eval.reasoning_metrics import extract_tagged_answer, has_reasoning_format

    good = "<think>2 + 3 = 5.</think>\n<answer>5</answer>"
    assert extract_tagged_answer(good) == 5
    assert has_reasoning_format(good)
    assert extract_tagged_answer("<answer>5</answer>") == 5
    assert not has_reasoning_format("<answer>5</answer>")
    assert extract_tagged_answer("The final answer is 5") is None
    assert not has_reasoning_format("<think></think><answer>5</answer>")
    assert not has_reasoning_format("<answer>5</answer><think>done</think>")


def test_mini_eval_exposes_reasoning_metrics() -> None:
    source = (ROOT / "eval" / "mini_eval.py").read_text(encoding="utf-8")
    assert "def eval_reasoning(" in source
    assert '"reasoning": eval_reasoning(' in source


if __name__ == "__main__":
    test_reasoning_tag_parser_requires_ordered_nonempty_tags()
    test_mini_eval_exposes_reasoning_metrics()
    print("reasoning metrics tests ok")
