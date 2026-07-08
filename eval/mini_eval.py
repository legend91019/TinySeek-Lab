from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset import ByteTokenizer, JsonlTextDataset, format_prompt
from model import TinySeekConfig, TinySeekForCausalLM
from trainer.utils import load_config


def load_model(config_path: str, ckpt_path: str, device: str) -> tuple[TinySeekForCausalLM, ByteTokenizer]:
    cfg = load_config(config_path)
    model_cfg = TinySeekConfig.from_dict(cfg["model"])
    model = TinySeekForCausalLM(model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, ByteTokenizer()


@torch.no_grad()
def eval_ppl(model: TinySeekForCausalLM, tokenizer: ByteTokenizer, data_path: str, max_seq_len: int, device: str, max_batches: int) -> dict:
    dataset = JsonlTextDataset(data_path, tokenizer, max_seq_len)
    losses = []
    for idx in range(min(len(dataset), max_batches)):
        x, y = dataset[idx]
        out = model(x.unsqueeze(0).to(device), y.unsqueeze(0).to(device))
        losses.append(float((out["loss"] + out["aux_loss"]).item()))
    mean_loss = sum(losses) / max(1, len(losses))
    return {"loss": mean_loss, "ppl": math.exp(min(20.0, mean_loss)), "num_examples": len(losses)}


@torch.no_grad()
def generate_text(model: TinySeekForCausalLM, tokenizer: ByteTokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True, add_eos=False)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0, top_k=0)
    return tokenizer.decode(out[0].tolist())


def eval_addition(model: TinySeekForCausalLM, tokenizer: ByteTokenizer, device: str, max_new_tokens: int) -> dict:
    cases = [(2, 3), (7, 8), (12, 9), (15, 17), (20, 22)]
    correct = 0
    rows = []
    for a, b in cases:
        prompt = format_prompt(f"Compute {a} + {b}. Answer with the final integer.")
        text = generate_text(model, tokenizer, prompt, device, max_new_tokens)
        pred = extract_last_number(text)
        target = a + b
        ok = pred == target
        correct += int(ok)
        rows.append({"prompt": f"{a}+{b}", "prediction": pred, "target": target, "ok": ok})
    return {"accuracy": correct / len(cases), "num_examples": len(cases), "examples": rows}


def eval_format(model: TinySeekForCausalLM, tokenizer: ByteTokenizer, device: str, max_new_tokens: int) -> dict:
    prompts = [
        "Explain RMSNorm in one sentence.",
        "What does MoE routing do?",
        "Why do we use a validation set?",
    ]
    hits = 0
    rows = []
    for prompt in prompts:
        text = generate_text(model, tokenizer, format_prompt(prompt), device, max_new_tokens)
        has_answer_shape = ("answer" in text.lower()) or ("response" in text.lower()) or len(text.strip()) > len(prompt)
        hits += int(has_answer_shape)
        rows.append({"prompt": prompt, "ok": has_answer_shape, "completion": text[-240:]})
    return {"format_score": hits / len(prompts), "num_examples": len(prompts), "examples": rows}


def extract_last_number(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="TinySeek mini evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default=None, help="Optional JSONL text data for perplexity")
    parser.add_argument("--out", default="out/mini_eval.json")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.config, args.ckpt, device)
    cfg = load_config(args.config)
    model_cfg = TinySeekConfig.from_dict(cfg["model"])
    report = {
        "checkpoint": args.ckpt,
        "device": device,
        "addition": eval_addition(model, tokenizer, device, args.max_new_tokens),
        "format": eval_format(model, tokenizer, device, args.max_new_tokens),
    }
    if args.data:
        report["perplexity"] = eval_ppl(model, tokenizer, args.data, model_cfg.max_seq_len, device, args.max_batches)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
