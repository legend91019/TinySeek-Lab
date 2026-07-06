from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset import ByteTokenizer
from model import TinySeekConfig, TinySeekForCausalLM
from trainer.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_dense.json")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", default="DeepSeek is")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = TinySeekConfig.from_dict(cfg["model"])
    model = TinySeekForCausalLM(model_cfg)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"])
    tok = ByteTokenizer()
    ids = torch.tensor([tok.encode(args.prompt, add_bos=True, add_eos=False)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
