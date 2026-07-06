from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset import ByteTokenizer, JsonlTextDataset
from model import TinySeekConfig, TinySeekForCausalLM
from trainer.utils import cosine_lr, load_config, set_seed


def train(config_path: str, data_path: str, max_steps_override: int | None = None, run_name_override: str | None = None) -> dict:
    cfg = load_config(config_path)
    if run_name_override:
        cfg["run_name"] = run_name_override
    train_cfg = cfg["train"]
    if max_steps_override is not None:
        train_cfg["max_steps"] = max_steps_override

    set_seed(train_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = ByteTokenizer()
    model_cfg = TinySeekConfig.from_dict(cfg["model"])
    model = TinySeekForCausalLM(model_cfg).to(device)
    dataset = JsonlTextDataset(data_path, tokenizer, model_cfg.max_seq_len)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(train_cfg["seed"]))
    loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95))

    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{cfg['run_name']}_last.pt"
    print(f"model params={model.parameter_count():,} activated_estimate={model.activated_parameter_estimate():,}")
    print(f"device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")

    model.train()
    step = 0
    start = time.time()
    pbar = tqdm(total=train_cfg["max_steps"], desc=cfg["run_name"])
    while step < train_cfg["max_steps"]:
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            lr = cosine_lr(step, train_cfg["max_steps"], train_cfg["learning_rate"], train_cfg["warmup_steps"], train_cfg["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            out = model(input_ids, labels)
            loss = out["loss"] + out["aux_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}", aux=f"{out['aux_loss'].item():.4f}", lr=f"{lr:.2e}")

            if step % train_cfg["eval_interval"] == 0 or step == train_cfg["max_steps"]:
                val_loss = evaluate(model, val_loader, device)
                elapsed = time.time() - start
                print(f"\nstep={step} train_loss={loss.item():.4f} val_loss={val_loss:.4f} lr={lr:.3e} elapsed={elapsed:.1f}s")

            if step % train_cfg["save_interval"] == 0 or step == train_cfg["max_steps"]:
                torch.save({"config": cfg, "model": model.state_dict(), "step": step}, ckpt_path)

            if step >= train_cfg["max_steps"]:
                break
    pbar.close()
    return {"run_name": cfg["run_name"], "ckpt": str(ckpt_path), "last_loss": float(loss.item()), "val_loss": float(evaluate(model, val_loader, device))}


@torch.no_grad()
def evaluate(model: TinySeekForCausalLM, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        out = model(input_ids, labels)
        losses.append((out["loss"] + out["aux_loss"]).item())
    model.train()
    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinySeek pretraining")
    parser.add_argument("--config", default="configs/tiny_dense.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()
    train(args.config, args.data, args.max_steps, args.run_name)
