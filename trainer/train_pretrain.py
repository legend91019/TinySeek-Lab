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
from trainer.cost_utils import append_jsonl, collect_cost_summary, reset_gpu_peak_memory, write_cost_summary
from trainer.utils import autocast_context, cosine_lr, load_config, resolve_amp_dtype, set_seed


def train(
    config_path: str,
    data_path: str,
    max_steps_override: int | None = None,
    run_name_override: str | None = None,
    hourly_rate: float = 0.0,
    currency: str = "CNY",
) -> dict:
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
    amp_dtype = resolve_amp_dtype(train_cfg.get("dtype", "float32"), device)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
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
    history_path = out_dir / f"{cfg['run_name']}_history.jsonl"
    if history_path.exists():
        history_path.unlink()
    model_params = model.parameter_count()
    activated_params = model.activated_parameter_estimate()
    print(f"model params={model_params:,} activated_estimate={activated_params:,}")
    print(f"device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")

    model.train()
    step = 0
    start = time.time()
    reset_gpu_peak_memory()
    pbar = tqdm(total=train_cfg["max_steps"], desc=cfg["run_name"])
    last_loss = torch.tensor(float("nan"))
    last_lm_loss = torch.tensor(float("nan"))
    train_compute_seconds = 0.0
    while step < train_cfg["max_steps"]:
        accum_count = 0
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            compute_start = time.perf_counter()
            lr = cosine_lr(step, train_cfg["max_steps"], train_cfg["learning_rate"], train_cfg["warmup_steps"], train_cfg["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            with autocast_context(device, amp_dtype):
                out = model(input_ids, labels)
                loss = out["loss"] + out["aux_loss"]
                scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
            accum_count += 1

            if accum_count < grad_accum_steps:
                if device == "cuda":
                    torch.cuda.synchronize()
                train_compute_seconds += time.perf_counter() - compute_start
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            model.update_router_bias()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            if device == "cuda":
                torch.cuda.synchronize()
            train_compute_seconds += time.perf_counter() - compute_start

            step += 1
            last_loss = loss.detach()
            last_lm_loss = out["lm_loss"].detach()
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                mtp=f"{out['mtp_loss'].item():.3f}",
                aux=f"{out['aux_loss'].item():.4f}",
                lr=f"{lr:.2e}",
            )

            if step % train_cfg["eval_interval"] == 0 or step == train_cfg["max_steps"]:
                val_metrics = evaluate(model, val_loader, device, amp_dtype)
                elapsed = time.time() - start
                print(
                    f"\nstep={step} train_loss={loss.item():.4f} "
                    f"val_lm_loss={val_metrics['lm_loss']:.4f} "
                    f"val_objective={val_metrics['objective']:.4f} "
                    f"lr={lr:.3e} elapsed={elapsed:.1f}s"
                )
                append_jsonl(
                    history_path,
                    {
                        "stage": "pretrain",
                        "run_name": cfg["run_name"],
                        "step": step,
                        "train_loss": float(loss.item()),
                        "lm_loss": float(out["lm_loss"].item()),
                        "mtp_loss": float(out["mtp_loss"].item()),
                        "val_loss": val_metrics["lm_loss"],
                        "val_lm_loss": val_metrics["lm_loss"],
                        "val_objective": val_metrics["objective"],
                        "val_mtp_loss": val_metrics["mtp_loss"],
                        "val_aux_loss": val_metrics["aux_loss"],
                        "aux_loss": float(out["aux_loss"].item()),
                        "learning_rate": float(lr),
                        "elapsed_seconds": round(elapsed, 4),
                        "expert_load": model.expert_load_summary(),
                    },
                )

            if step % train_cfg["save_interval"] == 0 or step == train_cfg["max_steps"]:
                torch.save({"config": cfg, "model": model.state_dict(), "step": step}, ckpt_path)

            if step >= train_cfg["max_steps"]:
                break
    pbar.close()
    final_val_metrics = evaluate(model, val_loader, device, amp_dtype)
    final_val_loss = final_val_metrics["lm_loss"]
    estimated_train_tokens = (
        step * train_cfg["batch_size"] * grad_accum_steps * model_cfg.max_seq_len
    )
    cost_summary = collect_cost_summary(
        run_name=cfg["run_name"],
        start_time=start,
        hourly_rate=hourly_rate,
        currency=currency,
        step=step,
        model_params=model_params,
        activated_params=activated_params,
    )
    cost_summary.update(
        {
            "config_path": config_path,
            "data_path": data_path,
            "checkpoint_path": str(ckpt_path),
            "history_path": str(history_path),
            "last_loss": float(last_loss.item()),
            "last_lm_loss": float(last_lm_loss.item()),
            "val_loss": final_val_loss,
            "val_lm_loss": final_val_metrics["lm_loss"],
            "val_objective": final_val_metrics["objective"],
            "val_mtp_loss": final_val_metrics["mtp_loss"],
            "val_aux_loss": final_val_metrics["aux_loss"],
            "batch_size": train_cfg["batch_size"],
            "grad_accum_steps": grad_accum_steps,
            "max_seq_len": model_cfg.max_seq_len,
            "dtype": train_cfg.get("dtype", "float32"),
            "router_balance_strategy": model_cfg.router_balance_strategy,
            "mtp_depth": model_cfg.mtp_depth,
            "mtp_loss_weight": model_cfg.mtp_loss_weight,
            "estimated_train_tokens": estimated_train_tokens,
            "train_compute_seconds": round(train_compute_seconds, 4),
            "training_tokens_per_second": round(
                estimated_train_tokens / max(train_compute_seconds, 1e-9), 2
            ),
            "throughput_note": "Training compute only; excludes data loading, validation, and checkpoint writes.",
            "estimated_training_flops": 6 * activated_params * estimated_train_tokens,
            "flops_note": "Rough 6 * activated_params * tokens estimate for tutorial-scale comparison.",
            "expert_load": model.expert_load_summary(),
        }
    )
    cost_path = write_cost_summary(out_dir, cfg["run_name"], cost_summary)
    print(f"cost_summary={cost_path} estimated_cost={cost_summary['estimated_cost']} {currency}")
    return {"run_name": cfg["run_name"], "ckpt": str(ckpt_path), "last_loss": float(last_loss.item()), "val_loss": final_val_loss, "cost_summary": str(cost_path)}


@torch.no_grad()
def evaluate(
    model: TinySeekForCausalLM,
    loader: DataLoader,
    device: str,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    model.eval()
    lm_loss_sum = 0.0
    valid_token_count = 0
    objective_sum = 0.0
    mtp_loss_sum = 0.0
    aux_loss_sum = 0.0
    batch_count = 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        with autocast_context(device, amp_dtype):
            out = model(input_ids, labels)
        valid_tokens = int((labels[:, 1:] != -100).sum().item())
        lm_loss_sum += float(out["lm_loss"].item()) * valid_tokens
        valid_token_count += valid_tokens
        objective_sum += float((out["loss"] + out["aux_loss"]).item())
        mtp_loss_sum += float(out["mtp_loss"].item())
        aux_loss_sum += float(out["aux_loss"].item())
        batch_count += 1
    model.train()
    return {
        "objective": objective_sum / max(1, batch_count),
        "lm_loss": lm_loss_sum / max(1, valid_token_count),
        "mtp_loss": mtp_loss_sum / max(1, batch_count),
        "aux_loss": aux_loss_sum / max(1, batch_count),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinySeek pretraining")
    parser.add_argument("--config", default="configs/tiny_dense.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--hourly_rate", type=float, default=0.0, help="GPU rental price per hour. AutoDL examples: RTX 4090=2.18, RTX 3080 Ti=0.98.")
    parser.add_argument("--currency", default="CNY")
    args = parser.parse_args()
    train(args.config, args.data, args.max_steps, args.run_name, args.hourly_rate, args.currency)
