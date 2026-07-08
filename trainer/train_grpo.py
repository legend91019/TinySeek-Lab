from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset import ByteTokenizer, JsonlPromptDataset, format_prompt
from model import TinySeekConfig, TinySeekForCausalLM
from trainer.cost_utils import collect_cost_summary, reset_gpu_peak_memory, write_cost_summary
from trainer.utils import autocast_context, cosine_lr, load_config, resolve_amp_dtype, set_seed


def train_grpo(
    config_path: str,
    data_path: str,
    init_ckpt: str,
    max_steps_override: int | None = None,
    run_name_override: str | None = None,
    hourly_rate: float = 0.0,
    currency: str = "CNY",
) -> dict:
    cfg = load_config(config_path)
    if run_name_override:
        cfg["run_name"] = run_name_override
    train_cfg = cfg["train"]
    grpo_cfg = cfg.get("grpo", {})
    if max_steps_override is not None:
        train_cfg["max_steps"] = max_steps_override

    set_seed(train_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = ByteTokenizer()
    model_cfg = TinySeekConfig.from_dict(cfg["model"])
    policy = TinySeekForCausalLM(model_cfg).to(device)
    ref = TinySeekForCausalLM(model_cfg).to(device)
    state = torch.load(init_ckpt, map_location=device)
    policy.load_state_dict(state["model"])
    ref.load_state_dict(state["model"])
    ref.eval()
    for param in ref.parameters():
        param.requires_grad_(False)

    amp_dtype = resolve_amp_dtype(train_cfg.get("dtype", "float32"), device)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    dataset = JsonlPromptDataset(data_path)
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"], betas=(0.9, 0.95))

    out_dir = Path(train_cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{cfg['run_name']}_last.pt"
    model_params = policy.parameter_count()
    activated_params = policy.activated_parameter_estimate()
    print(f"model params={model_params:,} activated_estimate={activated_params:,}")
    print(f"device={device} grpo_prompts={len(dataset)}")

    step = 0
    start = time.time()
    last_loss = torch.tensor(float("nan"))
    last_reward = 0.0
    reset_gpu_peak_memory()
    pbar = tqdm(total=train_cfg["max_steps"], desc=cfg["run_name"])
    while step < train_cfg["max_steps"]:
        for batch in loader:
            lr = cosine_lr(step, train_cfg["max_steps"], train_cfg["learning_rate"], train_cfg["warmup_steps"], train_cfg["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr

            losses = []
            rewards_for_log = []
            prompts = list(batch["prompt"])
            answers = list(batch["answer"])
            for prompt, answer in zip(prompts, answers):
                group_sequences = sample_group(policy, tokenizer, prompt, grpo_cfg, model_cfg, device)
                policy.train()
                rewards = torch.tensor([rule_reward(seq["completion"], answer) for seq in group_sequences], dtype=torch.float32, device=device)
                advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
                rewards_for_log.extend([float(r) for r in rewards.detach().cpu().tolist()])
                for seq, adv in zip(group_sequences, advantages):
                    ids = seq["ids"].to(device)
                    prompt_len = seq["prompt_len"]
                    with autocast_context(device, amp_dtype):
                        policy_logp = completion_logprob(policy, ids, prompt_len)
                        with torch.no_grad():
                            ref_logp = completion_logprob(ref, ids, prompt_len)
                        kl_proxy = (policy_logp - ref_logp).pow(2)
                        losses.append((-adv.detach() * policy_logp) + float(grpo_cfg.get("kl_beta", 0.02)) * kl_proxy)

            if not losses:
                continue
            loss = torch.stack(losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            last_loss = loss.detach()
            last_reward = sum(rewards_for_log) / max(1, len(rewards_for_log))
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.3f}", reward=f"{last_reward:.3f}", lr=f"{lr:.2e}")

            if step % train_cfg["eval_interval"] == 0 or step == train_cfg["max_steps"]:
                elapsed = time.time() - start
                print(f"\nstep={step} grpo_loss={loss.item():.4f} mean_reward={last_reward:.4f} lr={lr:.3e} elapsed={elapsed:.1f}s")

            if step % train_cfg["save_interval"] == 0 or step == train_cfg["max_steps"]:
                torch.save({"config": cfg, "model": policy.state_dict(), "step": step, "init_ckpt": init_ckpt}, ckpt_path)

            if step >= train_cfg["max_steps"]:
                break
    pbar.close()
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
            "stage": "grpo_mini",
            "config_path": config_path,
            "data_path": data_path,
            "checkpoint_path": str(ckpt_path),
            "init_ckpt": init_ckpt,
            "last_loss": float(last_loss.item()),
            "mean_reward": last_reward,
            "batch_size": train_cfg["batch_size"],
            "group_size": int(grpo_cfg.get("group_size", 4)),
            "max_new_tokens": int(grpo_cfg.get("max_new_tokens", 32)),
            "dtype": train_cfg.get("dtype", "float32"),
        }
    )
    cost_path = write_cost_summary(out_dir, cfg["run_name"], cost_summary)
    print(f"cost_summary={cost_path} estimated_cost={cost_summary['estimated_cost']} {currency}")
    return {"run_name": cfg["run_name"], "ckpt": str(ckpt_path), "mean_reward": last_reward, "cost_summary": str(cost_path)}


@torch.no_grad()
def sample_group(policy: TinySeekForCausalLM, tokenizer: ByteTokenizer, prompt: str, grpo_cfg: dict, model_cfg: TinySeekConfig, device: str) -> list[dict]:
    group = []
    prompt_ids = tokenizer.encode(format_prompt(prompt), add_bos=True, add_eos=False)
    prompt_tensor = torch.tensor([prompt_ids[-model_cfg.max_seq_len :]], dtype=torch.long, device=device)
    for _ in range(int(grpo_cfg.get("group_size", 4))):
        out = policy.generate(
            prompt_tensor.clone(),
            max_new_tokens=int(grpo_cfg.get("max_new_tokens", 32)),
            temperature=float(grpo_cfg.get("temperature", 0.8)),
            top_k=int(grpo_cfg.get("top_k", 40)),
        )
        ids = out[0].detach().cpu()
        completion = tokenizer.decode(ids[len(prompt_tensor[0]) :].tolist())
        group.append({"ids": ids.unsqueeze(0), "prompt_len": len(prompt_tensor[0]), "completion": completion})
    return group


def completion_logprob(model: TinySeekForCausalLM, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    logits = model(ids)["logits"][:, :-1, :]
    targets = ids[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(targets.size(1), device=ids.device) + 1
    mask = positions >= prompt_len
    masked = logp[:, mask]
    if masked.numel() == 0:
        return logp.new_zeros(())
    return masked.mean()


def rule_reward(completion: str, answer: str) -> float:
    pred = extract_last_number(completion)
    target = extract_last_number(answer)
    if pred is None or target is None:
        return 0.0
    return 1.0 if pred == target else 0.0


def extract_last_number(text: str) -> int | None:
    matches = re.findall(r"-?\d+", text)
    return int(matches[-1]) if matches else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinySeek rule-based GRPO mini")
    parser.add_argument("--config", default="configs/tiny_grpo.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--init_ckpt", required=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--hourly_rate", type=float, default=0.0)
    parser.add_argument("--currency", default="CNY")
    args = parser.parse_args()
    train_grpo(args.config, args.data, args.init_ckpt, args.max_steps, args.run_name, args.hourly_rate, args.currency)
