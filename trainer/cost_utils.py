from __future__ import annotations

import json
import platform
import socket
import time
from pathlib import Path

import torch


AUTODL_RATES_CNY = {
    "rtx_4090": 2.18,
    "rtx_3080_ti": 0.98,
}


def reset_gpu_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_gpu_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if not torch.cuda.is_available():
        return info

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    info.update(
        {
            "device_index": device,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_total_memory_gb": round(props.total_memory / 1024**3, 4),
            "compute_capability": f"{props.major}.{props.minor}",
            "device_count": torch.cuda.device_count(),
        }
    )
    return info


def collect_cost_summary(
    *,
    run_name: str,
    start_time: float,
    hourly_rate: float,
    currency: str,
    step: int,
    model_params: int,
    activated_params: int,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_seconds = time.time() - start_time
    gpu_hours = elapsed_seconds / 3600.0
    summary = {
        "run_name": run_name,
        "step": step,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "gpu_hours": round(gpu_hours, 6),
        "hourly_rate": hourly_rate,
        "currency": currency,
        "estimated_cost": round(gpu_hours * hourly_rate, 4),
        "model_params": model_params,
        "activated_params_estimate": activated_params,
        "hardware": get_gpu_info(),
    }

    if torch.cuda.is_available():
        summary["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        summary["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 4)
    else:
        summary["max_memory_allocated_gb"] = 0.0
        summary["max_memory_reserved_gb"] = 0.0
    return summary


def write_cost_summary(out_dir: str | Path, run_name: str, summary: dict) -> Path:
    path = Path(out_dir) / f"{run_name}_cost_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path
