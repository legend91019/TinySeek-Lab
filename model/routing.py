from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_load_balance_loss(
    router_scores: torch.Tensor,
    top_indices: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
    """Average the router importance/load agreement independently per sequence."""
    expected_tokens = batch_size * sequence_length
    if router_scores.shape != (expected_tokens, num_experts):
        raise ValueError("router_scores must have shape [batch * sequence, num_experts]")
    if top_indices.shape != (expected_tokens, top_k):
        raise ValueError("top_indices must have shape [batch * sequence, top_k]")

    scores = router_scores.view(batch_size, sequence_length, num_experts)
    scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    importance = scores.mean(dim=1)

    assignments = F.one_hot(top_indices, num_classes=num_experts).float().sum(dim=1)
    assignments = assignments.view(batch_size, sequence_length, num_experts)
    load = assignments.mean(dim=1) / top_k
    return num_experts * (importance * load).sum(dim=-1).mean()
