from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage0_deepseek_llm import CausalSelfAttention, RMSNorm, Stage0Config, SwiGLU


@dataclass
class Stage1Config(Stage0Config):
    num_routed_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    expert_ffn_multiplier: float = 1.0
    moe_aux_loss_weight: float = 0.01


class FineGrainedMoE(nn.Module):
    """Single-device teaching version of routed and shared DeepSeekMoE experts."""

    def __init__(self, config: Stage1Config):
        super().__init__()
        if not 0 < config.top_k <= config.num_routed_experts:
            raise ValueError("top_k must be between 1 and num_routed_experts")
        self.config = config
        expert_hidden = int(config.hidden_size * config.expert_ffn_multiplier)
        self.router = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.routed = nn.ModuleList(
            [SwiGLU(config.hidden_size, expert_hidden) for _ in range(config.num_routed_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(config.hidden_size, expert_hidden) for _ in range(config.num_shared_experts)]
        )
        self.last_expert_counts = [0 for _ in range(config.num_routed_experts)]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        router_probs = F.softmax(self.router(flat), dim=-1)
        top_weights, top_indices = torch.topk(router_probs, k=self.config.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        counts = torch.bincount(top_indices.reshape(-1), minlength=self.config.num_routed_experts)
        self.last_expert_counts = counts.detach().cpu().tolist()

        routed_out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.routed):
            token_mask = top_indices == expert_id
            if not token_mask.any():
                continue
            token_indices, slots = token_mask.nonzero(as_tuple=True)
            expert_out = expert(flat[token_indices])
            routed_out[token_indices] += expert_out * top_weights[token_indices, slots].unsqueeze(-1)

        shared_out = torch.zeros_like(flat)
        for expert in self.shared:
            shared_out = shared_out + expert(flat)
        if self.shared:
            shared_out = shared_out / len(self.shared)

        importance = router_probs.mean(dim=0)
        first_choice = F.one_hot(
            top_indices[:, 0], num_classes=self.config.num_routed_experts
        ).float().mean(dim=0)
        balance_proxy = self.config.num_routed_experts * torch.sum(importance * first_choice)
        aux_loss = self.config.moe_aux_loss_weight * balance_proxy
        return (routed_out + shared_out).view(shape), aux_loss

    def routed_parameter_count(self) -> int:
        return sum(parameter.numel() for expert in self.routed for parameter in expert.parameters())


class DeepSeekMoEBlock(nn.Module):
    def __init__(self, config: Stage1Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ffn = FineGrainedMoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        return x + ffn_out, aux_loss


class Stage1DeepSeekMoE(nn.Module):
    def __init__(self, config: Stage1Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DeepSeekMoEBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        x = self.embed(input_ids)
        aux_losses = []
        for block in self.blocks:
            x, aux_loss = block(x)
            aux_losses.append(aux_loss)
        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        aux_loss = torch.stack(aux_losses).sum() if aux_losses else logits.new_zeros(())
        return {"logits": logits, "loss": loss, "aux_loss": aux_loss}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def activated_parameter_estimate(self) -> int:
        total = self.parameter_count()
        routed = sum(block.ffn.routed_parameter_count() for block in self.blocks)
        active_routed = routed * self.config.top_k // self.config.num_routed_experts
        return total - routed + active_routed

    def kv_cache_elements_per_token(self) -> int:
        return self.config.num_layers * self.blocks[0].attn.kv_cache_elements_per_token()

    def expert_load_summary(self) -> dict:
        total_counts = [0 for _ in range(self.config.num_routed_experts)]
        per_layer = []
        for layer, block in enumerate(self.blocks):
            counts = [int(value) for value in block.ffn.last_expert_counts]
            count_sum = max(1, sum(counts))
            per_layer.append(
                {
                    "layer": layer,
                    "counts": counts,
                    "fractions": [value / count_sum for value in counts],
                }
            )
            total_counts = [left + right for left, right in zip(total_counts, counts)]
        total = max(1, sum(total_counts))
        return {
            "top_k": self.config.top_k,
            "num_experts": self.config.num_routed_experts,
            "total_counts": total_counts,
            "total_fractions": [value / total for value in total_counts],
            "per_layer": per_layer,
        }
