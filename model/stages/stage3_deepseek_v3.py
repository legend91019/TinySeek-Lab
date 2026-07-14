from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage0_deepseek_llm import RMSNorm, SwiGLU, TransformerBlock
from .stage2_deepseek_v2 import EducationalMLA, Stage2Config


@dataclass
class Stage3Config(Stage2Config):
    router_balance_strategy: str = "bias"
    router_bias_update_rate: float = 0.001
    moe_aux_loss_weight: float = 0.0
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.1


class BiasBalancedMoE(nn.Module):
    """V3-style selection bias with a readable single-device dispatch loop."""

    def __init__(self, config: Stage3Config):
        super().__init__()
        if config.router_balance_strategy != "bias":
            raise ValueError("Stage 3 teaching model expects router_balance_strategy='bias'")
        if not 0 < config.top_k <= config.num_routed_experts:
            raise ValueError("top_k must be between 1 and num_routed_experts")
        self.config = config
        expert_hidden = int(config.hidden_size * config.expert_ffn_multiplier)
        self.gate = nn.Linear(config.hidden_size, config.num_routed_experts, bias=False)
        self.routed = nn.ModuleList(
            [SwiGLU(config.hidden_size, expert_hidden) for _ in range(config.num_routed_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLU(config.hidden_size, expert_hidden) for _ in range(config.num_shared_experts)]
        )
        self.register_buffer("expert_bias", torch.zeros(config.num_routed_experts))
        self.register_buffer(
            "last_expert_counts",
            torch.zeros(config.num_routed_experts, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        affinity = torch.sigmoid(self.gate(flat))

        # Bias changes expert selection only; the mixture weights use raw affinity.
        selection_score = affinity + self.expert_bias.to(affinity.dtype)
        top_indices = torch.topk(selection_score, k=self.config.top_k, dim=-1).indices
        top_weights = affinity.gather(1, top_indices)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        counts = torch.bincount(top_indices.reshape(-1), minlength=self.config.num_routed_experts)
        self.last_expert_counts.copy_(counts.detach())

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

        aux_loss = flat.new_zeros(())
        if self.config.moe_aux_loss_weight > 0:
            importance = affinity.mean(dim=0)
            first_choice = F.one_hot(
                top_indices[:, 0], num_classes=self.config.num_routed_experts
            ).float().mean(dim=0)
            balance_proxy = self.config.num_routed_experts * torch.sum(importance * first_choice)
            aux_loss = self.config.moe_aux_loss_weight * balance_proxy
        return (routed_out + shared_out).view(shape), aux_loss

    @torch.no_grad()
    def update_bias(self) -> None:
        counts = self.last_expert_counts.float()
        target = counts.mean()
        direction = torch.sign(target - counts)
        self.expert_bias.add_(self.config.router_bias_update_rate * direction)
        self.expert_bias.sub_(self.expert_bias.mean())

    def routed_parameter_count(self) -> int:
        return sum(parameter.numel() for expert in self.routed for parameter in expert.parameters())


class DeepSeekV3Block(nn.Module):
    def __init__(self, config: Stage3Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attn = EducationalMLA(config)
        self.ffn = BiasBalancedMoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        return x + ffn_out, aux_loss


class MultiTokenPredictionModule(nn.Module):
    """One sequential MTP depth, sharing token embedding and LM head externally."""

    def __init__(self, config: Stage3Config):
        super().__init__()
        self.previous_norm = RMSNorm(config.hidden_size)
        self.token_norm = RMSNorm(config.hidden_size)
        self.concat_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.block = TransformerBlock(config)

    def forward(self, previous_hidden: torch.Tensor, future_token_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat(
            (self.previous_norm(previous_hidden), self.token_norm(future_token_embed)),
            dim=-1,
        )
        return self.block(self.concat_proj(combined))


class Stage3DeepSeekV3(nn.Module):
    def __init__(self, config: Stage3Config):
        super().__init__()
        if config.mtp_depth < 0:
            raise ValueError("mtp_depth must be non-negative")
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DeepSeekV3Block(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.mtp_modules = nn.ModuleList(
            [MultiTokenPredictionModule(config) for _ in range(config.mtp_depth)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_targets = targets.reshape(-1)
        valid = flat_targets != -100
        if not valid.any():
            return logits.new_zeros(())
        flat_logits = logits.reshape(-1, logits.size(-1))
        return F.cross_entropy(flat_logits[valid], flat_targets[valid])

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        x = self.embed(input_ids)
        aux_losses = []
        for block in self.blocks:
            x, aux_loss = block(x)
            aux_losses.append(aux_loss)
        logits = self.lm_head(self.norm(x))

        lm_loss = None
        mtp_loss = logits.new_zeros(())
        total_loss = None
        if labels is not None:
            lm_loss = self._masked_cross_entropy(logits[:, :-1], labels[:, 1:])
            mtp_losses = []
            previous_hidden = x
            for depth, module in enumerate(self.mtp_modules):
                if previous_hidden.size(1) < 3:
                    break
                future_embed = self.embed(input_ids[:, depth + 1 :])
                previous_hidden = module(previous_hidden[:, :-1], future_embed)
                mtp_logits = self.lm_head(self.norm(previous_hidden[:, :-1]))
                mtp_targets = labels[:, depth + 2 :]
                mtp_losses.append(self._masked_cross_entropy(mtp_logits, mtp_targets))
            if mtp_losses:
                mtp_loss = torch.stack(mtp_losses).mean()
            total_loss = lm_loss + self.config.mtp_loss_weight * mtp_loss

        aux_loss = torch.stack(aux_losses).sum() if aux_losses else logits.new_zeros(())
        return {
            "logits": logits,
            "loss": total_loss,
            "lm_loss": lm_loss,
            "mtp_loss": mtp_loss,
            "aux_loss": aux_loss,
        }

    @torch.no_grad()
    def update_router_bias(self) -> None:
        for block in self.blocks:
            block.ffn.update_bias()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def activated_parameter_estimate(self) -> int:
        total = self.parameter_count()
        routed = sum(block.ffn.routed_parameter_count() for block in self.blocks)
        active_routed = routed * self.config.top_k // self.config.num_routed_experts
        return total - routed + active_routed

    def kv_cache_elements_per_token(self) -> int:
        return self.blocks[0].attn.kv_cache_elements_per_token()

    def expert_load_summary(self) -> dict:
        total_counts = [0 for _ in range(self.config.num_routed_experts)]
        per_layer = []
        for layer, block in enumerate(self.blocks):
            counts = [int(value) for value in block.ffn.last_expert_counts.detach().cpu().tolist()]
            count_sum = max(1, sum(counts))
            per_layer.append(
                {
                    "layer": layer,
                    "counts": counts,
                    "fractions": [value / count_sum for value in counts],
                    "selection_bias": block.ffn.expert_bias.detach().cpu().tolist(),
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
