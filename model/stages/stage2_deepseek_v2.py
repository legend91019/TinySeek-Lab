from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stage0_deepseek_llm import RMSNorm, apply_rope, precompute_rope, repeat_kv
from .stage1_deepseek_moe import FineGrainedMoE, Stage1Config


@dataclass
class Stage2Config(Stage1Config):
    kv_lora_rank: int = 64
    qk_rope_head_dim: int = 16


class EducationalMLA(nn.Module):
    """MLA data path for learning; it does not implement a fused decode cache."""

    def __init__(self, config: Stage2Config):
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.rope_head_dim = config.qk_rope_head_dim
        self.content_head_dim = self.head_dim - self.rope_head_dim
        if self.content_head_dim <= 0:
            raise ValueError("qk_rope_head_dim must be smaller than attention head_dim")
        if self.rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even")
        self.kv_lora_rank = config.kv_lora_rank
        self.kv_repeats = config.num_heads // config.num_kv_heads
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.k_content_up = nn.Linear(
            config.kv_lora_rank,
            config.num_kv_heads * self.content_head_dim,
            bias=False,
        )
        self.v_up = nn.Linear(
            config.kv_lora_rank,
            config.num_kv_heads * self.head_dim,
            bias=False,
        )
        self.k_rope_proj = nn.Linear(config.hidden_size, self.rope_head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        cos, sin = precompute_rope(self.rope_head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_content, q_rope = torch.split(
            q, [self.content_head_dim, self.rope_head_dim], dim=-1
        )
        q_rope = apply_rope(q_rope, self.rope_cos, self.rope_sin)

        compressed_kv = self.kv_down(x)
        k_content = self.k_content_up(compressed_kv).view(
            batch, seq_len, self.num_kv_heads, self.content_head_dim
        ).transpose(1, 2)
        v = self.v_up(compressed_kv).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        k_content = repeat_kv(k_content, self.kv_repeats)
        v = repeat_kv(v, self.kv_repeats)

        k_rope = self.k_rope_proj(x).view(batch, seq_len, 1, self.rope_head_dim).transpose(1, 2)
        k_rope = apply_rope(k_rope, self.rope_cos, self.rope_sin)
        k_rope = k_rope.expand(batch, self.num_heads, seq_len, self.rope_head_dim)

        q = torch.cat((q_content, q_rope), dim=-1)
        k = torch.cat((k_content, k_rope), dim=-1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(y)

    def kv_cache_elements_per_token(self) -> int:
        return self.kv_lora_rank + self.rope_head_dim


class DeepSeekV2Block(nn.Module):
    def __init__(self, config: Stage2Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attn = EducationalMLA(config)
        self.ffn = FineGrainedMoE(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        return x + ffn_out, aux_loss


class Stage2DeepSeekV2(nn.Module):
    def __init__(self, config: Stage2Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([DeepSeekV2Block(config) for _ in range(config.num_layers)])
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
        return self.blocks[0].attn.kv_cache_elements_per_token()

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
