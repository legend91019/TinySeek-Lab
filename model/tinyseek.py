from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinySeekConfig:
    vocab_size: int = 260
    max_seq_len: int = 128
    hidden_size: int = 192
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 4
    ffn_multiplier: float = 4.0
    activation: str = "swiglu"
    attention_impl: str = "mha"
    mla_latent_size: int = 64
    use_moe: bool = False
    num_experts: int = 4
    num_shared_experts: int = 0
    top_k: int = 2
    moe_aux_loss_weight: float = 0.01
    router_balance_strategy: str = "aux"
    router_bias_update_rate: float = 0.001
    mtp_depth: int = 0
    mtp_loss_weight: float = 0.1
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "TinySeekConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, num_repeats: int) -> torch.Tensor:
    if num_repeats == 1:
        return x
    bsz, n_kv, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, n_kv, num_repeats, seq_len, head_dim)
    return x.reshape(bsz, n_kv * num_repeats, seq_len, head_dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TinySeekConfig):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0
        assert config.num_heads % config.num_kv_heads == 0
        self.config = config
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.kv_repeat = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        if config.attention_impl == "educational_mla":
            self.kv_down = nn.Linear(config.hidden_size, config.mla_latent_size, bias=False)
            self.k_up = nn.Linear(config.mla_latent_size, config.num_kv_heads * self.head_dim, bias=False)
            self.v_up = nn.Linear(config.mla_latent_size, config.num_kv_heads * self.head_dim, bias=False)
        else:
            self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        cos, sin = precompute_rope(self.head_dim, config.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.config.attention_impl == "educational_mla":
            latent = self.kv_down(x)
            k = self.k_up(latent).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_up(latent).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        else:
            k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)
        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(y)

    def kv_cache_elements_per_token(self) -> int:
        if self.config.attention_impl == "educational_mla":
            return self.config.mla_latent_size
        return 2 * self.num_kv_heads * self.head_dim


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DenseFFN(nn.Module):
    def __init__(self, config: TinySeekConfig):
        super().__init__()
        hidden_dim = int(config.hidden_size * config.ffn_multiplier)
        self.net = SwiGLU(config.hidden_size, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.net(x), x.new_zeros(())


class MoEFFN(nn.Module):
    def __init__(self, config: TinySeekConfig):
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_size * config.ffn_multiplier)
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config.hidden_size, hidden_dim) for _ in range(config.num_experts)])
        self.shared = nn.ModuleList([SwiGLU(config.hidden_size, hidden_dim) for _ in range(config.num_shared_experts)])
        self.register_buffer("expert_bias", torch.zeros(config.num_experts), persistent=False)
        self.register_buffer(
            "last_expert_counts",
            torch.zeros(config.num_experts, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.gate(flat)
        if self.config.router_balance_strategy == "bias":
            probs = torch.sigmoid(logits)
            selection_scores = probs + self.expert_bias.to(probs.dtype)
            top_i = torch.topk(selection_scores, k=self.config.top_k, dim=-1).indices
            top_p = probs.gather(1, top_i)
        elif self.config.router_balance_strategy == "aux":
            probs = F.softmax(logits, dim=-1)
            top_p, top_i = torch.topk(probs, k=self.config.top_k, dim=-1)
        else:
            raise ValueError(
                f"Unknown router_balance_strategy: {self.config.router_balance_strategy}"
            )
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)
        with torch.no_grad():
            counts = torch.bincount(top_i.reshape(-1), minlength=self.config.num_experts)
            self.last_expert_counts.copy_(counts.detach())

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = top_i == expert_id
            if not mask.any():
                continue
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            expert_out = expert(flat[token_idx])
            out[token_idx] += expert_out * top_p[token_idx, slot_idx].unsqueeze(-1)

        for expert in self.shared:
            out = out + expert(flat) / max(1, len(self.shared))

        # Switch-style load balancing proxy: encourage probability mass and
        # actual token assignment to agree and stay near-uniform.
        aux_loss = flat.new_zeros(())
        if self.config.moe_aux_loss_weight > 0:
            importance = probs.mean(dim=0)
            assignment = F.one_hot(top_i[:, 0], num_classes=self.config.num_experts).float().mean(dim=0)
            aux_loss = self.config.num_experts * torch.sum(importance * assignment)
            aux_loss = aux_loss * self.config.moe_aux_loss_weight
        return out.view(shape), aux_loss

    @torch.no_grad()
    def update_bias(self) -> None:
        if self.config.router_balance_strategy != "bias":
            return
        counts = self.last_expert_counts.float()
        direction = torch.sign(counts.mean() - counts)
        self.expert_bias.add_(self.config.router_bias_update_rate * direction)
        self.expert_bias.sub_(self.expert_bias.mean())


class TinySeekBlock(nn.Module):
    def __init__(self, config: TinySeekConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ffn = MoEFFN(config) if config.use_moe else DenseFFN(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.attn_norm(x))
        ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, aux_loss


class MultiTokenPredictionModule(nn.Module):
    def __init__(self, config: TinySeekConfig):
        super().__init__()
        self.previous_norm = RMSNorm(config.hidden_size)
        self.token_norm = RMSNorm(config.hidden_size)
        self.concat_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        dense_config = replace(config, use_moe=False, mtp_depth=0)
        self.block = TinySeekBlock(dense_config)

    def forward(self, previous_hidden: torch.Tensor, future_token_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat(
            (self.previous_norm(previous_hidden), self.token_norm(future_token_embed)),
            dim=-1,
        )
        hidden, _ = self.block(self.concat_proj(combined))
        return hidden


class TinySeekForCausalLM(nn.Module):
    def __init__(self, config: TinySeekConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TinySeekBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.mtp_modules = nn.ModuleList(
            [MultiTokenPredictionModule(config) for _ in range(config.mtp_depth)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
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
        x = self.norm(x)
        logits = self.lm_head(x)

        lm_loss = None
        mtp_loss = logits.new_zeros(())
        loss = None
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
            loss = lm_loss + self.config.mtp_loss_weight * mtp_loss
        aux_loss = torch.stack([a for a in aux_losses]).sum() if aux_losses else logits.new_zeros(())
        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "mtp_loss": mtp_loss,
            "aux_loss": aux_loss,
        }

    @torch.no_grad()
    def update_router_bias(self) -> None:
        if not self.config.use_moe:
            return
        for block in self.blocks:
            block.ffn.update_bias()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 80, temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.config.max_seq_len :]
            logits = self(idx)["logits"][:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def activated_parameter_estimate(self) -> int:
        total = self.parameter_count()
        if not self.config.use_moe:
            return total
        expert_params = sum(p.numel() for block in self.blocks for expert in block.ffn.experts for p in expert.parameters())
        active_expert_params = expert_params * self.config.top_k // self.config.num_experts
        return total - expert_params + active_expert_params

    def expert_load_summary(self) -> dict | None:
        if not self.config.use_moe:
            return None
        per_layer = []
        totals = [0 for _ in range(self.config.num_experts)]
        for layer_idx, block in enumerate(self.blocks):
            counts = getattr(block.ffn, "last_expert_counts", None)
            if counts is None:
                continue
            counts = [int(v) for v in counts.detach().cpu().tolist()]
            total = max(1, sum(counts))
            fractions = [v / total for v in counts]
            per_layer.append(
                {
                    "layer": layer_idx,
                    "counts": counts,
                    "fractions": fractions,
                    "selection_bias": block.ffn.expert_bias.detach().cpu().tolist(),
                }
            )
            totals = [a + b for a, b in zip(totals, counts)]
        total_assignments = max(1, sum(totals))
        return {
            "top_k": self.config.top_k,
            "num_experts": self.config.num_experts,
            "total_counts": totals,
            "total_fractions": [v / total_assignments for v in totals],
            "per_layer": per_layer,
        }
