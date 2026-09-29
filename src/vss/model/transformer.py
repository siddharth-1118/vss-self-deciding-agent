"""VSS transformer stack.

Modern transformer blocks used as a bidirectional encoder (no causal
masking): RoPE [std], RMSNorm [std], SwiGLU FFN [std], grouped-query
attention [std]. Sequence order:

    tokens -> token emb -> blocks -> final norm
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cos_sin(seq_len: int, head_dim: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin tables [std]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D/2] or [B, 1, T, D/2] (per-token pos)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    """Gated FFN: down( silu(gate(x)) * up(x) ) [std]."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Multi-head attention with GQA and RoPE; bidirectional (no mask)."""

    def __init__(self, dim: int, heads: int, kv_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if heads % kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        group = self.heads // self.kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,       # additive bias (padding mask) [std]
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # bidirectional encoder
        )
        out = out.transpose(1, 2).reshape(bsz, seq, self.heads * self.head_dim)
        return self.o_proj(out)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, kv_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, heads, kv_heads, dropout)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), cos, sin, attn_bias))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm blocks ending with a final RMSNorm."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        layers: int,
        heads: int,
        kv_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        pad_token_id: int = 0,
        question_masked: bool = False,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.question_masked = question_masked
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList(
            Block(dim, heads, kv_heads, ffn_dim, dropout) for _ in range(layers)
        )
        self.norm = RMSNorm(dim)
        self.rope_theta = rope_theta
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._rope_len = 0

    def _rope(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos is None or self._rope_len < seq_len or self._cos.device != device:
            head_dim = self.blocks[0].attn.head_dim
            # build once, cache (capped at max_seq_len)
            cap = min(max(seq_len, 256), self.max_seq_len)
            cos, sin = rope_cos_sin(cap, head_dim, self.rope_theta, device)
            self._cos, self._sin, self._rope_len = cos, sin, cap
        return self._cos[:seq_len], self._sin[:seq_len]

    def forward(
        self,
        token_ids: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """token_ids: [B, T] -> contextual features [B, T, D]. Padding masked.

        `attn_bias` (optional): additive [1,1,T,T] or [B,1,T,T] bias combined
        with the padding mask (used for question-masked attention [vss-qmask]).
        `positions` (optional): [B, T] int tensor of RoPE positions; required
        with question masking so tokens keep stable positions regardless of
        how many questions follow them in the request.
        """
        bsz, seq = token_ids.shape
        if seq > self.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max {self.max_seq_len}")
        x = self.token_emb(token_ids)
        if positions is None:
            cos, sin = self._rope(seq, token_ids.device)
        else:
            cos_full, sin_full = self._rope(int(positions.max().item()) + 1,
                                            token_ids.device)
            cos = cos_full[positions]  # [B, T, D/2]
            sin = sin_full[positions]
            cos = cos.unsqueeze(1)  # [B, 1, T, D/2] broadcast over heads
            sin = sin.unsqueeze(1)

        pad = token_ids.eq(self.pad_token_id)  # [B, T]
        pad_bias = torch.zeros((bsz, 1, 1, seq), dtype=x.dtype, device=x.device)
        pad_bias.masked_fill_(pad[:, None, None, :], torch.finfo(x.dtype).min)
        if attn_bias is not None:
            if attn_bias.shape[-1] != seq or attn_bias.shape[-2] != seq:
                raise ValueError(
                    f"attn_bias seq mismatch: {tuple(attn_bias.shape)} vs T={seq}")
            if attn_bias.shape[0] == 1:
                pad_bias = pad_bias + attn_bias.to(pad_bias.dtype)
            else:
                pad_bias = pad_bias + attn_bias.to(pad_bias.dtype)

        for block in self.blocks:
            x = block(x, cos, sin, pad_bias)
        x = self.norm(x)
        return x.masked_fill(pad[:, :, None], 0.0)
