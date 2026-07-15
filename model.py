"""A single Expert: a tiny decoder-only transformer.

Each expert owns:
  - an input embedding table sized to *its own* vocabulary
  - a stack of transformer blocks (shared architecture, separate weights)
  - an LM head tied to the input embeddings (standard for small models)

The expert exposes:
  - forward(ids)            -> logits, hidden   (for pre-training LM loss)
  - encode(ids)             -> hidden           (last-layer states, [B,T,d])
  - project(hidden)         -> z                (into shared space)
  - unproject(z)            -> hidden'          (from shared space back)
  - logits_from_hidden(h)   -> logits           (for next-token prediction
                                                  after a switch)

The projection layers (Linear d_model -> shared_dim and back) live on the
expert so that joint fine-tuning only needs to touch these small matrices.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ExpertConfig, SharedSpaceConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, self.n_heads, 3, self.head_dim)
        # [B, T, n_heads, 3, head_dim] -> split along dim=3
        q, k, v = qkv[..., 0, :], qkv[..., 1, :], qkv[..., 2, :]
        q = q.transpose(1, 2)  # [B, h, T, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Scaled dot-product attention with causal mask.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att + attn_mask  # [B,1,T,T] broadcast
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v  # [B, h, T, hd]
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class CrossAttention(nn.Module):
    """CALM-style cross-attention block (Schuster et al. 2022).

    Queries come from the receiving expert's own hidden states (x), while
    keys and values come from an external memory -- here, the *other*
    expert's carried hidden states projected through the shared space.

    This is the inter-expert analogue of CALM's cross-attention between an
    early-exit decoder and a deeper one: every position of the receiving
    segment can attend to all K carried sender states in a content-
    addressable way, instead of only seeing them as fixed seed positions.

    Layout:
        x:   [B, T, d_model]            (receiving expert's hidden states)
        mem: [B, S, d_model]            (carried states, already projected
                                         back into this expert's d_model via
                                         from_shared_space)
    Returns: [B, T, d_model] (same shape as x), with a residual + LayerNorm
    applied so it can be composed with the expert's own blocks.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 residual: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"cross-attn: d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.residual = residual
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # Separate projections for query (from own hidden) and key/value
        # (from external memory). Both live in d_model so the receiving
        # expert's from_shared_space output feeds straight in.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_q = nn.LayerNorm(d_model, bias=False)
        self.norm_kv = nn.LayerNorm(d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        S = mem.size(1)
        q = self.q_proj(self.norm_q(x))   # [B, T, C]
        k = self.k_proj(self.norm_kv(mem))  # [B, S, C]
        v = self.v_proj(self.norm_kv(mem))  # [B, S, C]
        # Reshape to heads: [B, h, *, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention. No causal mask: the carried memory is
        # a fixed set of sender states that every query position may attend
        # to fully (it is "past" context from the other expert).
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v  # [B, h, T, head_dim]
        y = y.transpose(1, 2).reshape(B, T, C)
        y = self.out_proj(y)
        y = self.drop(y)
        # Refine the expert representation residually, or replace it when
        # explicitly configured to use a non-residual bridge.
        return self.norm_out(x + y if self.residual else y)


class Block(nn.Module):
    def __init__(self, cfg: ExpertConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class Expert(nn.Module):
    """A single cooperating expert."""

    def __init__(self, cfg: ExpertConfig, shared_cfg: SharedSpaceConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.shared_cfg = shared_cfg
        self.vocab_size = vocab_size

        # Token + position embeddings (per-expert vocab).
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)

        # LM head tied to embeddings.
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        # Lightweight projections to/from the shared space.
        self.to_shared = nn.Linear(cfg.d_model, shared_cfg.dim, bias=False)
        self.from_shared = nn.Linear(shared_cfg.dim, cfg.d_model, bias=False)

        # Optional CALM-style cross-attention bridge. When enabled, the
        # expert gets a CrossAttention block that attends over the *other*
        # expert's carried states (already projected back into this expert's
        # d_model via from_shared_space). See SharedSpaceConfig.cross_attn.
        self.cross_attn: nn.Module = None
        if shared_cfg.cross_attn:
            self.cross_attn = CrossAttention(
                cfg.d_model, shared_cfg.cross_attn_n_heads,
                shared_cfg.cross_attn_dropout,
                residual=shared_cfg.cross_attn_residual,
            )

        # Causal mask is built dynamically in _blocks (see above).

        self.apply(self._init_weights)

    # ------------------------------------------------------------------ #
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------ #
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)
        x = self.tok_emb(ids) + self.pos_emb(pos)[None, :, :]
        return self.drop(x)

    def _blocks(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        # Build the causal mask dynamically so it works for any sequence
        # length (including the +1 prepended carried-state in joint_loss).
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_f(x)

    # ------------------------------------------------------------------ #
    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard LM forward. Returns (logits, hidden)."""
        x = self._embed(ids)
        h = self._blocks(x)  # [B, T, d_model]
        logits = self.head(h)
        return logits, h

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Return last-layer hidden states [B, T, d_model]."""
        x = self._embed(ids)
        return self._blocks(x)

    def encode_with_seed(
        self, ids: torch.Tensor, seed: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode ids, optionally seeded by carried hidden states.

        `seed` is [B, K, d_model] (or None). When provided, the K seed vectors
        are prepended as virtual positions 0..K-1 that every real token can
        attend to (causally). The real tokens keep their own positional
        embeddings (arange(T)); the seed vectors carry no token-position
        embedding -- they are hand-off signals from another expert, not tokens.

        Returns the full last-layer hidden states:
          - shape [B, K + T, d_model] when a seed is given (index 0..K-1 are
            the seed outputs; K.. are the real-token outputs), or
          - shape [B, T, d_model] when seed is None.

        Keeping the real tokens' positions at arange(T) (rather than shifting
        them by K) means the positional-embedding table is never indexed past
        max_seq_len - 1, so long segments cannot cause an out-of-range lookup.
        """
        x = self._embed(ids)  # [B, T, d]; real tokens keep positions arange(T)
        if seed is not None:
            if seed.dim() == 2:
                seed = seed.unsqueeze(1)  # [B, d] -> [B, 1, d]
            x = torch.cat([seed, x], dim=1)  # [B, K + T, d]
        return self._blocks(x)

    def encode_with_cross_attn(
        self, ids: torch.Tensor, memory: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode ids, optionally cross-attending to external carried memory.

        CALM-style bridge: instead of (or in addition to) prepending the
        carried states as virtual seed positions, the expert's own hidden
        states are used as queries that attend to the *other* expert's
        carried states (keys/values). This gives every position a learned,
        content-addressable channel to all K carried sender states.

        `memory` is [B, S, d_model] -- the carried states ALREADY projected
        back into this expert's d_model via `from_shared_space`. When None,
        this falls back to a plain encode(ids).

        The cross-attention is applied AFTER the transformer blocks (so the
        expert first forms its own representation, then refines it by
        attending to the sender's memory -- mirroring CALM, where a deeper
        layer's state informs an earlier/auxiliary prediction).

        When `shared.cross_attn_residual` is True AND a seed is also being
        used, callers may combine both mechanisms; this method only handles
        the cross-attention path. Returns [B, T, d_model] (no seed prefix).
        """
        x = self._embed(ids)  # [B, T, d]
        h = self._blocks(x)   # [B, T, d]  (own representation)
        if memory is not None and self.cross_attn is not None:
            if memory.dim() == 2:
                memory = memory.unsqueeze(1)  # [B, d] -> [B, 1, d]
            h = self.cross_attn(h, memory)    # [B, T, d]
        return h

    def logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Map a hidden state (this expert's space) to its vocab logits."""
        return self.head(h)

    # Shared-space projections ------------------------------------------------
    def to_shared_space(self, h: torch.Tensor) -> torch.Tensor:
        """h [.., d_model] -> z [.., shared_dim]."""
        return self.to_shared(h)

    def from_shared_space(self, z: torch.Tensor) -> torch.Tensor:
        """z [.., shared_dim] -> h [.., d_model]."""
        return self.from_shared(z)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def next_token_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits for the *last* position only — used in generation."""
        h = self.encode(ids)
        return self.head(h[:, -1, :])

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
