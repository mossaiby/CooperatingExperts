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
