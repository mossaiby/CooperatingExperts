"""Evaluation utilities for the Cooperating Experts framework.

The core hypothesis of this project is that *routing between two small,
specialised experts* is at least as good as a single monolithic model of
comparable size on interleaved code/prose. Without a baseline this claim is
untested, so this module provides:

  1. `segment_perplexity` — held-out per-segment-type perplexity for the
     cooperating model (python segments vs. english segments), computed with
     the SAME hand-off convention used in training/generation.
  2. `Monolith` + `pretrain_monolith` + `monolith_perplexity` — a single
     decoder-only transformer with ONE shared tokenizer, trained on the
     concatenated corpus, sized to roughly the cooperating model's total
     parameter count, as an apples-to-apples baseline.

All perplexities are reported as exp(mean token NLL) over non-pad,
non-switch targets on a held-out split, so the numbers are directly
comparable across models.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config, ExpertConfig, SharedSpaceConfig
from cooperating import CooperatingExperts
from dataset import MixedDataset, WindowDataset
from model import Expert
from tokenizer import ExpertTokenizer


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------- #
# Cooperating-model per-segment perplexity
# ---------------------------------------------------------------------- #
@torch.no_grad()
def segment_perplexity(
    model: CooperatingExperts,
    tokenizers: Dict[str, ExpertTokenizer],
    cfg: Config,
    max_sessions: int = 500,
) -> Dict[str, float]:
    """Per-segment-type perplexity of the cooperating model on held-out data.

    Runs whole sessions through the model exactly as `mixed_loss` does (carried
    hidden states across switches, unified hand-off query), but accumulates the
    token NLL SEPARATELY for python and english segments and excludes switch
    tokens. Returns {"python": ppl, "english": ppl, "overall": ppl}.
    """
    device = _device()
    model.to(device)
    model.eval()

    ds = MixedDataset(tokenizers, max_seq_len=cfg.experts["python"].max_seq_len,
                      max_sessions=max_sessions)
    nll: Dict[str, float] = {"python": 0.0, "english": 0.0}
    ntok: Dict[str, int] = {"python": 0, "english": 0}

    for idx in range(len(ds)):
        segments = ds[idx]
        carried: Optional[torch.Tensor] = None
        for seg_idx, (name, ids) in enumerate(segments):
            exp = model.expert(name)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            T = ids.size(1)
            if T > exp.cfg.max_seq_len:
                ids = ids[:, -exp.cfg.max_seq_len:]

            logits, targets, h = model._encode_receiver(name, exp, ids, carried)
            pad = tokenizers[name].pad_id
            switch_ids = {tokenizers[name].switch_id(n) for n in model.expert_names}

            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            per_tok = F.cross_entropy(
                flat_logits, flat_targets, ignore_index=pad, reduction="none",
            )
            keep = flat_targets != pad
            for sid in switch_ids:
                keep &= flat_targets != sid
            nll[name] += float(per_tok[keep].sum().item())
            ntok[name] += int(keep.sum().item())

            # Carry to next segment (detached), mirroring mixed_loss.
            nxt = segments[seg_idx + 1][0] if seg_idx + 1 < len(segments) else None
            if nxt is not None:
                K = cfg.shared.bridge_len
                k = min(K, h.size(1))
                carried = model._carry_through_shared(
                    exp, h[:, -k:, :], model.expert(nxt), detach=True,
                )
            else:
                carried = None

    out: Dict[str, float] = {}
    tot_nll, tot_tok = 0.0, 0
    for name in ("python", "english"):
        if ntok[name] > 0:
            out[name] = math.exp(nll[name] / ntok[name])
            tot_nll += nll[name]
            tot_tok += ntok[name]
        else:
            out[name] = float("nan")
    out["overall"] = math.exp(tot_nll / tot_tok) if tot_tok > 0 else float("nan")
    return out


# ---------------------------------------------------------------------- #
# Monolithic single-model baseline
# ---------------------------------------------------------------------- #
class Monolith(torch.nn.Module):
    """A single decoder-only transformer over ONE shared vocabulary.

    Reuses the `Expert` architecture (RoPE, tied head) with a shared
    tokenizer, so it is the natural "no routing" control for the cooperating
    model. Size it via `d_model`/`n_layers` to match the cooperating model's
    total parameter count for a fair comparison.
    """

    def __init__(self, cfg: ExpertConfig, vocab_size: int):
        super().__init__()
        # Cross-attention / shared bridge are irrelevant here; pass a disabled
        # SharedSpaceConfig so the Expert builds only its transformer stack.
        self.net = Expert(cfg, SharedSpaceConfig(dim=cfg.d_model, cross_attn=False),
                          vocab_size)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(ids)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pretrain_monolith(
    tokenizer: ExpertTokenizer,
    texts: List[str],
    cfg: Config,
    d_model: int = 640,
    n_layers: int = 6,
    steps_max: int = 3000,
) -> Monolith:
    """Train a monolithic baseline on the concatenated corpus.

    `texts` should be the code and prose examples concatenated (shuffled), and
    `tokenizer` a single BPE tokenizer trained on the combined corpus. The
    default width (d_model=640, 6 layers) lands near the cooperating model's
    ~44.6 M total; adjust to match your realized vocab.
    """
    device = _device()
    ecfg = ExpertConfig(name="monolith", d_model=d_model, n_layers=n_layers,
                        n_heads=cfg.experts["python"].n_heads,
                        d_ff=4 * d_model,
                        max_seq_len=cfg.experts["python"].max_seq_len)
    model = Monolith(ecfg, tokenizer.vocab_size).to(device)
    model.train()
    print(f"  [monolith] {model.num_params()/1e6:.2f}M params")

    ds = WindowDataset(texts, tokenizer, ecfg.max_seq_len)
    if len(ds) == 0:
        raise RuntimeError("monolith: empty training set")
    loader = DataLoader(ds, batch_size=cfg.train.pretrain_batch_size,
                        shuffle=True, num_workers=cfg.train.num_workers,
                        drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.pretrain_lr,
                            weight_decay=cfg.train.pretrain_weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.fp16 and device.type == "cuda")
    pad = tokenizer.pad_id

    import itertools
    step = 0
    for batch in itertools.cycle(loader):
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg.train.fp16 and device.type == "cuda"):
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                ignore_index=pad,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.pretrain_grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        if step % cfg.train.log_every == 0:
            print(f"  [monolith] step {step:5d} | loss {loss.item():.4f}")
        if step >= steps_max:
            break
    return model


@torch.no_grad()
def monolith_perplexity(
    model: Monolith,
    tokenizer: ExpertTokenizer,
    texts: List[str],
    cfg: Config,
) -> float:
    """Held-out perplexity of the monolithic baseline on `texts`."""
    device = _device()
    model.to(device)
    model.eval()
    ds = WindowDataset(texts, tokenizer, cfg.experts["python"].max_seq_len)
    loader = DataLoader(ds, batch_size=cfg.train.pretrain_batch_size,
                        shuffle=False, num_workers=cfg.train.num_workers)
    pad = tokenizer.pad_id
    tot_nll, tot_tok = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch[:, :-1])
        targets = batch[:, 1:]
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad, reduction="none",
        )
        keep = targets.reshape(-1) != pad
        tot_nll += float(per_tok[keep].sum().item())
        tot_tok += int(keep.sum().item())
    return math.exp(tot_nll / tot_tok) if tot_tok > 0 else float("nan")
