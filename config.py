"""Central configuration for the Cooperating Experts test framework.

Architecture (sized for 11 GB VRAM — RTX 2080 / RTX 3060 class):
  - decoder-only transformer: d_model=768, 8 layers, 12 heads, d_ff=3072
  - shared latent space: dim=384  (bottleneck < d_model forces meaningful compression)
  - per-expert BPE vocab capped at 12 000 merges (realistic for a 20k-function corpus)
  - context window: 1024 tokens
  - mixed-precision (fp16) training throughout
  - ~60M params per expert, ~120M total

VRAM budget (one expert training at a time, batch=8, seq=1024):
  weights fp16  : ~120 MB
  Adam states   : ~480 MB (fp32 m + v)
  activations   : ~1.5 GB (fp16 via AMP)
  gradients     : ~240 MB
  other expert  : ~120 MB (frozen, fp16)
  ─────────────
  Total         : ~2.5 GB  (comfortable inside 11 GB; room for larger batches)

For 4 GB GPUs, downgrade to d_model=512, n_layers=6, n_heads=8, d_ff=2048,
max_seq_len=512, shared dim=256 (~22M params/expert).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
CACHE_DIR = ROOT / ".cache"


@dataclass
class ExpertConfig:
    """Configuration for a single expert."""

    name: str
    # Tokenizer — realistic upper bound for a real code corpus.
    # The BPE trainer will stop at the actual number of unique merge pairs
    # it finds, so setting this too high wastes tokenizer-training time but
    # doesn't break anything. 12 000 is comfortably above what a 20k-function
    # corpus produces while leaving room to grow.
    vocab_size: int = 12_000

    # Transformer (sized for 11 GB VRAM — RTX 2080 / RTX 3060 class).
    # d_model=768, 8 layers, 12 heads → ~60M params/expert (~120M total).
    # This gives the model enough capacity to learn real code structure from
    # a diverse corpus, while leaving VRAM headroom for longer sequences and
    # larger batches. Downgrade to d_model=512/6L/8 heads for 4 GB GPUs.
    d_model:    int   = 768
    n_heads:    int   = 12
    n_layers:   int   = 8
    d_ff:       int   = 3072
    max_seq_len: int  = 1024
    dropout:    float = 0.1

    # Special tokens
    pad_token:            str = "<pad>"
    eos_token:            str = "<eos>"
    switch_token_template: str = "<switch:{name}>"


@dataclass
class PretrainOverride:
    """Per-expert overrides for pre-training hyper-parameters.

    Any field left as None falls back to the global TrainConfig default, so
    you only need to list the values you want to change for a specific expert.
    This exists because the two experts converge at very different rates
    (templated code fits fast; free-form prose is much slower), so a single
    shared `pretrain_steps_max` either under-trains the hard expert or
    over-trains the easy one.
    """
    steps_max:    int   = None
    lr:           float = None
    warmup_steps: int   = None
    min_lr:       float = None
    weight_decay: float = None
    grad_clip:    float = None
    max_windows:  int   = None
    batch_size:   int   = None
    val_frac:     float = None
    val_every:    int   = None


@dataclass
class SharedSpaceConfig:
    """Configuration of the shared latent bridge space.

    dim < d_model enforces an information bottleneck: the projection must
    compress each expert's hidden state into a smaller space, which prevents
    the to_shared / from_shared matrices from collapsing to identity and
    forces them to learn a genuinely compact inter-expert representation.
    dim = d_model // 2 = 384 is a good default for d_model = 768.
    """
    dim: int = 384


@dataclass
class TrainConfig:
    """Training hyper-parameters — all phases."""

    # ── Pre-training (per expert, independent) ─────────────────────────────
    # Batch size raised for 11 GB VRAM (was 24 for 4 GB). With d_model=768
    # and seq=1024, batch=8 fits comfortably; adjust down if OOM.
    pretrain_batch_size:   int   = 8
    pretrain_lr:           float = 2e-4
    # With a real code corpus (20k functions) the model can't memorize in a
    # few hundred steps, so we allow more. Early stopping (patience 3) will
    # halt at the val minimum regardless.
    pretrain_steps_max:    int   = 5_000
    pretrain_warmup_steps: int   = 300
    pretrain_min_lr:       float = 1e-5
    pretrain_weight_decay: float = 0.01
    pretrain_grad_clip:    float = 1.0
    # Max fixed-length windows sampled from each expert's corpus.
    pretrain_max_windows:  int   = 100_000
    # Validation: hold out this fraction of windows for an in-loop val check.
    pretrain_val_frac:     float = 0.1
    # Log validation loss every N optimizer steps during pre-training.
    pretrain_val_every:    int   = 200
    # Early stopping: if val loss hasn't improved by at least
    # pretrain_val_min_delta for this many consecutive val checks, stop.
    # 0 disables early stopping (run the full steps_max). This auto-catches
    # the val minimum regardless of the configured step count, so an expert
    # never wastes steps memorising past its generalisation ceiling.
    pretrain_early_stop_patience: int   = 3
    pretrain_val_min_delta:       float = 0.0
    # Per-expert overrides keyed by expert name. Any field left as None in a
    # PretrainOverride falls back to the global default above. Use this to
    # give a harder-to-fit expert more steps or a different LR without
    # affecting the others. See Config.default() for the python/english split.
    pretrain_overrides:   Dict[str, PretrainOverride] = field(default_factory=dict)

    # ── Joint projection fine-tuning ("stitching" phase) ───────────────────
    joint_epochs:      int   = 4
    joint_batch_size:  int   = 12
    joint_lr:          float = 1e-4
    joint_steps_max:   int   = 3_000
    joint_warmup_steps: int  = 100
    joint_min_lr:      float = 1e-5
    joint_weight_decay: float = 0.0
    joint_grad_clip:   float = 1.0
    joint_max_pairs:   int   = 8_000
    # Weight of the alignment regulariser in joint_loss.
    align_weight:      float = 0.1

    # ── Interleaved end-to-end mixed training ───────────────────────────────
    mixed_batch_size:  int   = 1          # variable-length segments; use grad accum
    mixed_grad_accum:  int   = 16         # was 8; larger effective batch → more stable
    # Lowered from 3e-5: the full 44.6M model is pre-trained and fragile, so
    # this is fine-tuning, not training from scratch. 3e-5 caused monotonic
    # divergence (loss 1.40 → 2.12 over 1500 steps) by eroding the pre-trained
    # weights faster than the switch mechanics could stabilise.
    mixed_lr:          float = 1e-5
    mixed_steps_max:   int   = 5_000      # was 3 000; more data → more useful steps
    # Longer warmup for the large unfrozen model: 100 steps let the LR spike
    # before the projections/LM heads have re-aligned, which kicked off the
    # divergence. 300 lets gradients settle in gently.
    mixed_warmup_steps: int  = 300
    mixed_min_lr:      float = 5e-6
    mixed_weight_decay: float = 0.01
    # Tighter clip than 1.0: variable-length segments + batch_size=1 produce
    # occasional large gradients that destabilise the full model.
    mixed_grad_clip:   float = 0.5

    # Down-weight switch tokens in mixed_loss so the model isn't equally
    # rewarded for switching as for generating real content. Lowered from 0.3
    # to 0.1 so the LM signal dominates early; the switch head stabilises
    # once the LM heads are solid, instead of destabilising routing first.
    # Values in (0, 1); lower = stronger suppression of over-switching.
    switch_loss_weight: float = 0.1

    # ── Mixed-phase early stopping & best-checkpoint saving ────────────────
    # The mixed phase is slow (batch=1, ~0.1 it/s on a 4 GB GPU) and the loss
    # can plateau or rise after its minimum (observed: min ~0.86 at step 1650,
    # then slow rise to 0.89 by step 3050). These knobs save the best model
    # seen so far and halt training once the loss stops improving, so you
    # never waste hours past the minimum.
    # Evaluate the monitor every N optimizer steps (use the smoothed running
    # average over the last `mixed_val_every` steps, since batch=1 is noisy).
    mixed_val_every:        int   = 100
    # Early stopping: stop if the smoothed loss hasn't improved by at least
    # mixed_val_min_delta for this many consecutive checks. 0 disables.
    mixed_early_stop_patience: int   = 10
    mixed_val_min_delta:       float = 0.0
    # Save the best checkpoint (lowest smoothed loss) to model_final.pt
    # whenever a new best is found. True by default — the mixed phase is slow
    # and you want the best model, not necessarily the last.
    mixed_save_best:        bool  = True

    # ── Data / corpus ───────────────────────────────────────────────────────
    # Number of sessions to generate/use. With the hybrid pipeline (real code
    # + LLM prose) 10k–50k sessions is the sweet spot for a 60M-param model.
    n_sessions:           int = 10_000
    mixed_max_sessions:   int = 10_000
    extract_max_files:    int = 12_000
    extract_max_sessions: int = 10_000
    extract_shard_size:   int = 1_000
    # Max windows/examples — raised to match the larger corpus.
    window_max_windows:   int = 100_000
    max_examples:         int = 20_000
    # Max functions to extract from the real code corpus (download_code_corpus.py).
    code_corpus_max_functions: int = 20_000
    # LLM data generation knobs (generate_llm_data.py).
    llm_max_tokens:       int   = 1024
    llm_temperature:      float = 0.9

    # ── Misc ────────────────────────────────────────────────────────────────
    fp16:        bool = True
    seed:        int  = 42
    log_every:   int  = 50
    num_workers: int  = 0   # Windows-friendly default; set to 2–4 on Linux


@dataclass
class GenConfig:
    """Generation (inference) hyper-parameters."""

    max_new_tokens: int   = 80
    temperature:    float = 0.8
    top_k:          int   = 40
    # Hard cap on expert switches per generate() call.
    # Prevents the rapid-cycling failure mode observed when switch tokens
    # are over-predicted (as seen in early runs with no budget).
    max_switches:   int   = 4


@dataclass
class Config:
    experts: dict             = field(default_factory=dict)
    shared:  SharedSpaceConfig = field(default_factory=SharedSpaceConfig)
    train:   TrainConfig      = field(default_factory=TrainConfig)
    gen:     GenConfig        = field(default_factory=GenConfig)

    @classmethod
    def default(cls) -> "Config":
        """Two experts: Python code and English prose.

        With the hybrid data pipeline (real Python code from
        download_code_corpus.py + LLM-generated prose from
        generate_llm_data.py), the corpus is genuinely diverse and the model
        can't memorize it in a few hundred steps. Both experts use the global
        pretrain_steps_max (5000) and rely on early stopping (patience 3) to
        halt at their val minimum. The per-expert overrides only tweak the
        warmup/LR: python (real code) uses the global defaults; english
        (LLM-generated prose) gets a slightly lower LR and longer warmup
        since prose is noisier than code.
        """
        cfg = cls()
        cfg.experts = {
            "python":  ExpertConfig(name="python"),
            "english": ExpertConfig(name="english"),
        }
        cfg.train.pretrain_overrides = {
            "python": PretrainOverride(
                warmup_steps=300,
            ),
            "english": PretrainOverride(
                warmup_steps=400, lr=1.5e-4,
            ),
        }
        return cfg


# Absolute special-token ids (relative to each expert's vocab) are assigned
# *after* the BPE vocab is built. The tokenizer appends these special tokens
# to every expert's vocabulary:
#   <unk>, <pad>, <eos>, and one <switch:NAME> per expert (self included).
# For the default 2-expert config: 1 + 1 + 1 + 2 = 5 special tokens.
NUM_SPECIAL_TOKENS_PER_EXPERT = 5
