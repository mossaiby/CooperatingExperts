"""Central configuration for the Cooperating Experts test framework.

Single source of truth for every hyper-parameter. The DEFAULT config below is
the one that all the documentation and the reported results refer to:

  - decoder-only transformer: d_model=512, 6 layers, 8 heads, d_ff=2048
  - context window: 512 tokens
  - shared latent bridge: dim=256  (bottleneck < d_model forces compression)
  - bridge width: 1 hidden state carried across a hand-off (see SharedSpaceConfig.bridge_len)
  - per-expert BPE vocab capped at 12 000 merges; on the synthetic corpus the
    trainer actually finds ~5.6k (python) / ~5.7k (english) merges, so each
    expert is ~22.3 M params -> ~44.6 M total. fp16 weights ~= 90 MB.
  - mixed-precision (fp16) training throughout.

VRAM budget for this default (one expert training at a time, batch=8, seq=512):
  weights fp16  : ~45 MB
  Adam states   : ~180 MB (fp32 m + v)
  activations   : ~0.8 GB (fp16 via AMP)
  gradients     : ~90 MB
  other expert  : ~45 MB (frozen, fp16)
  -------------
  Total         : ~1.2 GB  -> comfortable on a 4 GB laptop GPU (RTX 3050 Ti).

To scale up for an 11 GB GPU (RTX 2080 / 3060) use `Config.large()`, which sets
d_model=768, 8 layers, 12 heads, d_ff=3072, max_seq_len=1024, shared dim=384
(~63-67 M params/expert, ~125-134 M total, depending on realized vocab).
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
    """Configuration for a single expert.

    Defaults are the laptop-friendly (4 GB VRAM) sizes. `Config.large()`
    overrides these for an 11 GB GPU.
    """

    name: str
    # Tokenizer merge cap. The BPE trainer stops at the actual number of
    # unique merge pairs it finds, so setting this above the real vocab just
    # wastes a little tokenizer-training time. 12 000 is comfortably above
    # what the synthetic corpus (or a ~20k-function real corpus) produces.
    vocab_size: int = 12_000

    # Transformer (default: ~22.3 M params/expert with the ~5.6k real vocab).
    d_model:    int   = 512
    n_heads:    int   = 8
    n_layers:   int   = 6
    d_ff:       int   = 2048
    max_seq_len: int  = 512
    dropout:    float = 0.1

    # Special tokens
    pad_token:            str = "<pad>"
    eos_token:            str = "<eos>"
    switch_token_template: str = "<switch:{name}>"


@dataclass
class PretrainOverride:
    """Per-expert overrides for pre-training hyper-parameters.

    Any field left as None falls back to the global TrainConfig default, so
    you only list the values you want to change for a specific expert. This
    exists because the two experts can converge at different rates, so a
    single shared schedule may under-train one and over-train the other.
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

    `dim < d_model` enforces an information bottleneck: the projections must
    compress each expert's hidden state into a smaller space, which prevents
    `to_shared`/`from_shared` from collapsing to identity. dim = d_model // 2
    = 256 is the default for d_model = 512.

    `bridge_len` is how many hidden states are carried across a hand-off. The
    original design carried a single vector (bridge_len=1). Because a single
    256-d vector is a very narrow channel, this is now configurable: setting
    bridge_len=K carries the last K hidden states of the sending expert
    through the shared space, giving the receiving expert a short sequence of
    seed positions to attend to instead of one. K=1 reproduces the original
    behaviour exactly.
    """
    dim: int = 256
    bridge_len: int = 1


@dataclass
class TrainConfig:
    """Training hyper-parameters -- all phases."""

    # -- Pre-training (per expert, independent) --------------------------
    pretrain_batch_size:   int   = 8
    pretrain_lr:           float = 2e-4
    # Early stopping (patience below) halts at the val minimum regardless, so
    # a generous ceiling is safe.
    pretrain_steps_max:    int   = 5_000
    pretrain_warmup_steps: int   = 100
    pretrain_min_lr:       float = 1e-5
    pretrain_weight_decay: float = 0.01
    pretrain_grad_clip:    float = 1.0
    # Max fixed-length windows sampled from each expert's corpus.
    pretrain_max_windows:  int   = 100_000
    # Validation: hold out this fraction of windows for an in-loop val check.
    pretrain_val_frac:     float = 0.1
    # Log validation loss every N optimizer steps during pre-training.
    pretrain_val_every:    int   = 200
    # Number of val batches averaged per check. The val estimate is averaged
    # over this many batches (capped by the val split size) so early-stopping
    # decisions are not made off a single noisy batch.
    pretrain_val_batches:  int   = 20
    # Early stopping: if val loss hasn't improved by at least
    # pretrain_val_min_delta for this many consecutive val checks, stop.
    # 0 disables early stopping (run the full steps_max).
    pretrain_early_stop_patience: int   = 3
    pretrain_val_min_delta:       float = 0.0
    # Per-expert overrides keyed by expert name. See Config.default().
    pretrain_overrides:   Dict[str, PretrainOverride] = field(default_factory=dict)

    # -- Joint projection fine-tuning ("stitching" phase) ----------------
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

    # -- Interleaved end-to-end mixed training ---------------------------
    mixed_batch_size:  int   = 1          # variable-length segments; use grad accum
    mixed_grad_accum:  int   = 16
    # The full ~44.6 M model is pre-trained and fragile, so the mixed phase is
    # fine-tuning, not training from scratch. 3e-5 caused monotonic divergence;
    # 1e-5 is stable.
    mixed_lr:          float = 1e-5
    mixed_steps_max:   int   = 5_000
    mixed_warmup_steps: int  = 300
    mixed_min_lr:      float = 5e-6
    mixed_weight_decay: float = 0.01
    # Tighter clip than 1.0: variable-length segments + batch_size=1 produce
    # occasional large gradients that destabilise the full model.
    mixed_grad_clip:   float = 0.5

    # Down-weight switch tokens in mixed_loss so the model isn't equally
    # rewarded for switching as for generating real content. Values in (0, 1);
    # lower = stronger suppression of over-switching.
    switch_loss_weight: float = 0.1

    # -- Mixed-phase early stopping & best-checkpoint saving -------------
    # Evaluate the monitor every N optimizer steps, using the smoothed running
    # average over the last `mixed_val_every` steps (batch=1 is noisy).
    mixed_val_every:        int   = 100
    mixed_early_stop_patience: int = 10
    mixed_val_min_delta:       float = 0.0
    # Save the best checkpoint (lowest smoothed loss) to model_final.pt.
    mixed_save_best:        bool  = True

    # -- Data / corpus ---------------------------------------------------
    n_sessions:           int = 10_000
    mixed_max_sessions:   int = 10_000
    extract_max_files:    int = 12_000
    extract_max_sessions: int = 10_000
    extract_shard_size:   int = 1_000
    window_max_windows:   int = 100_000
    max_examples:         int = 20_000
    # Max functions to extract from the real code corpus (download_code_corpus.py).
    code_corpus_max_functions: int = 20_000
    # LLM data generation knobs (generate_llm_data.py).
    llm_max_tokens:       int   = 1024
    llm_temperature:      float = 0.9

    # -- Misc ------------------------------------------------------------
    fp16:        bool = True
    seed:        int  = 42
    log_every:   int  = 50
    num_workers: int  = 0   # Windows-friendly default; set to 2-4 on Linux


@dataclass
class GenConfig:
    """Generation (inference) hyper-parameters."""

    max_new_tokens: int   = 80
    temperature:    float = 0.8
    top_k:          int   = 40
    # Hard cap on expert switches per generate() call. Prevents the
    # rapid-cycling failure mode observed when switch tokens are over-predicted.
    max_switches:   int   = 4


@dataclass
class Config:
    experts: dict             = field(default_factory=dict)
    shared:  SharedSpaceConfig = field(default_factory=SharedSpaceConfig)
    train:   TrainConfig      = field(default_factory=TrainConfig)
    gen:     GenConfig        = field(default_factory=GenConfig)

    @classmethod
    def default(cls) -> "Config":
        """Two experts (Python code + English prose), laptop (4 GB) sizes.

        Both experts start from the global pre-train schedule and rely on
        early stopping to halt at their val minimum. The per-expert overrides
        only tweak warmup/LR: python (code) uses the global 2e-4 LR with a
        100-step warmup; english (prose) gets a slightly lower LR and a longer
        warmup since prose is noisier.
        """
        cfg = cls()
        cfg.experts = {
            "python":  ExpertConfig(name="python"),
            "english": ExpertConfig(name="english"),
        }
        cfg.train.pretrain_overrides = {
            "python":  PretrainOverride(warmup_steps=100),
            "english": PretrainOverride(warmup_steps=200, lr=1.5e-4),
        }
        return cfg

    @classmethod
    def large(cls) -> "Config":
        """Larger preset for an 11 GB GPU (~63-67 M params/expert, ~125-134 M total, depending on realized vocab)."""
        cfg = cls.default()
        big = dict(d_model=768, n_heads=12, n_layers=8, d_ff=3072, max_seq_len=1024)
        cfg.experts = {
            "python":  ExpertConfig(name="python", **big),
            "english": ExpertConfig(name="english", **big),
        }
        cfg.shared = SharedSpaceConfig(dim=384, bridge_len=cfg.shared.bridge_len)
        return cfg


# Special tokens appended to every expert's vocabulary:
#   <unk>, <pad>, <eos>, and one <switch:NAME> per expert.
# For the default 2-expert config: 3 + 2 = 5 special tokens per expert.
# (A "self" switch is <switch:NAME> where NAME is the expert's own name; it is
# masked out during generation, so there is no separate <switch:self> token.)
NUM_SPECIAL_TOKENS_PER_EXPERT = 5
