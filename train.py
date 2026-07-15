"""Training loops: per-expert pre-training + joint projection fine-tuning.

Both loops are written to run on a 4 GB VRAM GPU:
  - mixed precision (torch.amp),
  - gradient accumulation is NOT needed given the tiny batch/model sizes,
  - gradient clipping for stability.
"""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from config import Config, CKPT_DIR
from cooperating import CooperatingExperts
from dataset import HandoffDataset, MixedDataset, WindowDataset, load_raw_texts
from tokenizer import ExpertTokenizer


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cosine_schedule(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    """Cosine decay with linear warmup."""
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * ratio))


# ---------------------------------------------------------------------- #
# Per-expert pre-training
# ---------------------------------------------------------------------- #
def pretrain_expert(
    model: CooperatingExperts,
    name: str,
    texts: List[str],
    tokenizer: ExpertTokenizer,
    cfg: Config,
) -> Dict[str, float]:
    """Pre-train a single expert with standard next-token LM loss."""
    device = _get_device()
    exp = model.expert(name)
    exp.to(device)
    exp.train()

    ecfg = cfg.experts[name]

    # Resolve per-expert pre-training hyper-parameters. Any field the
    # override leaves as None falls back to the global TrainConfig default,
    # so a single shared config still works and you only need to override
    # the values that differ for this expert (e.g. more steps for the harder
    # prose expert). See config.PretrainOverride.
    ov = cfg.train.pretrain_overrides.get(name)
    def _resolve(getter, global_val):
        v = getter(ov) if ov is not None else None
        return v if v is not None else global_val
    pt_steps_max    = _resolve(lambda o: o.steps_max,    cfg.train.pretrain_steps_max)
    pt_lr           = _resolve(lambda o: o.lr,           cfg.train.pretrain_lr)
    pt_warmup       = _resolve(lambda o: o.warmup_steps, cfg.train.pretrain_warmup_steps)
    pt_min_lr       = _resolve(lambda o: o.min_lr,       cfg.train.pretrain_min_lr)
    pt_weight_decay = _resolve(lambda o: o.weight_decay, cfg.train.pretrain_weight_decay)
    pt_grad_clip    = _resolve(lambda o: o.grad_clip,    cfg.train.pretrain_grad_clip)
    pt_max_windows  = _resolve(lambda o: o.max_windows,  cfg.train.pretrain_max_windows)
    pt_batch_size   = _resolve(lambda o: o.batch_size,   cfg.train.pretrain_batch_size)
    pt_val_frac     = _resolve(lambda o: o.val_frac,     cfg.train.pretrain_val_frac)
    pt_val_every    = _resolve(lambda o: o.val_every,    cfg.train.pretrain_val_every)

    ds = WindowDataset(texts, tokenizer, ecfg.max_seq_len,
                       max_windows=pt_max_windows)
    if len(ds) == 0:
        print(f"  [{name}] no training data, skipping")
        return {"loss": float("nan"), "steps": 0}

    # Hold out a validation split so we can watch for over-fitting. The
    # training loop cycles the train loader; the val set is evaluated
    # periodically (every pt_val_every steps) without back-prop.
    n_val = max(1, int(len(ds) * pt_val_frac))
    n_val = min(n_val, len(ds) - 1)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )
    # Validation loader: size the batch to fit the (possibly tiny) val split
    # and never drop the last partial batch, so the loader always yields at
    # least one batch even when n_val < pt_batch_size. Without this, a
    # small val set + drop_last=True produces an empty loader and the first
    # next(val_iter) raises StopIteration (crashed the english expert).
    val_batch = min(pt_batch_size, n_val)
    val_loader = DataLoader(
        val_ds, batch_size=val_batch,
        shuffle=False, num_workers=cfg.train.num_workers, drop_last=False,
    )
    val_iter = itertools.cycle(val_loader) if len(val_loader) > 0 else None

    loader = DataLoader(
        train_ds,
        batch_size=pt_batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        drop_last=True,
    )

    # Only train this expert's parameters.
    opt = torch.optim.AdamW(exp.parameters(), lr=pt_lr,
                            weight_decay=pt_weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.fp16 and device.type == "cuda")

    step = 0
    running = 0.0
    t0 = time.time()
    # Early-stopping state: track the best val loss and how many consecutive
    # val checks have failed to improve it. If patience is exceeded we stop
    # before steps_max, so an expert never wastes steps memorising past its
    # generalisation ceiling (see config.pretrain_early_stop_patience).
    best_val = float("inf")
    bad_checks = 0
    patience = cfg.train.pretrain_early_stop_patience
    min_delta = cfg.train.pretrain_val_min_delta
    # Step-bounded: cycle through the (small) dataset until we hit
    # pt_steps_max, rather than stopping after a fixed number of epochs.
    for batch in itertools.cycle(loader):
        batch = batch.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg.train.fp16 and device.type == "cuda"):
            loss = model.pretrain_loss(name, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(exp.parameters(), pt_grad_clip)
        scaler.step(opt)
        scaler.update()
        # Cosine LR schedule with warmup.
        for g in opt.param_groups:
            g["lr"] = _cosine_schedule(
                step, pt_steps_max, pt_warmup, pt_lr, pt_min_lr,
            )

        running += loss.item()
        step += 1
        if step % cfg.train.log_every == 0:
            avg = running / cfg.train.log_every
            running = 0.0
            elapsed = time.time() - t0
            print(
                f"  [{name}] step {step:5d} | loss {avg:.4f} | "
                f"lr {opt.param_groups[0]['lr']:.2e} | {step/elapsed:.1f} it/s"
            )
        # Periodic validation: evaluate the held-out windows without
        # back-prop. A rising val loss while train loss falls is the
        # definitive over-fitting signal (see PLAN.md). Skipped if the val
        # set is too small to form even one batch. Early stopping halts
        # training if val loss hasn't improved for `patience` consecutive
        # checks, so the expert stops near its val minimum.
        if step % pt_val_every == 0 and val_iter is not None:
            exp.eval()
            with torch.no_grad():
                vbatch = next(val_iter).to(device)
                with torch.amp.autocast("cuda", enabled=cfg.train.fp16 and device.type == "cuda"):
                    vloss = model.pretrain_loss(name, vbatch)
            exp.train()
            v = vloss.item()
            print(f"  [{name}] step {step:5d} | val_loss {v:.4f}")
            if patience > 0:
                if v < best_val - min_delta:
                    best_val = v
                    bad_checks = 0
                else:
                    bad_checks += 1
                    if bad_checks >= patience:
                        print(f"  [{name}] early stopping at step {step} "
                              f"(no val improvement for {patience} checks, "
                              f"best={best_val:.4f})")
                        break
        if step >= pt_steps_max:
            break

    print(f"  [{name}] pre-training done: {step} steps")
    return {"loss": running, "steps": step}


# ---------------------------------------------------------------------- #
# Joint fine-tuning of projection layers
# ---------------------------------------------------------------------- #
def joint_finetune(
    model: CooperatingExperts,
    texts: Dict[str, List[str]],
    tokenizers: Dict[str, ExpertTokenizer],
    cfg: Config,
) -> Dict[str, float]:
    """Fine-tune the shared-space projections jointly across experts.

    We freeze the transformer blocks and only train the `to_shared` /
    `from_shared` linear layers, plus the LM heads stay as-is. This is the
    "stitching" phase: the experts are already good at their own languages,
    we just need to align their latent spaces at the boundaries.
    """
    device = _get_device()
    model.to(device)

    # Freeze everything except projection layers.
    for n, exp in model.experts.items():
        for p in exp.parameters():
            p.requires_grad = False
        exp.to_shared.weight.requires_grad = True
        exp.from_shared.weight.requires_grad = True

    proj_params = []
    for n in model.expert_names:
        proj_params.extend(
            [model.expert(n).to_shared.weight, model.expert(n).from_shared.weight]
        )
    opt = torch.optim.AdamW(proj_params, lr=cfg.train.joint_lr,
                            weight_decay=cfg.train.joint_weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.fp16 and device.type == "cuda")

    names = model.expert_names
    # Build hand-off datasets in both directions: A->B and B->A.
    seq_len = cfg.experts[names[0]].max_seq_len
    pairs_ab = HandoffDataset(
        texts[names[0]], tokenizers[names[0]],
        texts[names[1]], tokenizers[names[1]],
        seq_len, max_pairs=cfg.train.joint_max_pairs,
    )
    pairs_ba = HandoffDataset(
        texts[names[1]], tokenizers[names[1]],
        texts[names[0]], tokenizers[names[0]],
        seq_len, max_pairs=cfg.train.joint_max_pairs,
    )

    loader_ab = DataLoader(pairs_ab, batch_size=cfg.train.joint_batch_size,
                           shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)
    loader_ba = DataLoader(pairs_ba, batch_size=cfg.train.joint_batch_size,
                           shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)

    step = 0
    running_lm = 0.0
    running_align = 0.0
    t0 = time.time()
    done = False
    for epoch in range(cfg.train.joint_epochs):
        if done:
            break
        for (ab, ba) in zip(itertools.cycle(loader_ab), itertools.cycle(loader_ba)):
            ids_a, ids_b = ab
            ids_b2, ids_a2 = ba
            ids_a = ids_a.to(device); ids_b = ids_b.to(device)
            ids_a2 = ids_a2.to(device); ids_b2 = ids_b2.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.train.fp16 and device.type == "cuda"):
                loss1, info1 = model.joint_loss(names[0], ids_a, names[1], ids_b)
                loss2, info2 = model.joint_loss(names[1], ids_b2, names[0], ids_a2)
                loss = loss1 + loss2
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(proj_params, cfg.train.joint_grad_clip)
            scaler.step(opt)
            scaler.update()
            # Cosine LR schedule with warmup.
            for g in opt.param_groups:
                g["lr"] = _cosine_schedule(
                    step, cfg.train.joint_steps_max, cfg.train.joint_warmup_steps,
                    cfg.train.joint_lr, cfg.train.joint_min_lr,
                )

            running_lm += (info1["lm"] + info2["lm"]) / 2
            running_align += (info1["align"] + info2["align"]) / 2
            step += 1
            if step % cfg.train.log_every == 0:
                n = cfg.train.log_every
                print(
                    f"  [joint] step {step:5d} | lm {running_lm/n:.4f} | "
                    f"align {running_align/n:.6f} | {step/(time.time()-t0):.1f} it/s"
                )
                running_lm = 0.0
                running_align = 0.0
            if step >= cfg.train.joint_steps_max:
                done = True
                break

    # Unfreeze everything again for saving / later use.
    for n, exp in model.experts.items():
        for p in exp.parameters():
            p.requires_grad = True

    print(f"  [joint] fine-tuning done: {step} steps")
    return {"lm": running_lm, "align": running_align, "steps": step}


# ---------------------------------------------------------------------- #
# Interleaved joint training on the whole data (both experts unfrozen)
# ---------------------------------------------------------------------- #
def train_mixed(
    model: CooperatingExperts,
    tokenizers: Dict[str, ExpertTokenizer],
    cfg: Config,
    max_sessions: int = None,
) -> Dict[str, float]:
    """Train both experts end-to-end on interleaved synthetic sessions.

    Each session is a list of (expert_name, ids) segments with switch tokens
    inserted at code<->text boundaries. We unfreeze ALL parameters and train
    with `mixed_loss`. Because segments are variable-length we use
    batch_size=1 and gradient accumulation (mixed_grad_accum).
    """
    device = _get_device()
    model.to(device)
    # Unfreeze everything for full end-to-end training.
    for p in model.parameters():
        p.requires_grad = True

    if max_sessions is None:
        max_sessions = cfg.train.mixed_max_sessions
    ds = MixedDataset(tokenizers, max_seq_len=cfg.experts["python"].max_seq_len,
                      max_sessions=max_sessions)
    if len(ds) == 0:
        print("  [mixed] no interleaved examples built, skipping")
        return {"loss": float("nan"), "steps": 0}

    loader = DataLoader(
        ds, batch_size=cfg.train.mixed_batch_size, shuffle=True,
        num_workers=cfg.train.num_workers,
        # Each example is already a list of (name, ids) segments; the default
        # collate would try to stack the (str, tensor) tuples and break the
        # structure. Return the example list verbatim.
        collate_fn=lambda batch: batch[0],
    )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.mixed_lr, weight_decay=cfg.train.mixed_weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.fp16 and device.type == "cuda")

    accum = cfg.train.mixed_grad_accum
    step = 0
    batch_idx = 0
    running = 0.0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)

    # Early-stopping / best-checkpoint state. The mixed phase is slow and
    # batch=1 is noisy, so we monitor a SMOOTHED running average (over the
    # last `mixed_val_every` steps) rather than a single noisy step loss.
    # We keep the best (lowest) smoothed loss seen so far, save the model to
    # model_final.pt whenever a new best appears, and halt once the smoothed
    # loss fails to improve for `mixed_early_stop_patience` checks.
    mv_every = max(1, cfg.train.mixed_val_every)
    patience = cfg.train.mixed_early_stop_patience
    min_delta = cfg.train.mixed_val_min_delta
    save_best = cfg.train.mixed_save_best
    best_loss = float("inf")
    bad_checks = 0
    # Rolling buffer of recent per-step losses for smoothing.
    recent_losses: List[float] = []

    # Step-bounded: cycle through the sessions until we hit mixed_steps_max
    # optimizer steps (each = `accum` accumulated batches).
    for batch in itertools.cycle(loader):
        # collate_fn returns the example verbatim (a list of segments).
        segments = batch
        with torch.amp.autocast("cuda", enabled=cfg.train.fp16 and device.type == "cuda"):
            loss, info = model.mixed_loss(segments)
            loss = loss / accum
        scaler.scale(loss).backward()
        running += info["loss"]
        batch_idx += 1
        if batch_idx % accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.mixed_grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            for g in opt.param_groups:
                g["lr"] = _cosine_schedule(
                    step, cfg.train.mixed_steps_max, cfg.train.mixed_warmup_steps,
                    cfg.train.mixed_lr, cfg.train.mixed_min_lr,
                )
            step += 1
            # Track per-step loss for the smoothed monitor.
            recent_losses.append(info["loss"])
            if step % cfg.train.log_every == 0:
                avg = running / (cfg.train.log_every * accum)
                running = 0.0
                print(
                    f"  [mixed] step {step:5d} | loss {avg:.4f} | "
                    f"lr {opt.param_groups[0]['lr']:.2e} | "
                    f"{step/(time.time()-t0):.1f} it/s"
                )
            # Periodic monitor: smoothed loss over the last mv_every steps.
            if step % mv_every == 0:
                window = recent_losses[-mv_every:]
                smoothed = sum(window) / len(window)
                improved = smoothed < best_loss - min_delta
                if improved:
                    best_loss = smoothed
                    bad_checks = 0
                    if save_best:
                        save_checkpoint(model, tokenizers, cfg, tag="final")
                    print(f"  [mixed] step {step:5d} | smoothed {smoothed:.4f} "
                          f"| best {best_loss:.4f} | saved")
                else:
                    bad_checks += 1
                    print(f"  [mixed] step {step:5d} | smoothed {smoothed:.4f} "
                          f"| best {best_loss:.4f} | no improve x{bad_checks}")
                    if patience > 0 and bad_checks >= patience:
                        print(f"  [mixed] early stopping at step {step} "
                              f"(no improvement for {patience} checks, "
                              f"best={best_loss:.4f})")
                        break
            if step >= cfg.train.mixed_steps_max:
                break
    print(f"  [mixed] training done: {step} steps | best smoothed {best_loss:.4f}")
    return {"loss": running, "steps": step, "best": best_loss}


# ---------------------------------------------------------------------- #
# Checkpointing
# ---------------------------------------------------------------------- #
def save_checkpoint(model: CooperatingExperts, tokenizers: Dict[str, ExpertTokenizer],
                    cfg: Config, tag: str = "final") -> None:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / f"model_{tag}.pt")
    for name, tok in tokenizers.items():
        tok.save(CKPT_DIR / f"tokenizer_{name}.json")
    print(f"Saved checkpoint '{tag}' to {CKPT_DIR}")


def load_checkpoint(model: CooperatingExperts, tokenizers: Dict[str, ExpertTokenizer],
                    tag: str = "final", device: torch.device = None) -> CooperatingExperts:
    if device is None:
        device = _get_device()
    sd = torch.load(CKPT_DIR / f"model_{tag}.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)
    for name in tokenizers:
        tokenizers[name] = ExpertTokenizer.load(CKPT_DIR / f"tokenizer_{name}.json")
    return model
