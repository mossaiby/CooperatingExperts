"""CLI entrypoint for the Cooperating Experts test framework.

Usage:
    # --- Templated synthetic data (original pipeline) ---
    python main.py extract      # extract code/text corpora from synthetic sessions
    python main.py prepare      # extract corpora + train tokenizers
    python main.py pretrain     # pre-train each expert independently
    python main.py joint        # interleaved end-to-end training on whole data
    python main.py generate "prompt text" --expert english
    python main.py all          # extract -> prepare -> pretrain -> joint -> sample
    python main.py status       # show what checkpoints exist

    # --- Hybrid data pipeline (real code + LLM prose) ---
    python main.py download         # download real Python code corpus (CodeSearchNet)
    python main.py gen-data --n 10000   # wrap real code in LLM-generated prose -> sessions
    python main.py prepare-hybrid   # download + gen-data + train tokenizers
    python main.py all-hybrid       # prepare-hybrid -> pretrain -> joint -> mixed -> sample

Configure the LLM API in a .env file (see .env-example).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

from config import CKPT_DIR, DATA_DIR, Config
from cooperating import CooperatingExperts
from dataset import configure_socks_proxy, extract_synthetic_corpora, load_raw_texts
from synthetic_data import write_combined, write_synthetic
from tokenizer import ExpertTokenizer, build_tokenizer
from train import joint_finetune, load_checkpoint, pretrain_expert, save_checkpoint, train_mixed


# ---------------------------------------------------------------------- #
def _all_expert_names(cfg: Config) -> List[str]:
    return list(cfg.experts.keys())


def prepare(cfg: Config) -> Dict[str, ExpertTokenizer]:
    """Extract synthetic corpora and train per-expert tokenizers."""
    names = _all_expert_names(cfg)
    tokenizers: Dict[str, ExpertTokenizer] = {}

    # 1. Generate (or refresh) the raw synthetic sessions, then extract
    # code/text corpora from them. We regenerate explicitly so the raw
    # directory always matches cfg.train.n_sessions (the loader only
    # auto-generates when the directory is missing, so a stale directory
    # from a previous run would otherwise be reused).
    print("== Preparing corpora (synthetic) ==")
    write_synthetic(n=cfg.train.n_sessions)
    extract_synthetic_corpora(DATA_DIR, cfg=cfg)
    # 1b. Also write the combined (whole-session) view with switch markers,
    # so switch tokens can be extracted from the full sessions, not only the
    # split corpora.
    write_combined(n=cfg.train.n_sessions)

    # 2. Train tokenizers (each on its own corpus).
    print("== Training tokenizers ==")
    import glob
    for name in names:
        shards = sorted(glob.glob(str(DATA_DIR / f"{name}_corpus_*.txt")))
        if not shards:
            print(f"  [{name}] no corpus shards found, skipping")
            continue
        tok = build_tokenizer(cfg.experts[name], [Path(s) for s in shards], names)
        tokenizers[name] = tok
        print(f"  [{name}] vocab size = {tok.vocab_size}")

    # Save tokenizers.
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tok in tokenizers.items():
        tok.save(CKPT_DIR / f"tokenizer_{name}.json")

    # The split *corpus_* shards are only intermediate inputs for tokenizer
    # training; once the tokenizers are saved they are no longer needed, so
    # remove them to avoid stale/accumulating files in data/.
    import glob as _glob
    for _f in _glob.glob(str(DATA_DIR / "*_corpus_*.txt")):
        try:
            Path(_f).unlink()
        except OSError:
            pass
    print("  [prepare] removed intermediate corpus shards (tokenizers saved)")
    return tokenizers


def cmd_extract(cfg: Config) -> None:
    write_synthetic(n=cfg.train.n_sessions)
    extract_synthetic_corpora(DATA_DIR, cfg=cfg)
    write_combined(n=cfg.train.n_sessions)


def _load_or_build_tokenizers(cfg: Config) -> Dict[str, ExpertTokenizer]:
    names = _all_expert_names(cfg)
    tokenizers: Dict[str, ExpertTokenizer] = {}
    all_present = True
    for name in names:
        path = CKPT_DIR / f"tokenizer_{name}.json"
        if path.exists():
            tokenizers[name] = ExpertTokenizer.load(path)
        else:
            all_present = False
    if not all_present or len(tokenizers) != len(names):
        print("Some tokenizers missing; rebuilding...")
        return prepare(cfg)
    return tokenizers


def _build_model(cfg: Config, tokenizers: Dict[str, ExpertTokenizer]) -> CooperatingExperts:
    model = CooperatingExperts(cfg, tokenizers)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total/1e6:.2f}M params total")
    for name in model.expert_names:
        n = model.expert(name).num_params()
        print(f"  expert '{name}': {n/1e6:.2f}M params")
    return model


# ---------------------------------------------------------------------- #
def cmd_prepare(cfg: Config) -> None:
    prepare(cfg)


def cmd_pretrain(cfg: Config) -> None:
    tokenizers = _load_or_build_tokenizers(cfg)
    model = _build_model(cfg, tokenizers)
    for name in model.expert_names:
        print(f"== Pre-training expert '{name}' ==")
        texts = load_raw_texts(name, cfg=cfg)
        pretrain_expert(model, name, texts, tokenizers[name], cfg)
    save_checkpoint(model, tokenizers, cfg, tag="pretrained")


def cmd_joint(cfg: Config) -> None:
    tokenizers = _load_or_build_tokenizers(cfg)
    model = _build_model(cfg, tokenizers)
    # Load pre-trained weights if available.
    pre_path = CKPT_DIR / "model_pretrained.pt"
    if pre_path.exists():
        sd = torch.load(pre_path, map_location="cuda" if torch.cuda.is_available() else "cpu",
                        weights_only=True)
        model.load_state_dict(sd)
        print("Loaded pre-trained expert weights.")
    else:
        print("WARNING: no pre-trained checkpoint found; fine-tuning from scratch.")

    # Phase 1: projection-only "stitching" fine-tuning. Freeze the
    # transformer blocks and only train the to_shared / from_shared linear
    # layers so the experts' latent spaces align at the boundaries before
    # the full end-to-end phase.
    print("== Joint projection fine-tuning (stitching phase) ==")
    texts = {name: load_raw_texts(name, cfg=cfg) for name in model.expert_names}
    joint_finetune(model, texts, tokenizers, cfg)
    save_checkpoint(model, tokenizers, cfg, tag="stitched")

    # Phase 2: interleaved end-to-end training on the whole synthetic
    # session data (both experts unfrozen).
    print("== Interleaved end-to-end training (mixed phase) ==")
    train_mixed(model, tokenizers, cfg, max_sessions=cfg.train.mixed_max_sessions)
    save_checkpoint(model, tokenizers, cfg, tag="final")


# ---------------------------------------------------------------------- #
# Hybrid data pipeline (real code + LLM prose)
# ---------------------------------------------------------------------- #
def cmd_download(cfg: Config) -> None:
    """Download/extract a real Python code corpus for the python expert."""
    print("== Downloading real Python code corpus ==")
    import download_code_corpus as dcc
    out_dir = DATA_DIR / "code_corpus"
    functions = dcc._download_csn(cfg.train.code_corpus_max_functions, out_dir) \
        if True else dcc._extract_from_local(Path("."), cfg.train.code_corpus_max_functions)
    if functions:
        dcc.write_corpus(functions, out_dir)


def cmd_gen_data(cfg: Config, n: int, model: str = None, clean: bool = False) -> None:
    """Generate sessions by wrapping real Python code in LLM-generated prose."""
    print("== Generating LLM-wrapped sessions (hybrid data) ==")
    import generate_llm_data as gld
    gld.generate_llm_sessions(
        n=n,
        max_tokens=cfg.train.llm_max_tokens,
        temperature=cfg.train.llm_temperature,
        seed=cfg.train.seed,
        raw_dir=DATA_DIR / "synthetic_raw",
        combined_dir=DATA_DIR / "synthetic_combined",
        model_override=model,
        clean=clean,
    )


def cmd_prepare_hybrid(cfg: Config) -> None:
    """Full hybrid data pipeline: download code -> generate LLM sessions -> train tokenizers."""
    cmd_download(cfg)
    cmd_gen_data(cfg, cfg.train.n_sessions)
    # Now train tokenizers on the hybrid corpora.
    print("== Training tokenizers (hybrid corpora) ==")
    names = _all_expert_names(cfg)
    tokenizers: Dict[str, ExpertTokenizer] = {}
    import glob
    # Use the real code corpus for the python expert's tokenizer, and the
    # LLM-generated sessions for the english expert's tokenizer.
    code_corpus = DATA_DIR / "code_corpus" / "python_corpus.txt"
    for name in names:
        if name == "python" and code_corpus.exists():
            shards = [code_corpus]
        else:
            # Fall back to extracting from the generated sessions.
            extract_synthetic_corpora(DATA_DIR, cfg=cfg)
            shards = sorted(glob.glob(str(DATA_DIR / f"{name}_corpus_*.txt")))
        if not shards:
            print(f"  [{name}] no corpus found, skipping")
            continue
        tok = build_tokenizer(cfg.experts[name], [Path(s) for s in shards], names)
        tokenizers[name] = tok
        print(f"  [{name}] vocab size = {tok.vocab_size}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, tok in tokenizers.items():
        tok.save(CKPT_DIR / f"tokenizer_{name}.json")
    print("  [prepare-hybrid] tokenizers saved")


def cmd_generate(cfg: Config, prompt: str, expert: str, max_tokens: int) -> None:
    tokenizers = _load_or_build_tokenizers(cfg)
    model = _build_model(cfg, tokenizers)
    final = CKPT_DIR / "model_final.pt"
    if final.exists():
        load_checkpoint(model, tokenizers, tag="final")
    else:
        pre = CKPT_DIR / "model_pretrained.pt"
        if pre.exists():
            load_checkpoint(model, tokenizers, tag="pretrained")
        else:
            print("No checkpoint found. Run `pretrain` and `joint` first.")
            return
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"\n--- Generating (start expert: {expert}) ---")
    print(f"Prompt: {prompt!r}\n")
    out = model.generate(prompt, start_expert=expert, max_new_tokens=max_tokens)
    print(out)
    print("--- done ---\n")


def cmd_status(cfg: Config) -> None:
    print("Checkpoints in", CKPT_DIR)
    if not CKPT_DIR.exists():
        print("  (none)")
        return
    for p in sorted(CKPT_DIR.iterdir()):
        print(f"  {p.name:40s} {p.stat().st_size/1024:.1f} KB")


def cmd_all(cfg: Config) -> None:
    prepare(cfg)
    cmd_pretrain(cfg)
    cmd_joint(cfg)
    print("\n== Sample generations ==")
    cmd_generate(cfg, "def quicksort", expert="python", max_tokens=120)
    cmd_generate(cfg, "The user asked me to", expert="english", max_tokens=120)


# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Cooperating Experts test framework")
    parser.add_argument("command", choices=[
        "extract", "prepare", "pretrain", "joint", "generate", "all", "status",
        "download", "gen-data", "prepare-hybrid", "all-hybrid",
    ])
    parser.add_argument("prompt", nargs="?", default=None)
    parser.add_argument("--expert", default="english", choices=["python", "english"])
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--n", type=int, default=None,
                        help="number of sessions (for gen-data / all-hybrid)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name for gen-data (overrides LLM_MODEL in .env)")
    parser.add_argument("--clean", action="store_true",
                        help="delete existing llm-* sessions before gen-data "
                             "(default: append to existing sessions)")
    parser.add_argument(
        "--socks5",
        default=None,
        metavar="HOST:PORT",
        help="Route all HuggingFace/HTTP traffic through a SOCKS5 proxy "
             "(e.g. 127.0.0.1:1080). Uses socks5h:// so DNS is resolved at "
             "the proxy, bypassing a local DNS hijack.",
    )
    args = parser.parse_args()

    # Configure proxy BEFORE any network access (extract/prepare/pretrain).
    if args.socks5:
        configure_socks_proxy(args.socks5)

    cfg = Config.default()
    torch.manual_seed(cfg.train.seed)

    if args.command == "extract":
        cmd_extract(cfg)
    elif args.command == "prepare":
        cmd_prepare(cfg)
    elif args.command == "pretrain":
        cmd_pretrain(cfg)
    elif args.command == "joint":
        cmd_joint(cfg)
    elif args.command == "generate":
        if args.prompt is None:
            print("Error: generate requires a prompt.")
            sys.exit(1)
        cmd_generate(cfg, args.prompt, args.expert, args.max_tokens)
    elif args.command == "status":
        cmd_status(cfg)
    elif args.command == "all":
        cmd_all(cfg)
    elif args.command == "download":
        cmd_download(cfg)
    elif args.command == "gen-data":
        n = args.n if args.n is not None else cfg.train.n_sessions
        cmd_gen_data(cfg, n, model=args.model, clean=args.clean)
    elif args.command == "prepare-hybrid":
        cmd_prepare_hybrid(cfg)
    elif args.command == "all-hybrid":
        # Full hybrid pipeline: data -> tokenizers -> pretrain -> joint -> mixed -> sample
        cmd_prepare_hybrid(cfg)
        cmd_pretrain(cfg)
        cmd_joint(cfg)


if __name__ == "__main__":
    main()
