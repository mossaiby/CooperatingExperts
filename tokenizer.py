"""Per-expert BPE tokenizers built with the HuggingFace `tokenizers` library.

Each expert trains its own byte-level BPE tokenizer on its own corpus, so the
vocabularies are genuinely different (code tokens vs. prose tokens). We then
append a *fixed* set of special tokens to every expert so that the switching
mechanism is uniform:

    <pad>  <eos>  <switch:python>  <switch:english>

The wrapper rewrites the switch-token ids to absolute positions at runtime, so
the per-expert tokenizer only needs to expose:
    - encode / decode
    - vocab_size (including special tokens)
    - id_of(name)  -> int   (for special tokens)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelPostProcessor

from config import ExpertConfig, CKPT_DIR


def _special_tokens_for(expert_name: str, all_names: List[str]) -> List[str]:
    """Special tokens appended to *this* expert's vocab.

    Order is fixed: pad, eos, then one switch token per expert (self included).
    """
    tokens = ["<unk>", "<pad>", "<eos>"]
    tokens += [f"<switch:{n}>" for n in all_names]
    return tokens


class ExpertTokenizer:
    """Wraps a HuggingFace `Tokenizer` with expert-specific helpers."""

    def __init__(self, tokenizer: Tokenizer, special_tokens: List[str]):
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        # Build a quick lookup for special-token strings -> id.
        self._special_ids: Dict[str, int] = {}
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        for tok in special_tokens:
            if tok in vocab:
                self._special_ids[tok] = vocab[tok]
            else:
                # Fallback: encode the single token string.
                enc = tokenizer.encode(tok, add_special_tokens=True)
                self._special_ids[tok] = enc.ids[0]

    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def id_of(self, special_token_name: str) -> int:
        return self._special_ids[special_token_name]

    @property
    def pad_id(self) -> int:
        return self.id_of("<pad>")

    @property
    def eos_id(self) -> int:
        return self.id_of("<eos>")

    def switch_id(self, target_expert: str) -> int:
        return self.id_of(f"<switch:{target_expert}>")

    # ------------------------------------------------------------------ #
    def encode_batch(self, texts: List[str], max_len: int) -> "torch.Tensor":  # noqa: F821
        """Encode a list of raw strings to a padded int tensor [B, T]."""
        import torch

        enc = self.tokenizer.encode_batch(texts)
        ids = [e.ids for e in enc]
        return _pad(ids, max_len, self.pad_id, torch)

    def decode(self, ids: List[int]) -> str:
        # Strip pad / eos / switch tokens for readable output.
        clean = [i for i in ids if i not in self._special_ids.values()]
        return self.tokenizer.decode(clean, skip_special_tokens=True)

    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "ExpertTokenizer":
        tok = Tokenizer.from_file(str(path))
        # Re-derive special tokens from the tokenizer's added-tokens list.
        added = tok.get_vocab(with_added_tokens=True)
        special = sorted(
            [t for t in added if t.startswith("<") and t.endswith(">")],
            key=lambda t: added[t],
        )
        return cls(tok, special)


def _pad(seqs: List[List[int]], max_len: int, pad_id: int, torch) -> "torch.Tensor":  # noqa: F821
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = min(len(s), max_len)
        out[i, :n] = torch.tensor(s[:n], dtype=torch.long)
    return out


# ---------------------------------------------------------------------- #
def build_tokenizer(
    expert_cfg: ExpertConfig,
    corpus_files: List[Path],
    all_expert_names: List[str],
) -> ExpertTokenizer:
    """Train a byte-level BPE tokenizer on the given corpus files."""
    special = _special_tokens_for(expert_cfg.name, all_expert_names)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.post_processor = ByteLevelPostProcessor(trim_offsets=True)

    # vocab_size is the *BPE* size; special tokens are added on top.
    trainer = BpeTrainer(
        vocab_size=expert_cfg.vocab_size,
        special_tokens=special,
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        _iter_files(corpus_files),
        trainer=trainer,
    )
    return ExpertTokenizer(tokenizer, special)


def _iter_files(files: List[Path]):
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            # Yield line by line to keep memory low.
            for line in fh:
                if line.strip():
                    yield line
