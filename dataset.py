"""Dataset loading for the two experts.

Both experts are trained on a clean, locally-generated synthetic corpus
(`synthetic_data.py`) that simulates user<->agent coding sessions. We:
  1. segment the synthetic sessions into code (python) and text (english)
     pieces,
  2. write them to raw text files (one piece per line) for tokenizer training,
  3. build in-memory tokenized datasets of fixed-length windows / hand-off
     pairs / interleaved sessions for training.

All sizes and caps are defined in `config.py` (TrainConfig) so there are no
magic numbers in this module.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from config import DATA_DIR, Config, ExpertConfig
from tokenizer import ExpertTokenizer

# Synthetic sessions (see synthetic_data.py) simulate a user<->agent coding
# dialogue. Each session is a JSONL file of events; a message event has:
#   type="message", message.role in {"user","assistant"},
#   message.content = [ {type, text, ...}, ... ]
# Item types: "text" (prose / fenced code). The agent's code is wrapped in a
# fenced ```python block so the segmenter can split code from prose cleanly.
# When a SOCKS5 proxy is configured (via configure_socks_proxy), DNS is
# resolved *at* the proxy (socks5h://), so the local DNS hijack no longer
# matters. We track this so _dns_looks_valid can skip the loopback check.
_SOCKS_PROXY_ACTIVE = False


def configure_socks_proxy(proxy: str) -> None:
    """Route all HTTP(S) traffic through a SOCKS5 proxy.

    `proxy` is a URL like "socks5h://127.0.0.1:1080" or "127.0.0.1:1080".
    The "socks5h" scheme (note the trailing 'h') makes the proxy resolve DNS,
    which is what we need to bypass a local DNS hijack. If the scheme is
    omitted we default to socks5h://.

    This sets the standard HTTP(S)/ALL_PROXY environment variables that
    `requests`, `huggingface_hub`, and `fsspec` all honor. (The current
    dataset is generated locally and needs no network, but the switch is kept
    for any future remote data source.)
    """
    global _SOCKS_PROXY_ACTIVE
    p = str(proxy).strip()
    if not p:
        return
    if "://" not in p:
        p = "socks5h://" + p
    elif p.startswith("socks5://"):
        # Upgrade to socks5h so DNS is resolved at the proxy.
        p = "socks5h://" + p[len("socks5://"):]
    os.environ["HTTP_PROXY"] = p
    os.environ["HTTPS_PROXY"] = p
    os.environ["ALL_PROXY"] = p
    os.environ["http_proxy"] = p
    os.environ["https_proxy"] = p
    os.environ["all_proxy"] = p
    _SOCKS_PROXY_ACTIVE = True
    print(f"== SOCKS5 proxy enabled: {p} (DNS resolved at proxy) ==")


# ---------------------------------------------------------------------- #
# ---------------------------------------------------------------------- #
# Synthetic session extraction (code vs. text segmentation)
# ---------------------------------------------------------------------- #
def _iter_synthetic_events(max_files: int = None) -> Iterator[dict]:
    """Stream raw session events from the local synthetic dataset.

    The synthetic data (see `synthetic_data.py`) is a clean, fully-controlled
    simulation of user<->agent coding sessions written to
    `data/synthetic_raw/*.jsonl`. Each line is one event dict in the same
    schema the segmenter expects. This replaces the previous noisy web-scraped
    download, which produced duplicated text and tool-call artifacts.

    If the synthetic directory is missing we generate it on the fly.
    """
    from synthetic_data import SYNTHETIC_DIR, write_synthetic

    local = SYNTHETIC_DIR
    if not local.exists():
        write_synthetic(local)

    import glob

    files = sorted(glob.glob(str(local / "*.jsonl")))[:max_files]
    if not files:
        raise RuntimeError(f"No synthetic session files found in {local}")
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# A fenced code block: ```lang\n ... \n```  (lang optional).
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
# Heuristic: a line looks like code if it has code-ish punctuation density.
_CODE_LINE_RE = re.compile(r"[{}();=]|^\s{0,3}(def|class|import|from|function|const|let|var|if|for|while|return|public|private|#include|SELECT|</?)\b")


def _split_text_into_pieces(text: str) -> List[Tuple[str, str]]:
    """Split a text blob into (kind, piece) where kind in {'code','text'}.

    Code pieces are fenced ``` blocks (preferring ```python / ```js / etc.).
    Everything else is a text piece. This is the core code/text separator.
    """
    pieces: List[Tuple[str, str]] = []
    pos = 0
    for m in _FENCE_RE.finditer(text):
        # Text before the fence.
        before = text[pos:m.start()].strip()
        if before:
            pieces.append(("text", before))
        code = m.group(2).strip()
        if code:
            pieces.append(("code", code))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        pieces.append(("text", tail))
    return pieces


def _event_to_pieces(ev: dict) -> List[Tuple[str, str]]:
    """Turn one message event into a list of (kind, text) pieces.

    - assistant/user `text` items -> split by fences into code/text.
    - `thinking` items -> text (reasoning prose).
    - `toolCall` items -> serialize name + arguments as a CODE piece (it is
      executable tool input, e.g. a Bash command or a file path).
    - tool results (user messages with no role-specific prose but command
      output) -> code if they look like code/logs, else text.
    """
    if ev.get("type") != "message":
        return []
    msg = ev.get("message") or {}
    role = msg.get("role")
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    out: List[Tuple[str, str]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "text":
            text = item.get("text") or ""
            out.extend(_split_text_into_pieces(text))
        elif itype == "thinking":
            text = item.get("thinking") or item.get("text") or ""
            if text.strip():
                out.append(("text", text.strip()))
        elif itype == "toolCall":
            name = item.get("name") or "tool"
            args = item.get("arguments")
            if isinstance(args, dict):
                arg_s = json.dumps(args, ensure_ascii=False)
            else:
                arg_s = str(args or "")
            call = f"{name}({arg_s})"
            out.append(("code", call))
    return out


def extract_synthetic_corpora(
    out_dir: Path = DATA_DIR,
    max_sessions: int = None,
    shard_size: int = None,
    cfg: Config = None,
) -> Tuple[List[Path], List[Path]]:
    """Segment synthetic sessions into separate code and text corpora.

    Walks events grouped by session (a session = one JSONL file; we detect
    session boundaries by the `session_id` field or by a `type` reset). For
    each session we collect (kind, piece) pieces in order, then write code
    pieces to `python_corpus_*.txt` and text pieces to `english_corpus_*.txt`
    (one piece per line). Returns (code_shards, text_shards).
    """
    if cfg is None:
        cfg = Config.default()
    if max_sessions is None:
        max_sessions = cfg.train.extract_max_sessions
    if shard_size is None:
        shard_size = cfg.train.extract_shard_size

    out_dir.mkdir(parents=True, exist_ok=True)
    code_shards: List[Path] = []
    text_shards: List[Path] = []
    cur_code: List[str] = []
    cur_text: List[str] = []
    n_sessions = 0
    last_session: Optional[str] = None

    def _flush():
        nonlocal cur_code, cur_text, code_shards, text_shards
        if cur_code:
            p = out_dir / f"python_corpus_{len(code_shards)}.txt"
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("\n".join(cur_code) + "\n")
            code_shards.append(p)
            cur_code = []
        if cur_text:
            p = out_dir / f"english_corpus_{len(text_shards)}.txt"
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("\n".join(cur_text) + "\n")
            text_shards.append(p)
            cur_text = []

    for ev in _iter_synthetic_events(max_files=cfg.train.extract_max_files):
        sid = ev.get("session_id")
        # New session -> flush previous and count it.
        if sid is not None and last_session is not None and sid != last_session:
            _flush()
            n_sessions += 1
            if n_sessions >= max_sessions:
                break
        if sid is not None and last_session is None:
            # First session begins.
            n_sessions += 1
            if n_sessions > max_sessions:
                break
        last_session = sid
        for kind, piece in _event_to_pieces(ev):
            piece = " ".join(piece.split())
            if not piece:
                continue
            if kind == "code":
                cur_code.append(piece)
            else:
                cur_text.append(piece)
        # Rotate shards to bound memory.
        if len(cur_code) >= shard_size:
            _flush()
        if len(cur_text) >= shard_size:
            _flush()
    _flush()
    print(f"  [synthetic] wrote {len(code_shards)} code shard(s), "
          f"{len(text_shards)} text shard(s) ({n_sessions} sessions)")
    return code_shards, text_shards


# ---------------------------------------------------------------------- #
# Tokenized window dataset
# ---------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Fixed-length sliding-window token dataset for one expert."""

    def __init__(
        self,
        raw_texts: List[str],
        tokenizer: ExpertTokenizer,
        max_seq_len: int,
        max_windows: int = None,
    ):
        if max_windows is None:
            max_windows = Config.default().train.window_max_windows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.windows: List[List[int]] = []

        # Concatenate all texts with <eos> separators, then slice into windows.
        flat: List[int] = []
        eos = tokenizer.eos_id
        for text in raw_texts:
            ids = tokenizer.tokenizer.encode(text).ids
            flat.extend(ids)
            flat.append(eos)
            if len(self.windows) >= max_windows:
                break
            # Slice windows as we go to bound memory.
            while len(flat) >= max_seq_len:
                self.windows.append(flat[:max_seq_len])
                flat = flat[max_seq_len:]
                if len(self.windows) >= max_windows:
                    break
        # Trailing partial window (left-padded).
        if flat and len(self.windows) < max_windows:
            pad = [tokenizer.pad_id] * (max_seq_len - len(flat))
            self.windows.append(pad + flat)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.windows[idx], dtype=torch.long)


def load_raw_texts(
    expert_name: str,
    max_examples: int = None,
    cfg: Config = None,
) -> List[str]:
    """Load raw text examples for an expert from the synthetic sessions.

    We reconstruct the per-expert pieces directly from `synthetic_raw` using
    the same segmentation as the corpus extractor, so this never depends on
    the (now-removed) HuggingFace fallback or on the intermediate corpus
    shards (which `prepare` deletes after tokenizer training).
    """
    if cfg is None:
        cfg = Config.default()
    if max_examples is None:
        max_examples = cfg.train.max_examples
    kind = "code" if expert_name == "python" else "text"
    texts: List[str] = []
    for ev in _iter_synthetic_events(max_files=cfg.train.extract_max_files):
        for k, piece in _event_to_pieces(ev):
            if k == kind and piece.strip():
                texts.append(piece)
                if len(texts) >= max_examples:
                    return texts
    return texts


# ---------------------------------------------------------------------- #
# Hand-off pairs for joint training
# ---------------------------------------------------------------------- #
class HandoffDataset(Dataset):
    """Pairs of (prefix_A, continuation_B) for joint projection training.

    We build pairs by taking a Python snippet as the prefix and an English
    sentence as the continuation (and vice-versa), simulating a context where
    one expert hands off to another. The projections must carry enough signal
    across the boundary for the second expert to continue coherently.
    """

    def __init__(
        self,
        texts_a: List[str],
        tok_a: ExpertTokenizer,
        texts_b: List[str],
        tok_b: ExpertTokenizer,
        seq_len: int,
        max_pairs: int = None,
    ):
        if max_pairs is None:
            max_pairs = Config.default().train.joint_max_pairs
        self.seq_len = seq_len
        self.pairs: List[Tuple[List[int], List[int]]] = []

        n = min(len(texts_a), len(texts_b), max_pairs)
        for i in range(n):
            ids_a = tok_a.tokenizer.encode(texts_a[i]).ids[:seq_len]
            ids_b = tok_b.tokenizer.encode(texts_b[i]).ids[:seq_len]
            if len(ids_a) >= 8 and len(ids_b) >= 8:
                # Pad to seq_len.
                ids_a = ids_a + [tok_a.pad_id] * (seq_len - len(ids_a))
                ids_b = ids_b + [tok_b.pad_id] * (seq_len - len(ids_b))
                self.pairs.append((ids_a, ids_b))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.pairs[idx]
        return torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)


# ---------------------------------------------------------------------- #
# Session-aware hand-off pairs (real code<->text boundaries)
# ---------------------------------------------------------------------- #
def _reconstruct_sessions(
    cfg: Config, max_sessions: int
) -> List[List[Tuple[str, str]]]:
    """Rebuild the ordered (kind, piece) stream for each synthetic session."""
    sessions: Dict[str, List[Tuple[str, str]]] = {}
    order: List[str] = []
    last_session: Optional[str] = None
    for ev in _iter_synthetic_events(max_files=cfg.train.extract_max_files):
        sid = ev.get("session_id") or f"__anon_{last_session}"
        if sid not in sessions:
            sessions[sid] = []
            order.append(sid)
        last_session = sid
        for kind, piece in _event_to_pieces(ev):
            piece = " ".join(piece.split())
            if piece:
                sessions[sid].append((kind, piece))
        if len(order) >= max_sessions:
            break
    return [sessions[s] for s in order if sessions[s]]


def build_boundary_handoff_pairs(
    tokenizers: Dict[str, ExpertTokenizer],
    seq_len: int,
    max_pairs: int = None,
    cfg: Config = None,
) -> Tuple[List[Tuple[List[int], List[int]]], List[Tuple[List[int], List[int]]]]:
    """Build hand-off pairs from REAL code<->text boundaries within sessions.

    Unlike the (legacy) `HandoffDataset`, which pairs a random code snippet
    with an unrelated prose snippet, this walks each session, merges
    consecutive same-kind pieces into segments, and for every boundary between
    a code segment and a text segment emits a (prefix, continuation) pair:

      - prefix      = the TAIL of the segment before the boundary
                      (its last `seq_len//2` tokens -- the hand-off happens at
                      its end, and joint_loss carries the last hidden states),
      - continuation = the HEAD of the segment after the boundary.

    Both are tokenized in their own expert's vocabulary, so the projection has
    a *semantically related* continuation to learn from.

    Returns (ab_pairs, ba_pairs):
      ab_pairs: python -> english  (prefix in python vocab, cont in english vocab)
      ba_pairs: english -> python
    """
    if cfg is None:
        cfg = Config.default()
    if max_pairs is None:
        max_pairs = cfg.train.joint_max_pairs
    tok_py = tokenizers["python"]
    tok_en = tokenizers["english"]
    half = max(8, seq_len // 2)
    ab: List[Tuple[List[int], List[int]]] = []
    ba: List[Tuple[List[int], List[int]]] = []

    for pieces in _reconstruct_sessions(cfg, cfg.train.extract_max_sessions):
        # Merge consecutive same-kind pieces into segments.
        merged: List[Tuple[str, str]] = []
        for kind, piece in pieces:
            if merged and merged[-1][0] == kind:
                merged[-1] = (kind, merged[-1][1] + " " + piece)
            else:
                merged.append((kind, piece))
        for i in range(len(merged) - 1):
            k0, p0 = merged[i]
            k1, p1 = merged[i + 1]
            if k0 == k1:
                continue
            tok0 = tok_py if k0 == "code" else tok_en
            tok1 = tok_py if k1 == "code" else tok_en
            ids0 = tok0.tokenizer.encode(p0).ids
            ids1 = tok1.tokenizer.encode(p1).ids
            if len(ids0) < 8 or len(ids1) < 8:
                continue
            prefix = ids0[-half:]
            cont = ids1[:half]
            if k0 == "code":       # python -> english
                ab.append((prefix, cont))
            else:                  # english -> python
                ba.append((prefix, cont))
        if len(ab) >= max_pairs and len(ba) >= max_pairs:
            break
    return ab[:max_pairs], ba[:max_pairs]


class BoundaryHandoffDataset(Dataset):
    """Padded (prefix, continuation) pairs produced by build_boundary_handoff_pairs.

    `pad_prefix` / `pad_cont` are the pad ids of the prefix and continuation
    experts respectively (they can differ, since each expert has its own vocab).
    """

    def __init__(
        self,
        pairs: List[Tuple[List[int], List[int]]],
        pad_prefix: int,
        pad_cont: int,
        seq_len: int,
    ):
        self.data: List[Tuple[List[int], List[int]]] = []
        for a, b in pairs:
            a = a[:seq_len]
            b = b[:seq_len]
            a = a + [pad_prefix] * (seq_len - len(a))
            b = b + [pad_cont] * (seq_len - len(b))
            self.data.append((a, b))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b = self.data[idx]
        return torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)


# ---------------------------------------------------------------------- #
# Interleaved (mixed) dataset for joint training on the WHOLE data
# ---------------------------------------------------------------------- #
class MixedDataset(Dataset):
    """Sequences of (expert_name, ids) segments with switch tokens inserted.

    Each example is one synthetic session, replayed in order. We walk the
    session's pieces; whenever the kind flips between 'code' and 'text' we
    append the matching switch token to the *end* of the preceding segment's
    ids (so it is a real LM target that conditions the next expert). The
    result is a list of segments:

        [("python", [ids... <switch:english>]),
         ("english", [ids... <switch:python>]),
         ("python", [ids... <eos>])]

    The model is trained end-to-end to predict every token, including the
    switch tokens, so it learns *when* to hand off between experts.
    """

    def __init__(
        self,
        tokenizers: Dict[str, ExpertTokenizer],
        max_seq_len: int = 512,
        max_sessions: int = None,
        cfg: Config = None,
    ):
        if cfg is None:
            cfg = Config.default()
        if max_sessions is None:
            max_sessions = cfg.train.mixed_max_sessions
        self.tokenizers = tokenizers
        self.max_seq_len = max_seq_len
        self.examples: List[List[Tuple[str, List[int]]]] = []

        # Reconstruct the per-session ordered piece stream from the raw events.
        sessions: Dict[str, List[Tuple[str, str]]] = {}
        order: List[str] = []
        last_session: Optional[str] = None
        for ev in _iter_synthetic_events(max_files=cfg.train.extract_max_files):
            sid = ev.get("session_id") or f"__anon_{last_session}"
            if sid not in sessions:
                sessions[sid] = []
                order.append(sid)
            last_session = sid
            for kind, piece in _event_to_pieces(ev):
                piece = " ".join(piece.split())
                if piece:
                    sessions[sid].append((kind, piece))
            if len(order) >= max_sessions:
                break

        for sid in order:
            pieces = sessions[sid]
            if not pieces:
                continue
            segs = self._pieces_to_segments(pieces)
            if segs:
                self.examples.append(segs)

    def _pieces_to_segments(
        self, pieces: List[Tuple[str, str]]
    ) -> List[Tuple[str, List[int]]]:
        """Group ordered (kind, piece) into expert segments with switch tokens.

        `kind` is 'code' or 'text'; it maps to expert name 'python' or
        'english' respectively. Switch tokens are appended to the end of a
        segment when the kind flips, so the model learns when to hand off.
        """
        tok_py = self.tokenizers["python"]
        tok_en = self.tokenizers["english"]
        sw_en = tok_py.switch_id("english")  # switch python -> english
        sw_py = tok_en.switch_id("python")  # switch english -> python
        eos_py = tok_py.eos_id
        eos_en = tok_en.eos_id

        def _expert_for(kind: str) -> str:
            return "python" if kind == "code" else "english"

        segs: List[Tuple[str, List[int]]] = []
        cur_kind: Optional[str] = None
        cur_ids: List[int] = []
        cur_tok = tok_py

        def _flush(kind: str, ids: List[int], switch_id: Optional[int],
                   eos_id: int):
            if not ids:
                return
            if switch_id is not None:
                ids = ids + [switch_id]
            else:
                ids = ids + [eos_id]
            # Cap to max_seq_len AFTER appending the switch/eos token. Without
            # this, a piece that is exactly max_seq_len long becomes
            # max_seq_len + 1 after the append, which then mismatches the
            # logits/target shapes in mixed_loss (off-by-one crash).
            if len(ids) > self.max_seq_len:
                ids = ids[: self.max_seq_len]
            segs.append((_expert_for(kind), ids))

        for kind, piece in pieces:
            tok = tok_py if kind == "code" else tok_en
            ids = tok.tokenizer.encode(piece).ids[: self.max_seq_len]
            if not ids:
                continue
            if cur_kind is None:
                cur_kind = kind
                cur_tok = tok
                cur_ids = ids
            elif kind == cur_kind:
                # Same expert: append (cap to max_seq_len by starting new seg
                # only if overflow; here we just concatenate and rely on the
                # trainer to window if needed — but keep simple: extend).
                if len(cur_ids) + len(ids) <= self.max_seq_len:
                    cur_ids = cur_ids + ids
                else:
                    # Flush current, start fresh segment of same kind.
                    _flush(cur_kind, cur_ids, None, cur_tok.eos_id)
                    cur_ids = ids
            else:
                # Kind flipped -> emit switch token for the *previous* kind.
                sw = sw_en if cur_kind == "code" else sw_py
                _flush(cur_kind, cur_ids, sw, cur_tok.eos_id)
                cur_kind = kind
                cur_tok = tok
                cur_ids = ids
        # Final segment: terminate with its own <eos>.
        if cur_kind is not None:
            _flush(cur_kind, cur_ids, None,
                   eos_py if cur_kind == "code" else eos_en)
        return segs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> List[Tuple[str, torch.Tensor]]:
        return [(name, torch.tensor(ids, dtype=torch.long))
                for name, ids in self.examples[idx]]
