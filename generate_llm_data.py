"""Generate training sessions by wrapping REAL Python code in LLM-generated prose.

This is the "hybrid" data strategy: the python expert pre-trains on real
Python functions (from download_code_corpus.py), and we use an LLM only to
generate the *surrounding prose* — a problem description, an explanation of
the approach, and a follow-up — so the english expert learns real
code-adjacent prose and the mixed phase learns realistic switch boundaries.

Each generated session has the shape:

    [user]    problem description (English prose)        <- LLM generated
    [agent]   explanation + ```python <real code> ``` + explanation  <- LLM prose + real code
    [user]    follow-up question (English prose)         <- LLM generated
    [agent]   short answer + ```python <real code> ``` (optional)    <- LLM prose + real code
    [user]    closing remark (English prose)             <- LLM generated

The code is REAL (from the corpus); only the prose is LLM-generated. This
gives the python expert the real Python distribution while teaching the
cooperation mechanism with natural switch boundaries.

Output is written to data/synthetic_raw/*.jsonl in the same event schema
that dataset.py already consumes (so no downstream changes are needed), and
data/synthetic_combined/*.txt for the human-readable combined view.

Usage:
    python generate_llm_data.py                    # uses .env for API config
    python generate_llm_data.py --n 10000          # generate 10000 sessions
    python generate_llm_data.py --max-tokens 4096  # longer LLM responses
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DATA_DIR

# Load environment variables from .env (if present).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed; rely on real environment variables.
    pass

from openai import OpenAI

SYNTHETIC_DIR = DATA_DIR / "synthetic_raw"
COMBINED_DIR = DATA_DIR / "synthetic_combined"
FUNCTIONS_PATH = DATA_DIR / "code_corpus" / "functions.jsonl"

# Fenced code block regex (same as dataset.py / synthetic_data.py).
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------- #
# API client
# ---------------------------------------------------------------------- #
def _make_client(model_override: str = None) -> Tuple[OpenAI, str, int]:
    """Build the OpenAI-compatible client from environment variables.

    If `model_override` is given (e.g. from a CLI --model flag), it takes
    precedence over the LLM_MODEL env var.
    """
    base = os.environ.get("LLM_API_BASE")
    key = os.environ.get("LLM_API_KEY")
    model = model_override or os.environ.get("LLM_MODEL")
    timeout = int(os.environ.get("LLM_TIMEOUT", "120"))
    if not base or not key or not model:
        print("ERROR: LLM_API_BASE, LLM_API_KEY, and LLM_MODEL must be set.")
        print("Copy .env-example to .env and fill in your values.")
        sys.exit(1)
    client = OpenAI(base_url=base, api_key=key, timeout=timeout)
    return client, model, timeout


# ---------------------------------------------------------------------- #
# Prompt design
# ---------------------------------------------------------------------- #
_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. You explain Python code clearly and "
    "concisely. You always wrap code in ```python fenced blocks. Your "
    "explanations are short, natural, and reference the code directly."
)

_USER_PROMPT_TEMPLATE = """Here is a Python function:

```python
{code}
```

Write a natural coding-assistant dialogue about this function. Respond with ONLY a JSON object (no markdown, no explanation) with this exact shape:

{{
  "problem": "A one or two sentence description of a task this function could solve, written as a user request.",
  "explanation_before": "1-3 sentences the assistant says before showing the code, explaining the approach.",
  "explanation_after": "1-2 sentences the assistant says after the code, summarizing or noting a caveat.",
  "followup": "A short follow-up question from the user about the function or a related edge case.",
  "answer": "1-2 sentences answering the follow-up. If the answer needs code, include a ```python block.",
  "closing": "A one-sentence closing remark from the user."
}}

Keep every field short and natural. Do NOT repeat the function code in explanation_before or explanation_after — the code will be inserted separately."""


def _build_user_prompt(code: str) -> str:
    return _USER_PROMPT_TEMPLATE.format(code=code)


# ---------------------------------------------------------------------- #
# LLM call with retry
# ---------------------------------------------------------------------- #
def _call_llm(
    client: OpenAI, model: str, prompt: str, max_tokens: int,
    temperature: float, retries: int = 3,
) -> Optional[str]:
    """Call the LLM and return the text response, or None on failure."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [llm] API error (attempt {attempt+1}/{retries}): {e}; retrying in {wait}s")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------- #
# Parse the LLM JSON response
# ---------------------------------------------------------------------- #
def _parse_llm_response(text: str) -> Optional[dict]:
    """Extract the JSON object from the LLM response (tolerates markdown fences)."""
    # Strip markdown code fences if the model wrapped the JSON.
    text = text.strip()
    if text.startswith("```"):
        # Remove the opening fence (```json or ```) and the closing fence.
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block as a fallback.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    # Validate required fields.
    required = ["problem", "explanation_before", "explanation_after",
                "followup", "answer", "closing"]
    for k in required:
        if k not in obj or not isinstance(obj[k], str) or not obj[k].strip():
            return None
    return obj


# ---------------------------------------------------------------------- #
# Session assembly (event schema expected by dataset.py)
# ---------------------------------------------------------------------- #
def _build_session_events(
    sid: str, func: dict, dialogue: dict,
) -> List[dict]:
    """Assemble one session in the event schema dataset.py consumes.

    The agent's first reply interleaves explanation_before + the real code
    (in a ```python fence) + explanation_after, so the segmenter splits it
    into prose/code/prose pieces with natural switch boundaries.
    """
    code = func["code"].strip()
    # The agent's first reply: prose, then the fenced code, then prose.
    agent_reply_1 = (
        f"{dialogue['explanation_before']}\n\n"
        f"```python\n{code}\n```\n\n"
        f"{dialogue['explanation_after']}"
    )
    # The agent's second reply: the follow-up answer (may contain code).
    agent_reply_2 = dialogue["answer"]

    events = [
        {"type": "message", "session_id": sid,
         "message": {"role": "user",
                     "content": [{"type": "text", "text": dialogue["problem"]}]}},
        {"type": "message", "session_id": sid,
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": agent_reply_1}]}},
        {"type": "message", "session_id": sid,
         "message": {"role": "user",
                     "content": [{"type": "text", "text": dialogue["followup"]}]}},
        {"type": "message", "session_id": sid,
         "message": {"role": "assistant",
                     "content": [{"type": "text", "text": agent_reply_2}]}},
        {"type": "message", "session_id": sid,
         "message": {"role": "user",
                     "content": [{"type": "text", "text": dialogue["closing"]}]}},
    ]
    return events


# ---------------------------------------------------------------------- #
# Combined-view writer (mirrors synthetic_data.write_combined)
# ---------------------------------------------------------------------- #
def _split_text_into_pieces(text: str) -> List[Tuple[str, str]]:
    """Split a text blob into (kind, piece) where kind in {'code','text'}."""
    pieces: List[Tuple[str, str]] = []
    pos = 0
    for m in _FENCE_RE.finditer(text):
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


def _write_combined_file(sid: str, events: List[dict], out_dir: Path) -> None:
    """Write the human-readable combined view with <switch:NAME> markers."""
    pieces: List[Tuple[str, str]] = []
    for ev in events:
        msg = ev.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text") or ""
                pieces.extend(_split_text_into_pieces(text))
    lines: List[str] = []
    prev_kind: Optional[str] = None
    for kind, piece in pieces:
        expert = "python" if kind == "code" else "english"
        if prev_kind is not None and kind != prev_kind:
            lines.append(f"<switch:{expert}>")
        lines.append(f"[{expert}] {piece}")
        prev_kind = kind
    with open(out_dir / f"{sid}.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------- #
# Main generation loop
# ---------------------------------------------------------------------- #
def generate_llm_sessions(
    n: int, max_tokens: int, temperature: float, seed: int,
    raw_dir: Path, combined_dir: Path, model_override: str = None,
    clean: bool = False,
) -> int:
    """Generate `n` sessions by wrapping real functions in LLM prose.

    By default this APPENDS to the output directories: it scans for existing
    `llm-*.jsonl` files and starts numbering new sessions after the highest
    existing id, so re-running `--n 100` then `--n 1000` produces 1100 total
    sessions with no overwrites. Pass `clean=True` (or --clean on the CLI) to
    delete all existing `llm-*` files in both output dirs before generating.

    Returns the number of sessions successfully generated (this run only).
    """
    if not FUNCTIONS_PATH.exists():
        print(f"ERROR: {FUNCTIONS_PATH} not found.")
        print("Run `python download_code_corpus.py` first to fetch real Python code.")
        sys.exit(1)

    # Load the real functions.
    functions: List[dict] = []
    with open(FUNCTIONS_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                functions.append(json.loads(line))
    print(f"  [llm_data] loaded {len(functions)} real functions from {FUNCTIONS_PATH.name}")
    if not functions:
        print("ERROR: no functions loaded. Run download_code_corpus.py first.")
        sys.exit(1)

    raw_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Determine the starting offset for session ids. By default we append:
    # scan for the highest existing llm-NNNNN index and continue after it.
    # With --clean we wipe all llm-* files first and start from 0.
    import glob
    import re as _re
    if clean:
        for d in (raw_dir, combined_dir):
            for f in glob.glob(str(d / "llm-*")):
                try:
                    Path(f).unlink()
                except OSError:
                    pass
        start_offset = 0
        print(f"  [llm_data] --clean: removed existing llm-* files")
    else:
        existing = glob.glob(str(raw_dir / "llm-*.jsonl"))
        start_offset = 0
        for f in existing:
            m = _re.search(r"llm-(\d+)\.jsonl$", f)
            if m:
                start_offset = max(start_offset, int(m.group(1)) + 1)
        if start_offset > 0:
            print(f"  [llm_data] appending: {start_offset} existing sessions, "
                  f"new sessions start at llm-{start_offset:05d}")

    client, model, _ = _make_client(model_override)
    print(f"  [llm_data] using model: {model}")
    rng = random.Random(seed)

    generated = 0
    skipped = 0
    # Cycle through functions (with shuffling) so we can generate more
    # sessions than functions if desired (each gets different prose).
    indices = list(range(len(functions)))
    rng.shuffle(indices)
    pos = 0
    t0 = time.time()

    while generated < n:
        if pos >= len(indices):
            # Reshuffle and go again (so n > len(functions) is supported).
            rng.shuffle(indices)
            pos = 0
        idx = indices[pos]
        pos += 1
        func = functions[idx]
        code = func["code"]

        # Quick sanity: the code must still parse.
        try:
            ast.parse(code)
        except SyntaxError:
            skipped += 1
            continue

        prompt = _build_user_prompt(code)
        resp_text = _call_llm(client, model, prompt, max_tokens, temperature)
        if resp_text is None:
            skipped += 1
            continue
        dialogue = _parse_llm_response(resp_text)
        if dialogue is None:
            skipped += 1
            continue

        sid = f"llm-{start_offset + generated:05d}"
        events = _build_session_events(sid, func, dialogue)

        # Write the raw JSONL session.
        with open(raw_dir / f"{sid}.jsonl", "w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        # Write the combined view.
        _write_combined_file(sid, events, combined_dir)

        generated += 1
        if generated % 100 == 0:
            elapsed = time.time() - t0
            rate = generated / max(elapsed, 1)
            print(f"  [llm_data] {generated}/{n} sessions "
                  f"({rate:.1f}/s, {skipped} skipped)")

    print(f"  [llm_data] done: {generated} sessions in {time.time()-t0:.0f}s "
          f"({skipped} skipped)")
    return generated


# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training sessions by wrapping real Python code in LLM prose."
    )
    parser.add_argument("--n", type=int, default=10000,
                        help="number of sessions to generate (default 10000)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="max tokens per LLM response (default 1024)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="LLM sampling temperature (default 0.9 for diversity)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name (overrides LLM_MODEL in .env). "
                             "e.g. openai/gpt-oss-20b:free or cohere/north-mini-code:free")
    parser.add_argument("--clean", action="store_true",
                        help="delete existing llm-* sessions before generating "
                             "(default: append to existing sessions)")
    parser.add_argument("--raw-out", type=str, default=str(SYNTHETIC_DIR),
                        help="output dir for raw sessions (default data/synthetic_raw)")
    parser.add_argument("--combined-out", type=str, default=str(COMBINED_DIR),
                        help="output dir for combined sessions")
    args = parser.parse_args()

    print("== Generating LLM-wrapped sessions (hybrid data) ==")
    generate_llm_sessions(
        n=args.n,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        raw_dir=Path(args.raw_out),
        combined_dir=Path(args.combined_out),
        model_override=args.model,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
