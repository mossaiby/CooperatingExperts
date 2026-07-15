"""Download / extract a real Python code corpus for the python expert.

The cooperating-experts framework needs *real* Python code to train the
python expert on the actual distribution of Python (not just a handful of
templated snippets). This script fetches a corpus of real Python functions
and writes them in a format the rest of the pipeline can consume.

Two sources are supported:

1. **CodeSearchNet Python** (default): a curated corpus of ~500k
   human-written Python functions. We download a small, self-contained
   slice and extract clean, parseable functions with their docstrings.
   This is the recommended source — it's free, high-quality, and diverse.

2. **Local directory**: if you already have a folder of `.py` files (e.g.
   cloned GitHub repos), point `--source local --local-dir <path>` and we
   extract functions from them with `ast`.

Output:
  - `data/code_corpus/functions.jsonl`  — one JSON object per function:
        {"code": "def foo(...):\n    ...", "docstring": "...", "name": "foo"}
  - `data/code_corpus/python_corpus.txt` — all function bodies concatenated
    (one per line, whitespace-normalised) for tokenizer training.
  - `data/code_corpus/english_corpus.txt` — all docstrings concatenated,
    for the english expert's pre-training (real code-adjacent prose).

Usage:
    python download_code_corpus.py                    # CodeSearchNet, default size
    python download_code_corpus.py --n 50000          # more functions
    python download_code_corpus.py --source local --local-dir ./my_repos
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from config import DATA_DIR

CORPUS_DIR = DATA_DIR / "code_corpus"

# CodeSearchNet Python split. We download the raw parquet shard directly
# from HuggingFace. Note: the URL must use /resolve/main/ (the raw-file
# endpoint), NOT /blob/main/ (which serves the HTML viewer page and fails
# to download). The dataset has both a `test` and `train` split; `test` is
# smaller and sufficient for our purposes (~40k functions).
CSN_PYTHON_URL = (
    "https://huggingface.co/datasets/code-search-net/code_search_net/"
    "resolve/main/python/test-00000-of-00001.parquet"
)


# ---------------------------------------------------------------------- #
# Source 1: CodeSearchNet Python
# ---------------------------------------------------------------------- #
def _download_csn(n: int, out_dir: Path) -> List[dict]:
    """Download a slice of CodeSearchNet Python and extract functions.

    Returns a list of {"code", "docstring", "name"} dicts.
    Falls back to a local parquet if the download fails.
    """
    print(f"  [code_corpus] downloading CodeSearchNet Python slice...")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  [code_corpus] ERROR: pyarrow is required to read CodeSearchNet.")
        print("    Install it with:  pip install pyarrow")
        sys.exit(1)

    parquet_path = out_dir / "csn_python.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not parquet_path.exists():
        try:
            # HuggingFace rejects requests without a User-Agent header.
            req = urllib.request.Request(
                CSN_PYTHON_URL,
                headers={"User-Agent": "CooperatingExperts/1.0"},
            )
            with urllib.request.urlopen(req) as resp, open(parquet_path, "wb") as fh:
                fh.write(resp.read())
        except Exception as e:
            print(f"  [code_corpus] download failed: {e}")
            print("  [code_corpus] falling back to local .py extraction.")
            print("  [code_corpus] Put .py files in a folder and re-run with:")
            print("    python download_code_corpus.py --source local --local-dir <dir>")
            return []

    print(f"  [code_corpus] reading parquet...")
    table = pq.read_table(parquet_path)
    df = table.to_pylist()
    print(f"  [code_corpus] {len(df)} raw rows; columns: {list(df[0].keys()) if df else 'none'}")
    print(f"  [code_corpus] extracting clean functions...")

    functions: List[dict] = []
    seen_code: set = set()
    for row in df:
        if len(functions) >= n:
            break
        # CodeSearchNet uses several possible column names across versions.
        code = (row.get("code")
                or row.get("func_code")
                or row.get("func_code_string")
                or row.get("whole_code_string")
                or "").strip()
        docstring = (row.get("docstring")
                     or row.get("func_documentation_string")
                     or "").strip()
        name = (row.get("func_name")
                or row.get("name")
                or "")
        if not code or not _is_clean_function(code):
            continue
        # Dedup by code content (CSN has near-duplicate functions).
        key = code[:200]
        if key in seen_code:
            continue
        seen_code.add(key)
        functions.append({"code": code, "docstring": docstring, "name": name})

    print(f"  [code_corpus] extracted {len(functions)} clean functions")
    return functions


# ---------------------------------------------------------------------- #
# Source 2: local .py files
# ---------------------------------------------------------------------- #
def _extract_from_local(local_dir: Path, n: int) -> List[dict]:
    """Walk a directory of .py files and extract top-level functions via ast."""
    print(f"  [code_corpus] scanning {local_dir} for .py files...")
    py_files = sorted(local_dir.rglob("*.py"))
    print(f"  [code_corpus] found {len(py_files)} .py files")

    functions: List[dict] = []
    seen_code: set = set()
    for path in py_files:
        if len(functions) >= n:
            break
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if len(functions) >= n:
                break
            # Extract the function source.
            code = ast.get_source_segment(src, node)
            if not code or not _is_clean_function(code):
                continue
            # Extract docstring.
            docstring = ast.get_docstring(node) or ""
            key = code[:200]
            if key in seen_code:
                continue
            seen_code.add(key)
            functions.append({
                "code": code.strip(),
                "docstring": docstring.strip(),
                "name": node.name,
            })

    print(f"  [code_corpus] extracted {len(functions)} clean functions")
    return functions


# ---------------------------------------------------------------------- #
# Filtering
# ---------------------------------------------------------------------- #
def _is_clean_function(code: str) -> bool:
    """Heuristic filter: must parse, be a def, and be a reasonable length."""
    if not code.startswith("def ") and not code.startswith("async def "):
        return False
    if len(code) < 40 or len(code) > 4000:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    # Must contain at least one function def.
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))


# ---------------------------------------------------------------------- #
# Writers
# ---------------------------------------------------------------------- #
def write_corpus(functions: List[dict], out_dir: Path) -> None:
    """Write the extracted functions in the formats the pipeline expects."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. functions.jsonl — one JSON object per function (used by generate_llm_data.py
    #    to build sessions: each function is wrapped in LLM-generated prose).
    with open(out_dir / "functions.jsonl", "w", encoding="utf-8") as fh:
        for f in functions:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    # 2. python_corpus.txt — all function bodies, one per line, whitespace-
    #    normalised. Used to train the python expert's tokenizer and for
    #    pre-training (real code distribution).
    with open(out_dir / "python_corpus.txt", "w", encoding="utf-8") as fh:
        for f in functions:
            norm = " ".join(f["code"].split())
            if norm:
                fh.write(norm + "\n")

    # 3. english_corpus.txt — all docstrings, one per line. Real code-adjacent
    #    prose for the english expert's pre-training.
    with open(out_dir / "english_corpus.txt", "w", encoding="utf-8") as fh:
        for f in functions:
            if f["docstring"]:
                norm = " ".join(f["docstring"].split())
                if norm:
                    fh.write(norm + "\n")

    print(f"  [code_corpus] wrote {len(functions)} functions to {out_dir}")
    print(f"  [code_corpus]   functions.jsonl   (for session generation)")
    print(f"  [code_corpus]   python_corpus.txt (for python expert pre-train)")
    print(f"  [code_corpus]   english_corpus.txt(for english expert pre-train)")


# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Download/extract real Python code corpus.")
    parser.add_argument("--source", choices=["csn", "local"], default="csn",
                        help="data source: 'csn' (CodeSearchNet) or 'local' (.py files)")
    parser.add_argument("--n", type=int, default=20000,
                        help="max number of functions to extract (default 20000)")
    parser.add_argument("--local-dir", type=str, default=None,
                        help="directory of .py files (for --source local)")
    parser.add_argument("--out", type=str, default=str(CORPUS_DIR),
                        help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.source == "csn":
        functions = _download_csn(args.n, out_dir)
    else:
        if not args.local_dir:
            print("Error: --source local requires --local-dir <path>")
            sys.exit(1)
        functions = _extract_from_local(Path(args.local_dir), args.n)

    if not functions:
        print("  [code_corpus] no functions extracted; nothing to write.")
        sys.exit(1)

    write_corpus(functions, out_dir)


if __name__ == "__main__":
    main()
