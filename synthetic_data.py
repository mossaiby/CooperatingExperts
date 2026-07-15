"""Synthetic "Cooperating Experts" dataset generator.

Generates a clean, fully-controlled corpus that simulates real coding
sessions between a user and an agent. Each session is a *distinct* coding
task (random subject / action / context / constraint) with its own
natural-language description, discussion, and Python solution, so the
English prose and Python code are genuinely varied.

Two views of the data are written:
  - **Raw sessions** (`data/synthetic_raw/*.jsonl`): one JSONL file per
    session, one event dict per line, in the schema the segmenter in
    `dataset.py` expects:
        {"type": "message", "session_id": "...",
         "message": {"role": "...", "content": [{"type": "text", "text": "...}]}}
    The agent's code is wrapped in a fenced ```python block so the segmenter
    can split code from prose cleanly.
  - **Combined sessions** (`data/synthetic_combined/*.txt`): the whole
    session preserved in order, with explicit `<switch:NAME>` markers
    between english and python segments (a human-readable view).

Usage:
    python synthetic_data.py                 # writes data/synthetic_raw + data/synthetic_combined
    python synthetic_data.py --n 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import DATA_DIR

SYNTHETIC_DIR = DATA_DIR / "synthetic_raw"
COMBINED_DIR = DATA_DIR / "synthetic_combined"

# Default number of distinct, unique synthetic sessions to generate.
DEFAULT_N = 5000


# ---------------------------------------------------------------------- #
# Distinct task generation. Each task combines a random subject, action,
# context, and constraint so the sessions are genuinely varied (no cycling
# through a small fixed pool of archetypes).
# ---------------------------------------------------------------------- #
_SUBJECTS = [
    "graph", "tree", "linked list", "hash map", "array", "string",
    "stack", "queue", "heap", "trie", "binary tree", "matrix",
]
_ACTIONS = [
    "sorting", "searching", "inverting", "balancing", "traversing",
    "compressing", "serializing", "parsing", "merging", "partitioning",
]
_CONTEXTS = [
    "distributed systems", "embedded devices", "web servers", "game engines",
    "data pipelines", "financial apps", "real-time systems", "compilers",
]
_CONSTRAINTS = [
    "O(n) time", "O(1) space", "a recursive approach", "an iterative approach",
    "bit manipulation", "asyncio", "memoization", "a sliding window",
]

# Concrete code templates keyed by subject so the generated Python is real,
# parseable code (not just `pass`). Each subject has SEVERAL template variants
# (different algorithmic approaches) so the corpus is structurally diverse and
# the model cannot memorise a single template per subject.
# Each template fills in the task id via `{i}`.
_CODE_TEMPLATES = {
    "graph": [
        # DFS (iterative, explicit stack)
        "def solution_{i}(graph, start):\n"
        "    \"\"\"Depth-first traversal of a graph from start.\"\"\"\n"
        "    visited = set()\n"
        "    order = []\n"
        "    stack = [start]\n"
        "    while stack:\n"
        "        node = stack.pop()\n"
        "        if node not in visited:\n"
        "            visited.add(node)\n"
        "            order.append(node)\n"
        "            stack.extend(graph.get(node, []))\n"
        "    return order\n",
        # BFS (queue)
        "from collections import deque\n\n"
        "def solution_{i}(graph, start):\n"
        "    \"\"\"Breadth-first traversal of a graph from start.\"\"\"\n"
        "    visited = {{start}}\n"
        "    q = deque([start])\n"
        "    order = []\n"
        "    while q:\n"
        "        node = q.popleft()\n"
        "        order.append(node)\n"
        "        for nxt in graph.get(node, []):\n"
        "            if nxt not in visited:\n"
        "                visited.add(nxt)\n"
        "                q.append(nxt)\n"
        "    return order\n",
        # Count nodes (recursive)
        "def solution_{i}(graph, start):\n"
        "    \"\"\"Count the nodes reachable from start.\"\"\"\n"
        "    seen = set()\n"
        "    def visit(n):\n"
        "        if n in seen:\n"
        "            return\n"
        "        seen.add(n)\n"
        "        for m in graph.get(n, []):\n"
        "            visit(m)\n"
        "    visit(start)\n"
        "    return len(seen)\n",
    ],
    "tree": [
        "def solution_{i}(root):\n"
        "    \"\"\"Inorder traversal of a binary tree.\"\"\"\n"
        "    out = []\n"
        "    def go(node):\n"
        "        if node:\n"
        "            go(node.left)\n"
        "            out.append(node.val)\n"
        "            go(node.right)\n"
        "    go(root)\n"
        "    return out\n",
        "def solution_{i}(root):\n"
        "    \"\"\"Preorder traversal of a binary tree.\"\"\"\n"
        "    out = []\n"
        "    def go(node):\n"
        "        if node:\n"
        "            out.append(node.val)\n"
        "            go(node.left)\n"
        "            go(node.right)\n"
        "    go(root)\n"
        "    return out\n",
        "def solution_{i}(root):\n"
        "    \"\"\"Return the number of nodes in a binary tree.\"\"\"\n"
        "    if not root:\n"
        "        return 0\n"
        "    return 1 + solution_{i}(root.left) + solution_{i}(root.right)\n",
    ],
    "linked list": [
        "def solution_{i}(head):\n"
        "    \"\"\"Reverse a linked list in place.\"\"\"\n"
        "    prev = None\n"
        "    while head:\n"
        "        nxt = head.next\n"
        "        head.next = prev\n"
        "        prev = head\n"
        "        head = nxt\n"
        "    return prev\n",
        "def solution_{i}(head):\n"
        "    \"\"\"Return the length of a linked list.\"\"\"\n"
        "    n = 0\n"
        "    while head:\n"
        "        n += 1\n"
        "        head = head.next\n"
        "    return n\n",
        "def solution_{i}(head):\n"
        "    \"\"\"Return the middle node value of a linked list.\"\"\"\n"
        "    slow = fast = head\n"
        "    while fast and fast.next:\n"
        "        slow = slow.next\n"
        "        fast = fast.next.next\n"
        "    return slow.val if slow else None\n",
    ],
    "hash map": [
        "def solution_{i}(items, target):\n"
        "    \"\"\"Return indices of two items that sum to target.\"\"\"\n"
        "    seen = {{}}\n"
        "    for i, x in enumerate(items):\n"
        "        if target - x in seen:\n"
        "            return seen[target - x], i\n"
        "        seen[x] = i\n"
        "    return (-1, -1)\n",
        "def solution_{i}(items):\n"
        "    \"\"\"Return the most frequent item.\"\"\"\n"
        "    counts = {{}}\n"
        "    for x in items:\n"
        "        counts[x] = counts.get(x, 0) + 1\n"
        "    return max(counts, key=counts.get)\n",
        "def solution_{i}(items):\n"
        "    \"\"\"Return the unique items, preserving order.\"\"\"\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for x in items:\n"
        "        if x not in seen:\n"
        "            seen.add(x)\n"
        "            out.append(x)\n"
        "    return out\n",
    ],
    "array": [
        "def solution_{i}(items):\n"
        "    \"\"\"Return a sorted copy of items.\"\"\"\n"
        "    return sorted(items)\n",
        "def solution_{i}(items):\n"
        "    \"\"\"Return the maximum item.\"\"\"\n"
        "    best = items[0]\n"
        "    for x in items[1:]:\n"
        "        if x > best:\n"
        "            best = x\n"
        "    return best\n",
        "def solution_{i}(items):\n"
        "    \"\"\"Return the cumulative sums of items.\"\"\"\n"
        "    out = []\n"
        "    total = 0\n"
        "    for x in items:\n"
        "        total += x\n"
        "        out.append(total)\n"
        "    return out\n",
    ],
    "string": [
        "def solution_{i}(s):\n"
        "    \"\"\"Return s reversed.\"\"\"\n"
        "    return s[::-1]\n",
        "def solution_{i}(s):\n"
        "    \"\"\"Return True if s is a palindrome.\"\"\"\n"
        "    return s == s[::-1]\n",
        "def solution_{i}(s):\n"
        "    \"\"\"Return the character counts of s.\"\"\"\n"
        "    counts = {{}}\n"
        "    for ch in s:\n"
        "        counts[ch] = counts.get(ch, 0) + 1\n"
        "    return counts\n",
    ],
    "stack": [
        "def solution_{i}(tokens):\n"
        "    \"\"\"Evaluate a postfix expression.\"\"\"\n"
        "    st = []\n"
        "    for t in tokens:\n"
        "        if t in '+-*/':\n"
        "            b = st.pop(); a = st.pop()\n"
        "            st.append(a + b if t == '+' else a - b)\n"
        "        else:\n"
        "            st.append(int(t))\n"
        "    return st[0] if st else 0\n",
        "def solution_{i}(items):\n"
        "    \"\"\"Return items with duplicates removed, last occurrence kept.\"\"\"\n"
        "    st = []\n"
        "    for x in items:\n"
        "        if x in st:\n"
        "            st.remove(x)\n"
        "        st.append(x)\n"
        "    return st\n",
        "def solution_{i}(text):\n"
        "    \"\"\"Return text with balanced brackets removed.\"\"\"\n"
        "    st = []\n"
        "    for ch in text:\n"
        "        if ch in '([':\n"
        "            st.append(ch)\n"
        "        elif ch in ')]' and st:\n"
        "            st.pop()\n"
        "    return ''.join(st)\n",
    ],
    "queue": [
        "from collections import deque\n\n"
        "def solution_{i}(graph, start):\n"
        "    \"\"\"Breadth-first traversal of a graph from start.\"\"\"\n"
        "    visited = {{start}}\n"
        "    q = deque([start])\n"
        "    order = []\n"
        "    while q:\n"
        "        node = q.popleft()\n"
        "        order.append(node)\n"
        "        for n in graph.get(node, []):\n"
        "            if n not in visited:\n"
        "                visited.add(n)\n"
        "                q.append(n)\n"
        "    return order\n",
        "from collections import deque\n\n"
        "def solution_{i}(items):\n"
        "    \"\"\"Return the items rotated left by one position.\"\"\"\n"
        "    if not items:\n"
        "        return items\n"
        "    q = deque(items)\n"
        "    q.rotate(-1)\n"
        "    return list(q)\n",
        "from collections import deque\n\n"
        "def solution_{i}(items, k):\n"
        "    \"\"\"Return the last k items in order.\"\"\"\n"
        "    q = deque(items)\n"
        "    out = []\n"
        "    while q and len(out) < k:\n"
        "        out.append(q.pop())\n"
        "    return out[::-1]\n",
    ],
    "heap": [
        "import heapq\n\n"
        "def solution_{i}(items, k):\n"
        "    \"\"\"Return the k smallest elements.\"\"\"\n"
        "    return heapq.nsmallest(k, items)\n",
        "import heapq\n\n"
        "def solution_{i}(items):\n"
        "    \"\"\"Return the items sorted ascending.\"\"\"\n"
        "    return heapq.nsmallest(len(items), items)\n",
        "import heapq\n\n"
        "def solution_{i}(items):\n"
        "    \"\"\"Return the maximum element.\"\"\"\n"
        "    if not items:\n"
        "        return None\n"
        "    return heapq.nlargest(1, items)[0]\n",
    ],
    "trie": [
        "def solution_{i}(words):\n"
        "    \"\"\"Build a trie from a list of words.\"\"\"\n"
        "    root = {{}}\n"
        "    for w in words:\n"
        "        node = root\n"
        "        for ch in w:\n"
        "            node = node.setdefault(ch, {{}})\n"
        "        node['#'] = True\n"
        "    return root\n",
        "def solution_{i}(words):\n"
        "    \"\"\"Return the set of unique first characters.\"\"\"\n"
        "    return list({{w[0] for w in words if w}})\n",
        "def solution_{i}(words, prefix):\n"
        "    \"\"\"Return the words that start with prefix.\"\"\"\n"
        "    return [w for w in words if w.startswith(prefix)]\n",
    ],
    "binary tree": [
        "def solution_{i}(root):\n"
        "    \"\"\"Return the height of a binary tree.\"\"\"\n"
        "    if not root:\n"
        "        return 0\n"
        "    return 1 + max(solution_{i}(root.left), solution_{i}(root.right))\n",
        "def solution_{i}(root):\n"
        "    \"\"\"Return True if the tree is balanced.\"\"\"\n"
        "    def height(n):\n"
        "        if not n:\n"
        "            return 0\n"
        "        return 1 + max(height(n.left), height(n.right))\n"
        "    def check(n):\n"
        "        if not n:\n"
        "            return True\n"
        "        l, r = height(n.left), height(n.right)\n"
        "        return abs(l - r) <= 1 and check(n.left) and check(n.right)\n"
        "    return check(root)\n",
        "def solution_{i}(root):\n"
        "    \"\"\"Return the sum of all node values.\"\"\"\n"
        "    if not root:\n"
        "        return 0\n"
        "    return root.val + solution_{i}(root.left) + solution_{i}(root.right)\n",
    ],
    "matrix": [
        "def solution_{i}(matrix):\n"
        "    \"\"\"Return the transpose of a matrix.\"\"\"\n"
        "    return [list(row) for row in zip(*matrix)]\n",
        "def solution_{i}(matrix):\n"
        "    \"\"\"Return the main diagonal of a square matrix.\"\"\"\n"
        "    return [matrix[r][r] for r in range(len(matrix))]\n",
        "def solution_{i}(matrix):\n"
        "    \"\"\"Return the sum of all elements in a matrix.\"\"\"\n"
        "    return sum(sum(row) for row in matrix)\n",
    ],
}

# Fallback templates for any subject without a specific entry.
_CODE_FALLBACK = [
    "def solution_{i}(data):\n"
    "    \"\"\"Process data.\"\"\"\n"
    "    result = []\n"
    "    for item in data:\n"
    "        result.append(item)\n"
    "    return result\n",
    "def solution_{i}(data):\n"
    "    \"\"\"Return the length of data.\"\"\"\n"
    "    return len(data)\n",
]

# Follow-up requests and closings to vary the dialogue shape.
_FOLLOWUPS = [
    "That works, but can you make it handle the empty input case safely?",
    "Nice. Could you also add a short docstring and a type hint?",
    "Thanks! Now extend it so it also reports how many steps it took.",
    "Good. What happens with very large inputs? Can you make it linear?",
    "Could you rewrite it without using built-in helpers, for clarity?",
    "Please add input validation and raise a clear error on bad input.",
    "Can you make the API a bit more general so it works on other types too?",
    "That's close. Make the edge cases explicit with a comment in the code.",
]

_CLOSINGS = [
    "Great, that is exactly what I needed. Thank you!",
    "Perfect, this is much clearer now.",
    "Awesome, I understand the approach now.",
    "Thanks, that handles the edge cases well.",
    "Excellent, the extra examples really help.",
]

_TRANSITIONS = [
    "Let me show the core implementation first.",
    "I will start with the main routine.",
    "The key idea is captured in the function below.",
    "We can express this cleanly in a single function.",
    "Here is a straightforward way to do it.",
    "The implementation is shorter than it looks.",
    "Let's walk through the logic step by step.",
    "I will keep the code minimal and readable.",
]


def _generate_distinct_task(i: int, rng: random.Random) -> Dict[str, str]:
    """Generate one distinct coding task as a dict with name/problem/discussion/code."""
    sub = rng.choice(_SUBJECTS)
    act = rng.choice(_ACTIONS)
    ctx = rng.choice(_CONTEXTS)
    con = rng.choice(_CONSTRAINTS)

    name = f"{act} a {sub}"
    problem = (
        f"Write a Python function for {act} a {sub} in the context of {ctx}. "
        f"It must follow {con}. (task {i})"
    )
    discussion = (
        f"To perform {act} on a {sub}, we first initialize the structure and "
        f"then apply the {con} logic. This keeps the routine clear and efficient."
    )
    templates = _CODE_TEMPLATES.get(sub, _CODE_FALLBACK)
    template = rng.choice(templates)
    code = template.format(i=i)
    return {"name": name, "problem": problem, "discussion": discussion, "code": code}


# ---------------------------------------------------------------------- #
# Session construction (event schema expected by dataset.py)
# ---------------------------------------------------------------------- #
def _build_agent_reply(task: Dict[str, str], rng: random.Random) -> str:
    """Build one agent reply that interleaves prose with a fenced python block."""
    code = task["code"].rstrip("\n")
    opener = rng.choice(_TRANSITIONS)
    transition = rng.choice(_TRANSITIONS)
    parts = [
        opener,
        f"```python\n{code}\n```",
        transition,
    ]
    return "\n\n".join(parts)


def _make_session(rng: random.Random, idx: int, task: Dict[str, str]) -> List[dict]:
    """Build one clean user<->agent session as a list of event dicts.

    The schema matches what dataset._event_to_pieces expects:
        {"type": "message", "session_id": "...",
         "message": {"role": "...", "content": [{"type": "text", "text": "...}]}}
    """
    sid = f"synthetic-{idx:05d}"
    events: List[dict] = [
        {
            "type": "message",
            "session_id": sid,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": task["problem"]}],
            },
        },
        {
            "type": "message",
            "session_id": sid,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": _build_agent_reply(task, rng)}],
            },
        },
    ]

    # ~60% of sessions get a follow-up + second reply.
    if rng.random() < 0.6:
        followup = rng.choice(_FOLLOWUPS)
        events.append({
            "type": "message",
            "session_id": sid,
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": followup}],
            },
        })
        events.append({
            "type": "message",
            "session_id": sid,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": _build_agent_reply(task, rng)}],
            },
        })

    closing = rng.choice(_CLOSINGS)
    events.append({
        "type": "message",
        "session_id": sid,
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": closing}],
        },
    })
    return events


def generate_sessions(n: int = DEFAULT_N, seed: int = 42) -> List[Tuple[str, List[dict]]]:
    """Generate `n` distinct, unique synthetic sessions as (session_id, events) pairs.

    Uniqueness is guaranteed by tracking the full rendered text of each
    session and re-rolling any duplicate (different task / follow-up / closing
    combination) until `n` unique sessions are produced. The session id is the
    zero-padded index of the produced session (synthetic-00000 ...), so ids
    are unique by construction.
    """
    rng = random.Random(seed)
    sessions: List[Tuple[str, List[dict]]] = []
    seen_texts: set = set()
    i = 0
    attempts = 0
    max_attempts = n * 20  # generous bound to avoid infinite loops
    while len(sessions) < n and attempts < max_attempts:
        attempts += 1
        task = _generate_distinct_task(i, rng)
        events = _make_session(rng, i, task)
        # Render the session text to detect duplicates.
        text = json.dumps(events, ensure_ascii=False, sort_keys=True)
        if text in seen_texts:
            # Duplicate: bump the task index and retry with a fresh draw.
            i += 1
            continue
        seen_texts.add(text)
        sessions.append((f"synthetic-{i:05d}", events))
        i += 1
    if len(sessions) < n:
        raise RuntimeError(
            f"Could only generate {len(sessions)} unique sessions after "
            f"{attempts} attempts (requested {n}). Increase the variety of "
            f"task components or the seed."
        )
    return sessions


# ---------------------------------------------------------------------- #
# Writers
# ---------------------------------------------------------------------- #
def write_synthetic(out_dir: Path = SYNTHETIC_DIR, n: int = DEFAULT_N, seed: int = 42) -> Path:
    """Write synthetic sessions to out_dir as one JSONL file per session.

    Each line of a file is one event dict (matching the loader schema).
    Returns the output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = generate_sessions(n=n, seed=seed)
    for sid, events in sessions:
        path = out_dir / f"{sid}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"  [synthetic] wrote {len(sessions)} sessions to {out_dir}")
    return out_dir


# Local fence-splitter so write_combined can reuse the same segmentation as
# the dataset loader without importing dataset (avoids a circular import).
_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)


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


def write_combined(out_dir: Path = COMBINED_DIR, n: int = DEFAULT_N, seed: int = 42) -> Path:
    """Write the combined view of every session: all pieces in order, with
    explicit `<switch:NAME>` markers between english and python segments.

    One file per session, one line per segment, tagged with its expert and
    whether it ends in a switch token. This is a human-readable view; the
    training pipeline reconstructs segments directly from synthetic_raw.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sessions = generate_sessions(n=n, seed=seed)
    for sid, events in sessions:
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
    print(f"  [synthetic] wrote {len(sessions)} combined sessions to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sessions.")
    parser.add_argument("--n", type=int, default=100, help="number of sessions")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, default=str(SYNTHETIC_DIR),
                        help="output dir for raw sessions")
    parser.add_argument("--combined-out", type=str, default=str(COMBINED_DIR),
                        help="output dir for combined sessions")
    args = parser.parse_args()

    write_synthetic(Path(args.out), n=args.n, seed=args.seed)
    write_combined(Path(args.combined_out), n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()