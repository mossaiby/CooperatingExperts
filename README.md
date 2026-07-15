# Cooperating Experts — Test Framework

A minimal, laptop-friendly implementation of the **Cooperating Experts** idea:
several small LLMs ("experts"), each with its **own vocabulary and tokenizer**,
that cooperate by emitting **switch tokens** and exchanging representations
through a **shared embedding space** via lightweight linear projections.

> Designed to run on a **4 GB VRAM** GPU (e.g. RTX 3050 Ti Laptop). A full
> `python main.py all` run (prepare → pre-train → joint → mixed) completes in
> well under an hour on such hardware.

## Architecture

```
Expert A (vocab_A) ──► [hidden h_A] ──► to_shared ──► [z ∈ shared space]
                                                          │
Expert B (vocab_B) ◄── [hidden h_B] ◄── from_shared ◄─────┘
```

- Each expert is a small decoder-only transformer (`d_model=512`, 6 layers, 8
  heads, `d_ff=2048`, `max_seq_len=512`). With the per-expert BPE vocab
  (~5.6k–5.7k tokens) this is **~22.3 M params/expert → ~44.6 M total**; fp16
  weights ≈ 90 MB. All sizes live in `config.py` (`ExpertConfig`).
- Each expert has its **own byte-level BPE tokenizer** trained on its own
  corpus, so the vocabularies are genuinely different.
- Every expert's vocab is augmented with special tokens:
  `<unk>`, `<pad>`, `<eos>`, and one `<switch:NAME>` token per expert
  (5 special tokens per expert for the 2-expert default).
- Lightweight `Linear` layers (`to_shared` / `from_shared`) map between an
  expert's hidden space (`d_model=512`) and a **bottleneck** shared space
  (`dim=256 < d_model`). The bottleneck forces the projections to learn a
  genuinely compact inter-expert representation instead of collapsing to
  identity.
- The LM head is **tied** to the input embedding table (standard for small
  models, saves parameters).

### Training pipeline (three phases)

1. **Pre-training** (`pretrain_expert`): each expert is trained **independently**
   on its own corpus with standard next-token LM loss. A 10% validation split is
   held out and val loss is logged every 200 steps so over-fitting is visible.
   **Early stopping** halts training if val loss hasn't improved for 3
   consecutive checks, so an expert never wastes steps memorising past its
   generalisation ceiling. The two experts converge at similar speeds on this
   synthetic corpus (both hit their val minimum around step 400), so both use
   400 steps; English keeps a slightly lower LR and longer warmup:
   - `python` — 400 steps, 100 warmup, lr 2e-4
   - `english` — 400 steps, 200 warmup, lr 1.5e-4 (prose is noisier)

2. **Joint fine-tuning / "stitching"** (`joint_finetune`): the transformer
   blocks are **frozen** and only the `to_shared` / `from_shared` projections
   are trained on hand-off pairs (A→B and B→A). The loss is B's LM loss on the
   continuation *plus* an alignment regularizer that encourages the round-trip
   `A.to_shared → A.from_shared` (and same for B) to be close to identity —
   the representation-alignment assumption from the model-stitching literature.

3. **Mixed end-to-end training** (`train_mixed`): **both experts are unfrozen**
   and trained on the *whole* synthetic sessions, where `<switch:NAME>` tokens
   have been inserted at every real code↔text boundary. A hidden state is
   carried through the shared space from one segment to the next (mirroring
   generation), and a real LM loss is computed over every token — including the
   switch tokens (down-weighted by `switch_loss_weight`). The model learns
   *when* to hand off between experts. Because this phase is slow (batch=1,
   ~0.1 it/s on a 4 GB GPU), it **saves the best checkpoint** (lowest smoothed
   loss) to `model_final.pt` whenever a new best appears, and **early-stops**
   once the smoothed loss fails to improve for 10 consecutive checks — so you
   never waste hours past the loss minimum.

## Experts & Datasets

The corpus is a **clean, synthetic dataset** (`synthetic_data.py`) that
simulates **5000 distinct, unique** user↔agent coding sessions. Each session
is built from a random combination of a *subject* (e.g. graph, tree, hash
map), an *action* (e.g. sorting, searching), a *context* (e.g. distributed
systems), and a *constraint* (e.g. O(n) time), with a real, parseable Python
solution drawn from a **per-subject pool of 3 code templates** (different
algorithmic approaches per subject, so the corpus is structurally diverse and
cannot be memorised as one template per subject). `generate_sessions`
guarantees uniqueness by re-rolling any duplicate session. Each agent reply
**interleaves prose with a fenced ```python block**, producing english↔python
switches inside a single reply. This gives the two experts rich switch
behaviour to train on together.

```
user    -> problem description (English prose)
agent   -> prose + ```python block + prose + ```python block + prose   (several switches)
user    -> follow-up request   (English prose)
agent   -> prose + ```python block + prose + ```python block + prose   (several switches)
user    -> closing remark      (English prose)
```

Two views of the data are written:
- **Split corpora** (`data/*_corpus_*.txt`): code pieces and text pieces
  separated, used to train each expert's tokenizer and for pre-training.
- **Combined sessions** (`data/synthetic_combined/*.txt`): the *whole*
  session preserved in order, with explicit `<switch:NAME>` markers between
  english and python segments, so switch tokens can be extracted directly
  from the full sessions.

Because the text/code alternation is explicit and clean, the segmenter splits
code from prose trivially and inserts switch tokens at exactly the right
boundaries — no noisy tool-call artifacts or duplicated text.

| Expert   | Domain        | Extracted from synthetic sessions            |
|----------|---------------|----------------------------------------------|
| `python` | Python code   | fenced ```python blocks                      |
| `english`| English prose | problem descriptions, discussion, follow-ups |

At every code↔text boundary in a session we insert a `<switch:english>` or
`<switch:python>` token, producing interleaved sequences used for joint
training.

> The synthetic data is generated locally into `data/synthetic_raw/` (no
> network needed). The number of sessions is controlled by
> `config.TrainConfig.n_sessions` (default 5000). To regenerate or change the
> size directly: `python synthetic_data.py --n 5000 --seed 42`.

## Quick Start

### Option A — Templated synthetic data (original, no API needed)

```bash
pip install -r requirements.txt

# 1. Train per-expert tokenizers (also generates + extracts the synthetic
#    corpora into data/*_corpus_*.txt as a first step)
python main.py prepare

# 2. Pre-train each expert independently (per-expert step counts)
python main.py pretrain

# 3. Joint training: stitching phase (projections only) + interleaved
#    end-to-end phase on the whole data
python main.py joint

# 4. Generate (with live expert switching)
python main.py generate "def quicksort" --expert python
python main.py generate "The user asked me to" --expert english

# Or do everything end-to-end:
python main.py all

# (Optional) just (re)generate + extract the synthetic corpora without
# training tokenizers:
python main.py extract
```

### Option B — Hybrid data pipeline (real code + LLM prose, recommended)

This produces a much better model by training the python expert on **real
Python code** (from CodeSearchNet) and using an LLM only to generate the
surrounding prose (problem descriptions, explanations, follow-ups). The
cooperation mechanism learns from natural switch boundaries between the
real code and the generated prose.

```bash
pip install -r requirements.txt   # includes openai + python-dotenv

# 1. Configure the LLM API:
cp .env-example .env
# Edit .env: set LLM_API_BASE, LLM_MODEL, LLM_API_KEY

# 2. Download a real Python code corpus (CodeSearchNet, ~20k functions):
python main.py download

# 3. Generate sessions by wrapping real code in LLM prose:
python main.py gen-data --n 10000

# 4. Train tokenizers on the hybrid corpora:
python main.py prepare-hybrid

# 5. Pre-train + joint + mixed (same as Option A):
python main.py pretrain
python main.py joint

# Or do the full hybrid pipeline in one command:
python main.py all-hybrid
```

The hybrid pipeline writes to the same `data/synthetic_raw/` and
`data/synthetic_combined/` directories, so no downstream changes are needed
— `pretrain`, `joint`, and `generate` work identically with either data source.

## Files

| File              | Purpose                                              |
|-------------------|------------------------------------------------------|
| `config.py`       | All hyper-parameters (tuned for 11 GB VRAM)         |
| `tokenizer.py`    | Per-expert BPE tokenizers + special tokens           |
| `model.py`        | The `Expert` transformer + projection layers         |
| `cooperating.py`  | `CooperatingExperts` wrapper: routing, joint loss, generation |
| `dataset.py`      | Synthetic session segmentation + windowed / hand-off / mixed datasets |
| `train.py`        | Pre-training + joint fine-tuning loops (AMP)         |
| `synthetic_data.py`| Templated synthetic corpus generator (Option A)     |
| `download_code_corpus.py` | Fetch real Python code (CodeSearchNet) for the python expert |
| `generate_llm_data.py` | Wrap real code in LLM-generated prose → sessions (Option B) |
| `main.py`         | CLI entrypoint                                       |
| `smoke_test.py`   | End-to-end smoke test (tiny config, ~1 min)          |
| `REPORT.md`       | Full technical report on how every part is trained   |
| `.env-example`    | Template for the LLM API config (copy to `.env`)    |

## Memory Budget

The default config (`config.py`) uses `d_model=768`, 8 layers, 12 heads,
`d_ff=3072`, `max_seq_len=1024` → **~67 M params per expert (~134 M total)**.
In fp16 the weights are ≈ 270 MB; with AdamW optimizer states (~12 bytes/param)
and mixed-precision activations for `batch_size=8` + gradient accumulation,
the full pipeline fits comfortably inside **11 GB VRAM** (e.g. RTX 2080 /
RTX 3060). For 4 GB GPUs, downgrade to `d_model=512`, `n_layers=6`,
`n_heads=8`, `d_ff=2048`, `max_seq_len=512`, `shared.dim=256`
(~22 M params/expert) — see the config docstring.

See **`REPORT.md`** for a full walkthrough of how every component is trained
and how the experts cooperate at inference time.
