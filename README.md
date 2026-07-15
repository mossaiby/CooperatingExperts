# Cooperating Experts — Test Framework

A minimal, laptop-friendly implementation of the **Cooperating Experts** idea:
several small LLMs ("experts"), each with its **own vocabulary and tokenizer**,
that cooperate by emitting **switch tokens** and exchanging representations
through a **shared latent space** via lightweight linear projections.

> Designed to run on a **4 GB VRAM** GPU (e.g. RTX 3050 Ti Laptop). A full
> `python main.py all` run (prepare → pre-train → joint → mixed) completes in
> well under an hour on such hardware. A CPU-only smoke test (`python
> smoke_test.py`) exercises every code path in about a minute.

## Architecture

```
Expert A (vocab_A) ──► [hidden h_A] ──► to_shared ──► [z ∈ shared space]
                                                          │
Expert B (vocab_B) ◄── [hidden h_B] ◄── from_shared ◄─────┘
```

- Each expert is a small decoder-only transformer. **Default (this repo):**
  `d_model=512`, 6 layers, 8 heads, `d_ff=2048`, `max_seq_len=512`. With the
  per-expert BPE vocab (~5.6k–5.7k real merges on the synthetic corpus) this is
  **~22.3 M params/expert → ~44.6 M total**; fp16 weights ≈ 90 MB. All sizes
  live in `config.py` (`ExpertConfig`). `Config.large()` provides a bigger
  preset (`d_model=768`, 8 layers, 12 heads, dim 384 → ~63-67 M/expert) for 11 GB
  GPUs.
- Each expert has its **own byte-level BPE tokenizer** trained on its own
  corpus, so the vocabularies are genuinely different.
- Every expert's vocab is augmented with special tokens:
  `<unk>`, `<pad>`, `<eos>`, and one `<switch:NAME>` token per expert
  (5 special tokens per expert for the 2-expert default). A "self" hand-off is
  just `<switch:<own-name>>` and is masked out during generation — there is no
  separate `<switch:self>` token.
- Lightweight `Linear` layers (`to_shared` / `from_shared`) map between an
  expert's hidden space (`d_model=512`) and a **bottleneck** shared space
  (`dim=256 < d_model`). The bottleneck forces the projections to learn a
  genuinely compact inter-expert representation instead of collapsing to
  identity.
- **Bridge width (`shared.bridge_len`).** A hand-off carries the last
  `bridge_len` hidden states of the sending expert (not just one). The default
  is `1` (the original design); set it to `K > 1` to widen the inter-expert
  channel — the receiving expert then gets `K` seed positions to attend to.
- The LM head is **tied** to the input embedding table (standard for small
  models, saves parameters).

### Training pipeline (three phases)

1. **Pre-training** (`pretrain_expert`): each expert is trained **independently**
   on its own corpus with standard next-token LM loss. A 10% validation split is
   held out; val loss is computed **every 200 steps averaged over several
   batches** (not a single noisy batch) so over-fitting is visible. **Early
   stopping** halts training if val loss hasn't improved for 3 consecutive
   checks. Per-expert overrides (`PretrainOverride`) let each expert use its own
   warmup/LR:
   - `python`  — warmup 100, lr 2e-4
   - `english` — warmup 200, lr 1.5e-4 (prose is noisier)

2. **Joint fine-tuning / "stitching"** (`joint_finetune`): the transformer
   blocks are **frozen** and only the `to_shared` / `from_shared` projections
   are trained on hand-off pairs (A→B and B→A). **These pairs are now built from
   real code↔text boundaries *within the same session*** (see
   `build_boundary_handoff_pairs`), so the continuation is semantically related
   to the prefix. The loss is B's LM loss on the continuation *plus* an
   alignment regularizer that keeps each expert's `from_shared∘to_shared`
   round-trip close to identity. (The legacy random-pairing `HandoffDataset` is
   kept only as a fallback for the offline smoke test.)

3. **Mixed end-to-end training** (`train_mixed`): **both experts are unfrozen**
   and trained on the *whole* synthetic sessions, with `<switch:NAME>` tokens
   inserted at every real code↔text boundary. Hidden states are carried through
   the shared space from one segment to the next (mirroring generation), and a
   real LM loss is computed over every token — including switch tokens
   (down-weighted by `switch_loss_weight`). Because this phase is slow (batch=1),
   it **saves the best checkpoint** (lowest smoothed loss) to `model_final.pt`
   and **early-stops** once the smoothed loss stops improving.

## Experts & Datasets

Two data pipelines are provided (see the Quick Start below):

- **Option A — templated synthetic data** (`synthetic_data.py`): ~5000 distinct
  user↔agent coding sessions built from random (subject, action, context,
  constraint) combinations with real, parseable Python drawn from a per-subject
  pool of 3 templates. Fully local, no network. **Caveat:** this corpus is
  highly templated, so pre-training losses on it mostly reflect template
  memorisation, not general language modelling — treat any Option-A numbers as
  illustrative only.
- **Option B — hybrid real code + LLM prose** (recommended): real Python from
  CodeSearchNet, with an LLM generating only the surrounding prose, so switch
  boundaries between real code and generated prose are natural.

At every code↔text boundary a `<switch:english>` / `<switch:python>` token is
inserted, producing the interleaved sequences used for mixed training.

## Quick Start

### Option A — Templated synthetic data (no API needed)

```bash
pip install -r requirements.txt

python main.py prepare      # generate + extract synthetic corpora, train tokenizers
python main.py pretrain     # pre-train each expert independently
python main.py joint        # stitching (projections) + mixed end-to-end phase
python main.py generate "def quicksort" --expert python
python main.py all          # everything end-to-end
```

### Option B — Hybrid data pipeline (real code + LLM prose)

```bash
pip install -r requirements.txt   # includes openai + python-dotenv

cp .env-example .env              # then edit LLM_API_BASE / LLM_MODEL / LLM_API_KEY
python main.py download           # real Python corpus (CodeSearchNet)
python main.py gen-data --n 10000 # wrap real code in LLM prose → sessions
python main.py prepare-hybrid     # train tokenizers on the hybrid corpora
python main.py pretrain
python main.py joint
# or: python main.py all-hybrid
```

> `.env` is gitignored and must never be committed or shared. If a key was ever
> committed or shipped in an archive, rotate it.

## Files

| File              | Purpose                                              |
|-------------------|------------------------------------------------------|
| `config.py`       | All hyper-parameters (single source of truth)        |
| `tokenizer.py`    | Per-expert BPE tokenizers + special tokens           |
| `model.py`        | The `Expert` transformer + projection layers         |
| `cooperating.py`  | `CooperatingExperts`: routing, joint/mixed loss, generation |
| `dataset.py`      | Segmentation + window / boundary-handoff / mixed datasets |
| `train.py`        | Pre-training + joint + mixed training loops (AMP)    |
| `synthetic_data.py`| Templated synthetic corpus generator (Option A)     |
| `download_code_corpus.py` | Fetch real Python code (CodeSearchNet) |
| `generate_llm_data.py` | Wrap real code in LLM-generated prose → sessions (Option B) |
| `main.py`         | CLI entrypoint                                       |
| `smoke_test.py`   | End-to-end smoke test (tiny config, CPU, ~1 min)     |
| `REPORT.md`       | Full technical report                                |
| `.env-example`    | Template for the LLM API config (copy to `.env`)     |

## Memory Budget

The default config (`config.py`, `d_model=512`, 6 layers, 8 heads, `d_ff=2048`,
`max_seq_len=512`, `shared.dim=256`) is **~22.3 M params/expert (~44.6 M
total)**. One expert trains at a time (batch=8, seq=512): fp16 weights ≈ 45 MB,
Adam states ≈ 180 MB, activations ≈ 0.8 GB, gradients ≈ 90 MB → **~1.2 GB**,
comfortable on a 4 GB laptop GPU. For an 11 GB GPU use `Config.large()`
(`d_model=768`, 8 layers, 12 heads, dim 384 → ~63-67 M/expert, ~125-134 M total).

See **`REPORT.md`** for a full walkthrough.

## License / data note

The Option-B pipeline downloads and trains on **CodeSearchNet** (real GitHub
code under a mix of open-source licenses). Review its terms before
redistributing any trained checkpoints or derived corpora. No `LICENSE` is
included in this prototype; add one before publishing.
