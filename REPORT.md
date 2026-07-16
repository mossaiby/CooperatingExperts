# Cooperating Experts — Full Technical Report

This document describes, in detail, how every component of the Cooperating
Experts framework is built, trained, and used at inference time. It is the
authoritative reference for the implementation; `README.md` is the quick-start
summary.

---

## 1. Overview

Cooperating Experts is a multi-expert language-model architecture in which
several small decoder-only transformers ("experts") — each with its **own
vocabulary and tokenizer** — cooperate to generate a single coherent sequence.
Cooperation happens through two mechanisms:

1. **Switch tokens.** Every expert's vocabulary contains one
   `<switch:NAME>` token per expert. When an expert emits a switch token,
   control (and a carried hidden state) is handed to the target expert, which
   continues generating in its own vocabulary.
2. **Shared embedding space.** Each expert has two lightweight linear
   projections — `to_shared` (d_model → shared_dim) and `from_shared`
   (shared_dim → d_model). At a switch boundary the *current* expert's last
   hidden state is projected into the shared space `z`; the *next* expert
   projects `z` back into its own hidden space and continues from there.

The shared space is a **bottleneck** (`dim=256 < d_model=512`), which forces
the projections to learn a genuinely compact inter-expert representation
instead of collapsing to identity. The **bridge width** `shared.bridge_len` (K)
controls how many hidden states are carried across a hand-off: K=1 is the
original single-vector channel; K>1 carries the last K hidden states, giving the
receiving expert several seed positions to attend to. A single vector is a very
narrow channel, so `bridge_len` is exposed for experimentation.

### The two experts

| Expert    | Domain         | Tokenizer trained on            |
|-----------|----------------|---------------------------------|
| `python`  | Python code    | fenced ```python blocks         |
| `english` | English prose  | problem descriptions, discussion |

Each expert is a 6-layer, 8-head, `d_model=512` decoder-only transformer with
**rotary position embeddings (RoPE)** and tied input/output embeddings, giving
**~22.3 M params per expert (~44.6 M total)** in the default config. RoPE (no
absolute position table) makes the hand-off seed positions collision-free and
removes the `max_seq_len` index-overflow fragility of an absolute table.

All three cooperation paths (`joint_loss`, `mixed_loss`, `generate`) use a
**single unified hand-off convention**: after a switch, a leading `<pad>`
"hand-off query" position predicts the first token of the receiving expert,
for both the seed-prepend and cross-attention bridges. This guarantees
training matches inference exactly (locked in by a smoke-test assertion).

---

## 2. Data generation (`synthetic_data.py`)

### 2.1 Session structure

The corpus is a **clean, synthetic dataset** of **5000 distinct, unique**
user↔agent coding sessions. Each session is built from a random combination of:

- a **subject** (graph, tree, linked list, hash map, array, string, stack,
  queue, heap, trie, binary tree, matrix),
- an **action** (sorting, searching, traversing, reversing, …),
- a **context** (distributed systems, embedded firmware, game engine, …),
- a **constraint** (O(n) time, O(1) space, recursion, memoization, …).

`generate_sessions(n)` guarantees uniqueness: it keeps a deduplication set and
re-rolls any session whose (subject, action, context, constraint, template,
task-id) signature has already been emitted. Session ids are zero-padded
(`:05d`).

### 2.2 Code template diversity

Each subject maps to a **list of 3 code templates** representing different
algorithmic approaches (e.g. for "graph": iterative DFS, BFS with a queue, and
recursive node-count). `_generate_distinct_task` picks a random variant via
`rng.choice`, so the corpus is structurally diverse and the model cannot
memorise a single template per subject. Every generated code snippet is real,
parseable Python (verified with `ast.parse` during testing).

### 2.3 Event schema and the two data views

Each session is emitted as a sequence of **events** in a simple schema that
`dataset.py` consumes:

```
user    -> problem description (English prose)
agent   -> prose + ```python block + prose + ```python block + prose
user    -> follow-up request (English prose)
agent   -> prose + ```python block + prose
user    -> closing remark (English prose)
```

Two views are written to disk:

- **Split corpora** (`data/*_corpus_*.txt`): code pieces and text pieces
  separated into per-expert shards. Used to train each expert's tokenizer and
  for pre-training.
- **Combined sessions** (`data/synthetic_combined/*.txt`): the whole session
  preserved in order, with explicit `<switch:NAME>` markers between english
  and python segments. Used for mixed end-to-end training.

Because the text/code alternation is explicit and clean, the segmenter splits
code from prose trivially and inserts switch tokens at exactly the right
boundaries — no noisy tool-call artifacts.

---

## 3. Tokenizers (`tokenizer.py`)

Each expert gets its **own byte-level BPE tokenizer** trained on its own
corpus via the HuggingFace `tokenizers` library:

- `ByteLevel` pre-tokenizer + `ByteLevelDecoder` + `ByteLevelPostProcessor`.
- BPE vocab capped at `ExpertConfig.vocab_size = 12000` merges; the trainer
  stops at the actual number of unique merge pairs found (~5.6k for python,
  ~5.7k for english on 5000 sessions).
- Special tokens are appended **on top of** the BPE vocab:
  `<pad>`, `<eos>`, and one `<switch:NAME>` per expert (including a
  `<switch:self>` token). For the 2-expert default this is 4 special tokens
  per expert (`NUM_SPECIAL_TOKENS_PER_EXPERT = 4`). There is intentionally
  **no `<unk>`** token: byte-level BPE has no out-of-vocabulary tokens (every
  byte is in the base alphabet), so an `<unk>` would be unreachable.

Because each tokenizer is trained on a different corpus, the vocabularies are
genuinely different — the python expert's token for `def` is unrelated to any
english token. The switch tokens are the only shared "vocabulary" across
experts, and even their *ids* differ per expert (each expert has its own
`switch_id(target)` mapping).

---

## 4. The Expert model (`model.py`)

Each expert is a standard decoder-only transformer with rotary positions:

```
tok_emb  →  [Block × n_layers]  →  ln_f  →  head (tied to tok_emb)
            (RoPE applied to q/k inside attention; no absolute pos_emb table)
```

### 4.1 Components

- **`CausalSelfAttention`**: fused QKV projection (`Linear → 3*d_model`),
  multi-head scaled-dot-product attention with a causal mask, output
  projection. `n_heads=8`, `head_dim=64`. **Rotary position embeddings (RoPE)**
  are applied to the query and key vectors before the attention scores are
  computed (`_rope_cos_sin` / `_apply_rope`), so position is encoded
  *relatively*. `head_dim` must be even (asserted). Because position is
  relative, prepended seed states and real tokens never collide on an absolute
  index, and no position table can be indexed out of range.
- **`FeedForward`**: two linear layers (`d_model → d_ff → d_model`) with GELU
  and dropout. `d_ff=2048`.
- **`Block`**: pre-LayerNorm residual blocks
  (`x = x + attn(ln1(x))`, `x = x + ff(ln2(x))`).
- **`Expert`**: token embeddings (position via RoPE, no `pos_emb` table),
  `n_layers=6` blocks, final LayerNorm, and a **tied** LM head
  (`head.weight = tok_emb.weight`). At init, the residual output projections
  (`attn.proj`, `ff.fc2`) are scaled by `1/sqrt(2*n_layers)` (GPT-2 style) so
  the residual-stream variance does not grow with depth — this matters for the
  deeper `Config.large()` preset.

### 4.2 Shared-space projections

Each `Expert` owns two small linear layers:

- `to_shared: Linear(d_model=512, shared_dim=256, bias=False)` — compress a
  hidden state into the shared space.
- `from_shared: Linear(shared_dim=256, d_model=512, bias=False)` — expand a
  shared-space vector back into this expert's hidden space.

These are the *only* parameters trained during the joint stitching phase.

### 4.3 Key methods

| method | purpose |
|---|---|
| `forward(ids)` | standard LM forward → `(logits, hidden)` |
| `encode(ids)` | last-layer hidden states `[B,T,d_model]` (no head) |
| `logits_from_hidden(h)` | map a hidden state → vocab logits (used after a switch) |
| `to_shared_space(h)` | `h [..,d] → z [..,shared_dim]` |
| `from_shared_space(z)` | `z [..,shared_dim] → h [..,d]` |
| `next_token_logits(ids)` | logits for the last position only (generation) |

The causal mask is built **dynamically** in `_blocks` so it works for any
sequence length — including the `+K` seed positions prepended when carried
hidden states seed a segment, and the `+1` `<pad>` hand-off query position.

### 4.4 CALM-style cross-attention bridge (optional)

When `SharedSpaceConfig.cross_attn = True`, each `Expert` additionally owns a
**`CrossAttention`** block — a CALM (Confident Adaptive Language Modeling,
Schuster et al. 2022) cross-attention layer that gives the receiving expert a
richer inter-expert channel than the linear `from_shared` seed.

**Motivation.** The default bridge carries the sender's last `K` hidden states
through the shared bottleneck and *prepends* them as virtual seed positions
(§4.3, `encode_with_seed`). That works, but every real token only sees the
carried states as fixed prefix context. CALM's idea — let an auxiliary head
*query* a deeper/other computation's states — maps naturally here: the
receiving expert's own hidden states become **queries**, and the *other*
expert's carried states become **keys/values**. Every position in the
receiving segment can then attend to all `K` carried sender states in a
content-addressable way, instead of only as a static prefix.

**Block layout** (`CrossAttention` in `model.py`):

```
q = q_proj(LayerNorm(x))          # x: [B, T, d_model]  (receiving expert's own hidden)
k = k_proj(LayerNorm(mem))        # mem: [B, S, d_model] (carried states, already in this expert's d_model)
v = v_proj(LayerNorm(mem))
att = softmax(q kᵀ / √head_dim) v
out = LayerNorm(x + out_proj(att))   # residual + final LayerNorm (CALM-style)
```

- `q_proj` / `k_proj` / `v_proj` / `out_proj` are independent `Linear(d_model,
  d_model, bias=False)`; `n_heads = shared.cross_attn_n_heads` (must divide
  `d_model`), `dropout = shared.cross_attn_dropout`.
- **No causal mask** inside cross-attention: the carried memory is a fixed set
  of sender states that every query position may attend to fully (it is
  "past" context from the other expert).
- The block is applied **after** the expert's own transformer blocks (so the
  expert first forms its own representation, then refines it by attending to
  the sender's memory — mirroring CALM, where a deeper layer's state informs an
  earlier/auxiliary prediction). When `cross_attn_residual = True` (default)
  the output is added residually; the carried states are **never** prepended as
  seed positions in this mode.

**Entry point**: `Expert.encode_with_cross_attn(ids, memory)` — `memory` is the
carried states already projected back into this expert's `d_model` via
`from_shared_space` (`[B, S, d_model]`). When `memory is None` it falls back to
a plain `encode(ids)`. The output is `[B, T, d_model]` with **no seed prefix**,
so the caller indexes logits uniformly (`h[:, :-1, :]` vs `ids[:, 1:]`).

**Config knobs** (all in `SharedSpaceConfig`):

| knob | default | meaning |
|---|---|---|
| `cross_attn` | `False` | master switch for the CALM bridge |
| `cross_attn_n_heads` | `8` | heads in the cross-attn block (must divide `d_model`) |
| `cross_attn_dropout` | `0.1` | dropout inside the cross-attn block |
| `cross_attn_residual` | `True` | residual-add the cross-attn output (CALM-style) |

**Routing.** `CooperatingExperts` exposes `_cross_attn_enabled()`,
`_carry_through_shared()`, `_handoff_query_ids()`, and `_encode_receiver()`
helpers that route `joint_loss`, `mixed_loss`, and `generate` through the
**same** unified hand-off convention (a leading `<pad>` query predicts the
first destination token) for both the cross-attn and seed-prepend paths. The
two bridge modes are mutually exclusive per run (set by `shared.cross_attn`).
The smoke test (`smoke_test.py`) exercises **both** modes and asserts the
training path and a manual replay of generation produce identical first-token
logits.

---

## 5. The CooperatingExperts wrapper (`cooperating.py`)

`CooperatingExperts` is an `nn.Module` container that holds all experts and
implements routing, the three loss functions, and generation.

### 5.1 Pre-training loss (`pretrain_loss`)

Standard causal next-token LM loss for one expert:

```python
logits, _ = exp(ids[:, :-1])
loss = F.cross_entropy(logits, ids[:, 1:], ignore_index=pad_id)
```

No shared space, no switching — each expert learns its own language
independently.

### 5.2 Joint hand-off loss (`joint_loss`)

This is the **stitching** loss. Given a prefix `ids_a` (expert A) and a
continuation `ids_b` (expert B), with bridge width `K = shared.bridge_len`:

1. **Encode the prefix with A**, take the last `k = min(K, Ta)` hidden states
   (the tokens right at the hand-off boundary): `seed_src = A.encode(ids_a)[:, -k:, :]`.
2. **Project to shared space**: `z_a = A.to_shared_space(seed_src)`  →  `[B, k, 256]`.
3. **Project into B's space**: `seed_b = B.from_shared_space(z_a)`  →  `[B, k, d_b]`.
4. **Run B under the unified hand-off convention** via `_encode_receiver`: a
   leading `<pad>` "hand-off query" position predicts B's first token, then each
   real token predicts its successor. This is identical for both bridge modes
   — in the seed-prepend path the carried states occupy virtual positions
   `0..k-1` and the query sits right after them; in the cross-attention path
   (`shared.cross_attn=True`) B's own hidden states cross-attend to `seed_b` as
   memory. In both cases the returned logits predict all of `ids_b`.
5. **LM loss**: `F.cross_entropy(logits_b, ids_b, ignore_index=pad_b)`. Because
   the convention matches generation exactly, there is no path-specific index
   offset to get wrong.
6. **Alignment regularizer** — two complementary terms:
   - **Round-trip identity** (per expert): encourages each expert's own
     `from_shared∘to_shared` to be near identity,
     `||A.from_shared(A.to_shared(ref_a)) - ref_a||²` (and same for B), using
     detached hidden states already computed (A's boundary states and B's own
     hidden over its first `k` tokens). This is the invertibility assumption
     from the model-stitching literature.
   - **Cross-expert alignment** (the space is genuinely *shared*): A's boundary
     shared code `z_a` and B's continuation shared code
     `z_b = B.to_shared_space(ref_b_hidden)` are L2-normalized and pulled
     together via `1 - cos(z_a, z_b)`. Round-trip identity alone allows each
     expert to occupy a disjoint subspace; this term forces paired boundary
     states into the *same* region of the bottleneck.

`total = lm_loss + align_weight * (align_roundtrip + align_cross)`, with
`align_weight = 0.1` by default.

> **Hand-off pairs come from real boundaries.** The prefix/continuation pairs
> are built by `dataset.build_boundary_handoff_pairs` from adjacent code↔text
> segments *within the same session*, so B's continuation is semantically
> related to A's prefix. (An earlier version paired random, unrelated snippets,
> which gave the projection no real continuation signal to learn.)

### 5.3 Mixed interleaved loss (`mixed_loss`)

This is the **end-to-end** loss over whole switch-token-annotated sessions.
Each example is a list of `(expert_name, ids)` segments. The loss mirrors
generation exactly, using the same `_encode_receiver` hand-off convention:

1. For each segment, run it through `_encode_receiver(name, exp, ids, carried)`.
   The first segment has `carried=None` and is a plain causal LM
   (`logits[:, :-1]` predict `ids[:, 1:]`). Every subsequent segment prepends a
   `<pad>` "hand-off query" whose output predicts the segment's first token, so
   all of `ids` is a target. In the seed-prepend bridge the `K` carried states
   sit as virtual positions before the query; in the cross-attention bridge the
   carried states are consumed as cross-attention memory (no seed prefix). RoPE
   means no absolute position table is ever indexed out of range regardless of
   `K`.
2. Compute LM loss over the segment's real-token targets. **Switch tokens are
   real targets** but down-weighted by `switch_loss_weight` (0.1) using a
   **per-token** mask: an unreduced cross-entropy is computed and each position
   whose *target* is a switch token is scaled by `switch_loss_weight`, then the
   result is renormalized over the (weighted) valid positions. This is a change
   from the earlier per-vocab-class `weight=` vector, which also rescaled the
   overall loss magnitude and made the reported `mixed` loss non-comparable to
   the pretrain/joint LM loss. The per-token mask affects only the switch
   positions' contribution, so the number stays comparable across phases.
3. **Carry the last `K` hidden states** through the shared space to the next
   expert: `z = exp.to_shared_space(h[:, -K:, :])`, then
   `carried = next_expert.from_shared_space(z)`. The carried states are
   **detached** so gradients don't flow back through the previous expert's
   entire transformer via `from_shared` — this bounds memory for long
   batch=1 sessions and matches generation (where the carried state is
   produced under `no_grad`).
4. The total loss is the **mean of per-segment losses**
   (`torch.stack(seg_losses).mean()`), so gradients flow to both experts.

### 5.4 Generation (`generate`)

Autoregressive sampling with live expert switching:

1. Encode the prompt in the starting expert's vocab.
2. At each step, encode the current sequence (prepending the carried state if
   any), take the last-position logits. Just after a switch the sequence is
   empty, so it is seeded with a single `<pad>` "hand-off query" position
   — the **same convention used in `_encode_receiver`** during training — for
   both bridge modes (seed-prepend and cross-attention). This is what makes
   training and inference produce identical first-token logits.
3. **Mask out the self-switch** `<switch:<active>>` (no-op loops) and, once the
   switch budget is exhausted, mask out all switch tokens (`max_switches = 4`).
4. Apply temperature + top-k sampling.
5. If the sampled token is a `<switch:NAME>`:
   - Carry the last `K = bridge_len` hidden states through the shared space:
     `z = exp.to_shared_space(h[:, -K:, :])`, `carried = target.from_shared_space(z)`.
   - Switch active expert to the target.
   - **Reset the context to empty** — the new expert starts from just the
     `K` carried states (no lossy re-tokenization of the old history in the new
     tokenizer). The carried states *are* the hand-off signal, by design.
6. If the sampled token is EOS, stop.
7. Otherwise append the token and continue.
8. The output is rendered by grouping consecutive tokens by expert and
   decoding each run with its own tokenizer.

---

## 6. Training (`train.py`)

All three phases use mixed-precision (fp16) training with `torch.amp`, AdamW,
and a cosine LR schedule with linear warmup (`_cosine_schedule`).

### 6.1 Phase 1 — Pre-training (`pretrain_expert`)

Each expert is trained **independently** on its own corpus.

**Data**: `WindowDataset` samples fixed-length windows (`max_seq_len=512`)
from the expert's corpus, up to `pretrain_max_windows=40000` windows.

**Validation split**: 10% of the windows are held out via `random_split`
(seeded by `cfg.train.seed`). Every `pretrain_val_every=200` steps, the val
loss is computed under `no_grad`/`eval()` and logged. A rising val loss while
train loss falls is the definitive over-fitting signal.

**Early stopping**: if val loss hasn't improved by at least
`pretrain_val_min_delta` for `pretrain_early_stop_patience=3` consecutive val
checks, training halts before `steps_max`. This auto-catches the val minimum
regardless of the configured step count, so an expert never wastes steps
memorising past its generalisation ceiling. (Setting `patience=0` disables
it and runs the full `steps_max`.)

**Per-expert overrides** (`PretrainOverride`): because the two experts may
converge at different rates, each gets its own step count / warmup / LR.
Any field left as `None` falls back to the global `TrainConfig` default.

| expert   | steps_max | warmup | lr     | observed val @ end |
|----------|-----------|--------|--------|--------------------|
| python   | 400       | 100    | 2e-4   | ~0.29 (converged)  |
| english  | 400       | 200    | 1.5e-4 | ~0.62 (converged)  |

> Note: an earlier run gave English 1500 steps on the assumption that prose
> is slower to fit. In practice the synthetic prose is templated enough
(subject/action/context/constraint fill-in sentences) that English over-fits
> just as fast as the code — its val minimum was also around step 400, and
> running to 1500 drove val loss from 0.62 up to 1.02 (severe over-fitting).
> Both experts now stop at 400 steps; early stopping guards against this
> automatically if the corpus or config changes.

**Loop**: cycle the train loader until `steps_max` optimizer steps. Each step:
forward → `pretrain_loss` → backward (scaled) → unscale → grad-clip → step →
update LR via cosine schedule.

**Checkpoint**: the pre-trained weights are saved as `model_pretrained.pt`.

### 6.2 Phase 2 — Joint stitching (`joint_finetune`)

Only the **projection layers** are trained; the transformer blocks are frozen.

**Data**: `build_boundary_handoff_pairs` builds (prefix_A, continuation_B) pairs
in both directions (A→B and B→A) from **real code↔text boundaries within each
session**, up to `joint_max_pairs=8000` each. (The legacy random-pairing
`HandoffDataset` remains only as a fallback for the offline smoke test.)

**Loss**: `joint_loss` (§5.2) — B's LM loss on the continuation plus the
alignment regularizer. Both directions are computed per step and summed.

**Optimizer**: AdamW over just the `to_shared` / `from_shared` weights of
both experts. `joint_lr=1e-4`, `joint_grad_clip=1.0`, cosine schedule with
100-step warmup over `joint_steps_max=3000` steps.

**What it achieves**: the shared-space projections converge so the two
experts' hidden states are mapped into a common 256-dim bottleneck that both
can read. Observed: `align` loss drops ~5× (2.13 → 0.40), `lm` loss plateaus
around 0.47–0.48 without over-fitting.

**Checkpoint**: saved as `model_stitched.pt`. After this phase all parameters
are re-unfrozen for saving / later use.

### 6.3 Phase 3 — Mixed end-to-end (`train_mixed`)

**All 44.6 M parameters are unfrozen** and trained on whole interleaved
sessions.

**Data**: `MixedDataset` builds switch-token-annotated sessions from
`synthetic_combined`, up to `mixed_max_sessions=5000`. Each example is a list
of `(expert_name, ids)` segments.

**Loss**: `mixed_loss` (§5.3) — mean per-segment LM loss with switch tokens
down-weighted by `switch_loss_weight=0.1`.

**Optimizer**: AdamW over all parameters. Because segments are
variable-length, `batch_size=1` with `grad_accum=16` gives an effective batch
of 16. Each optimizer step = 16 accumulated forward/backward passes.

**Stabilization** (tuned after observing divergence with `lr=3e-5`):

| knob | value | rationale |
|---|---|---|
| `mixed_lr` | 1e-5 | full model is pre-trained & fragile; this is fine-tuning |
| `mixed_warmup_steps` | 300 | gentle warmup so LR doesn't spike before projections re-align |
| `mixed_grad_clip` | 0.5 | tighter clip; variable-length segments produce rogue gradients |
| `switch_loss_weight` | 0.1 | let LM signal dominate early; switch head stabilises once LM is solid |

**Early stopping & best-checkpoint saving** (because the mixed phase is slow):
The loss is noisy at batch=1, so the monitor uses a **smoothed running average**
over the last `mixed_val_every=100` steps rather than a single step loss. Every
100 steps:
- if the smoothed loss is a new best, `model_final.pt` is **saved in-place**
  (so you always have the best model, even if training continues or
  early-stops later);
- if it hasn't improved for `mixed_early_stop_patience=10` consecutive checks,
  training halts with `early stopping at step N (no improvement for 10 checks,
  best=X.XXXX)`.

On a 4 GB GPU at ~0.1 it/s, this saves hours: an observed run hit its minimum
(~0.86) at step 1650 then slowly rose to 0.89 by step 3050 — early stopping
would halt near 1650 instead of running to 5000.

**Loop**: cycle the session loader; accumulate 16 batches, then unscale →
grad-clip → step → update LR via cosine schedule. Run until `mixed_steps_max=5000`
or early stopping fires.

**Checkpoint**: `model_final.pt` is kept updated to the best smoothed loss seen
so far. A final `save_checkpoint(tag="final")` after `train_mixed` saves the
last state, but the best is already preserved if it was better.

---

## 7. Datasets (`dataset.py`)

Three dataset classes support the three training phases:

### 7.1 `WindowDataset` (pre-training)

Samples fixed-length windows from a list of raw texts. Each window is
`max_seq_len` tokens; windows are drawn at random offsets until
`max_windows` is reached. Used by `pretrain_expert`.

### 7.2 Hand-off pairs (joint stitching)

`build_boundary_handoff_pairs` walks each session, merges consecutive same-kind
pieces into segments, and for every code↔text boundary emits a (prefix,
continuation) pair: the **tail** of the segment before the boundary (its last
`seq_len//2` tokens — the hand-off happens at its end, and `joint_loss` carries
the last hidden states) paired with the **head** of the segment after it, each
tokenized in its own expert's vocab. Both directions (A→B and B→A) come out
naturally from python→english and english→python boundaries. `BoundaryHandoffDataset`
pads these to `seq_len`.

The legacy `HandoffDataset` (random, unrelated A-text ↔ B-text pairing) is kept
only as a fallback for the offline smoke test, where no session data exists.

### 7.3 `MixedDataset` (mixed training)

Builds switch-token-annotated sessions from `synthetic_combined`. Each session
is segmented into `(expert_name, ids)` pieces with `<switch:NAME>` tokens
inserted at every code↔text boundary. The dataset yields one full session at a
time (collated verbatim, since segments are variable-length).

### 7.4 `load_raw_texts`

Reconstructs per-expert raw texts directly from `synthetic_raw` using the same
segmentation as the corpus extractor, so pre-training never depends on the
intermediate corpus shards (which `prepare` deletes after tokenizer training).

---

## 8. Configuration (`config.py`)

`config.py` is the **single source of truth** for all hyper-parameters. The
key dataclasses:

- **`ExpertConfig`**: per-expert transformer dimensions (`d_model`, `n_heads`,
  `n_layers`, `d_ff`, `max_seq_len`, `dropout`) and tokenizer size.
- **`SharedSpaceConfig`**: `dim=256` (the bottleneck), `bridge_len` (how many
  hidden states are carried across a hand-off; default 1), and the optional
  CALM cross-attention bridge knobs — `cross_attn` (master switch, default
  `False`), `cross_attn_n_heads` (heads in the cross-attn block, default 8),
  `cross_attn_dropout` (default 0.1), `cross_attn_residual` (default `True`).
  When `cross_attn=True` the carried states are consumed via cross-attention
  instead of being prepended as seed positions (see §4.4).
- **`Config.large()`**: a bigger preset (`d_model=768`, 8 layers, 12 heads,
  dim 384, `max_seq_len=1024` → ~63-67 M/expert) for 11 GB GPUs.
- **`PretrainOverride`**: per-expert overrides for pre-training knobs; any
  `None` field falls back to the global default.
- **`TrainConfig`**: all training hyper-parameters for all three phases, plus
  data/corpus sizes and misc (fp16, seed, log_every, num_workers).
- **`GenConfig`**: generation knobs (`max_new_tokens`, `temperature`,
  `top_k`, `max_switches`).
- **`Config.default()`**: builds the 2-expert default (python + english) and
  sets the per-expert pre-train overrides.

---

## 9. CLI (`main.py`)

| command | what it does |
|---|---|
| `extract` | generate synthetic sessions + extract split corpora |
| `prepare` | `extract` + train per-expert tokenizers |
| `pretrain` | load tokenizers + pre-train each expert (per-expert steps) |
| `joint` | load pre-trained weights + joint stitching + mixed end-to-end |
| `generate "prompt" --expert NAME` | generate with live switching |
| `eval` | held-out per-segment (python/english/overall) perplexity |
| `all` | prepare → pretrain → joint → sample |
| `status` | show which checkpoints exist |

Checkpoints are saved to `checkpoints/`:
`model_pretrained.pt`, `model_stitched.pt`, `model_final.pt`, plus the two
tokenizer JSON files.

---

## 10. End-to-end data flow

```
synthetic_data.write_synthetic(n=5000)
   └─► data/synthetic_raw/*.txt          (5000 event-schema sessions)

extract_synthetic_corpora()
   ├─► data/*_corpus_*.txt               (split code/text shards)
   └─► data/synthetic_combined/*.txt     (whole sessions + switch markers)

build_tokenizer() per expert
   └─► checkpoints/tokenizer_{python,english}.json

pretrain_expert() per expert (per-expert steps)
   └─► checkpoints/model_pretrained.pt

joint_finetune()  (projections only)
   └─► checkpoints/model_stitched.pt

train_mixed()     (all params, end-to-end)
   └─► checkpoints/model_final.pt

generate()        (live switching via shared space)

segment_perplexity() / monolith baseline (eval.py)
   └─► python/english/overall perplexity vs. a single-model control
```

---

## 10a. Evaluation (`eval.py`)

The core hypothesis is that routing between two small, specialised experts is
at least as good as a single monolithic model of comparable size on
interleaved code/prose. `eval.py` makes that claim testable:

- **`segment_perplexity(model, tokenizers, cfg)`** — runs held-out sessions
  through the cooperating model exactly as `mixed_loss` does (carried hidden
  states across switches, the unified `_encode_receiver` hand-off), but
  accumulates token NLL **separately for python and english segments** and
  excludes switch tokens. Returns `{python, english, overall}` perplexity
  (`exp(mean token NLL)` over non-pad, non-switch targets). Exposed as
  `python main.py eval`.
- **`Monolith` + `pretrain_monolith` + `monolith_perplexity`** — a single
  decoder-only transformer over **one shared tokenizer**, trained on the
  concatenated corpus and sized (via `d_model`/`n_layers`) to roughly the
  cooperating model's total parameter count. This is the apples-to-apples
  "no routing" control. Its held-out perplexity is directly comparable to the
  cooperating model's `overall` number because both are `exp(mean token NLL)`
  over the same kind of held-out split.

Comparing the two answers the question the earlier prototype could not: does
the shared-space routing actually help, or would a single model of the same
size do as well or better?

---

## 11. Design notes and lessons learned

- **Bottleneck shared space** (`dim=256 < d_model=512`): prevents the
  projections from collapsing to identity and forces a genuinely compact
  inter-expert representation.
- **Per-expert pre-training**: the two experts may converge at different rates,
  so `PretrainOverride` lets each stop at its own val minimum. On the current
  synthetic corpus both happen to converge around step 400; early stopping
  guards against over-fitting if the corpus or config changes.
- **Validation logging + early stopping**: a 10% held-out split with periodic
  val-loss logging is the quickest check for over-fitting. Early stopping
  (patience 3) halts training when val loss stops improving, so an expert
  never wastes steps memorising past its generalisation ceiling. On templated
  synthetic data the train loss can fall to ~0.006 while val loss rises from
  0.62 to 1.02 — early stopping catches this automatically.
- **Detached carried state in `mixed_loss`**: gradients don't flow back
  through the previous expert's transformer via `from_shared`. This bounds
  memory for long batch=1 sessions and matches generation semantics.
- **Switch-token down-weighting**: `switch_loss_weight=0.1` prevents the
  model from being equally rewarded for switching as for generating content,
  which suppresses the over-switching failure mode. The down-weight is applied
  **per-token** (only positions whose target is a switch token), not via a
  per-vocab-class cross-entropy weight; the latter also rescaled the overall
  loss magnitude and made the reported loss non-comparable across phases.
- **Mixed-phase stabilization**: the full 44.6 M model is pre-trained and
  fragile. A low LR (1e-5), long warmup (300 steps), tight grad-clip (0.5),
  and low switch weight (0.1) are all needed to avoid monotonic divergence.
- **Mixed-phase early stopping & best checkpoint**: the mixed phase is slow
  (~0.1 it/s on a 4 GB GPU). A smoothed-loss monitor saves the best model to
  `model_final.pt` whenever a new best appears, and early-stops after 10
  consecutive non-improvements — so you never waste hours past the loss
  minimum. The smoothing (over 100 steps) handles batch=1 noise.
- **Segment length cap after switch/eos append**: `MixedDataset` appends a
  switch or EOS token to each segment, which can push a `max_seq_len`-long
  piece to `max_seq_len + 1`. The cap is applied *after* the append to avoid
  an off-by-one logits/target shape mismatch in `mixed_loss`.
- **Template diversity**: 3 code templates per subject (different algorithmic
  approaches) raise the loss floor and force more transferable
  representations, instead of letting the model memorise one template per
  subject.

---

## 12. Revisions

This version reconciles the docs with the shipped config and addresses several
limitations of the original prototype:

- **Single default config.** The default is now the laptop (4 GB) config
  (`d_model=512`, 6 layers, dim 256 → ~44.6 M total) that all reported numbers
  refer to; the larger 768/8 preset moved to `Config.large()`. Earlier the
  shipped default and the documented results described different configs.
- **Real hand-off pairs.** The joint (stitching) phase now trains on
  code↔text boundary pairs taken from within the same session
  (`build_boundary_handoff_pairs`) instead of random, unrelated snippet pairs,
  so the receiving expert has a semantically related continuation to model.
- **Configurable bridge width.** `shared.bridge_len` (K) carries the last K
  hidden states across a hand-off; K=1 reproduces the original single-vector
  channel. This is the main lever for the (deliberately narrow) inter-expert
  channel.
- **Cheaper joint loss.** The alignment regularizer reuses hidden states already
  computed rather than running a second forward pass through expert B.
- **Robust validation.** Pre-training val loss is averaged over several batches
  (`pretrain_val_batches`) instead of a single noisy batch, so early stopping is
  more reliable.
- **Honest caveats.** Losses on the templated Option-A corpus mostly reflect
  memorisation; the hybrid Option-B pipeline is preferred for any real signal.
- **Rotary positions (RoPE).** Absolute position embeddings were replaced with
  RoPE, so position is relative. The hand-off seed positions no longer collide
  with a real token's absolute index, and long segments can never index a
  position table out of range. Residual output projections are additionally
  scaled by `1/sqrt(2*n_layers)` at init for depth stability.
- **Unified hand-off convention.** `joint_loss`, `mixed_loss`, and `generate`
  now share one convention (a leading `<pad>` hand-off query predicts the first
  destination token) for both bridge modes, so training matches inference. A
  smoke-test assertion checks the training and generation paths yield identical
  first-token logits.
- **Cross-expert alignment.** `joint_loss` adds a term pulling A's boundary
  shared code and B's continuation shared code together, so the space is
  genuinely shared rather than merely per-expert invertible.
- **No `<unk>` token.** Byte-level BPE has no OOV, so the unreachable `<unk>`
  token was removed (`NUM_SPECIAL_TOKENS_PER_EXPERT` 5 → 4).
- **Evaluation of the hypothesis.** `eval.py` adds held-out per-segment
  perplexity and a monolithic single-model baseline so routing can be measured
  against a no-routing control (`python main.py eval`).

> **A note on positional embeddings.** The model uses **rotary position
> embeddings (RoPE)**, applied to the query/key vectors inside attention, so
> there is no absolute position-embedding table. Position is encoded
> relatively, which means carried seed states can be prepended as virtual
> positions (or supplied as cross-attention memory) without colliding with a
> real token's absolute index, and no lookup can go out of range regardless of
> sequence length or bridge width `K`.
