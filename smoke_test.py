"""End-to-end smoke test with synthetic data.

Runs the full pipeline (tokenizer training -> pre-training -> joint
fine-tuning -> generation) on a tiny config with synthetic corpora so it
completes in about a minute on a laptop GPU and needs no network access.

    python smoke_test.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import torch

from config import Config, ExpertConfig, SharedSpaceConfig, TrainConfig
from cooperating import CooperatingExperts
from dataset import HandoffDataset, WindowDataset
from tokenizer import ExpertTokenizer, build_tokenizer
from train import joint_finetune, pretrain_expert, train_mixed


# ---------------------------------------------------------------------- #
# Synthetic corpora
# ---------------------------------------------------------------------- #
PYTHON_SNIPPETS: List[str] = [
    "def add(a, b): return a + b",
    "class Foo:\n    def __init__(self, x):\n        self.x = x",
    "for i in range(10):\n    print(i)",
    "import numpy as np\narr = np.zeros(5)",
    "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
    "x = [i * 2 for i in range(20)]",
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]",
    "result = sum(x ** 2 for x in range(100))",
    "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
    "with open('file.txt', 'w') as f:\n    f.write('hello')",
] * 50  # repeat to get enough data

ENGLISH_SNIPPETS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "In machine learning, a neural network is a model inspired by the brain.",
    "She went to the market to buy fresh vegetables for dinner.",
    "The history of computing spans many decades of rapid innovation.",
    "A transformer is a deep learning architecture based on attention.",
    "The cat sat on the mat while the rain fell outside the window.",
    "Language models predict the next token in a sequence of text.",
    "The mountain trail was steep and covered with fresh snow.",
    "Scientists have discovered a new species of frog in the rainforest.",
    "The novel tells the story of a young woman traveling across Europe.",
] * 50


def _tiny_config(cross_attn: bool = False) -> Config:
    cfg = Config()
    cfg.experts = {
        "python": ExpertConfig(name="python", vocab_size=2000, d_model=64,
                               n_heads=2, n_layers=2, d_ff=256, max_seq_len=64),
        "english": ExpertConfig(name="english", vocab_size=2000, d_model=64,
                                n_heads=2, n_layers=2, d_ff=256, max_seq_len=64),
    }
    # bridge_len=2 exercises the multi-vector hand-off path (K>1).
    # cross_attn=True exercises the CALM-style cross-attention bridge.
    cfg.shared = SharedSpaceConfig(
        dim=64, bridge_len=2,
        cross_attn=cross_attn, cross_attn_n_heads=2,
    )
    cfg.train = TrainConfig(
        pretrain_batch_size=16, pretrain_lr=3e-4,
        pretrain_steps_max=60,
        joint_epochs=1, joint_batch_size=8, joint_lr=1e-3,
        joint_steps_max=40,
        fp16=True, seed=42, log_every=20,
    )
    return cfg


def _write_corpus(tmp: Path, name: str, texts: List[str]) -> List[Path]:
    path = tmp / f"{name}_corpus_0.txt"
    with open(path, "w", encoding="utf-8") as fh:
        for t in texts:
            fh.write(" ".join(t.split()) + "\n")
    return [path]


def _run_pipeline(cfg: Config, label: str) -> None:
    """Run the full pipeline once for a given config."""
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    names = list(cfg.experts.keys())

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 1. Write synthetic corpora and train tokenizers.
        print("\n[1/5] Training tokenizers on synthetic data...")
        tokenizers: Dict[str, ExpertTokenizer] = {}
        for name in names:
            files = _write_corpus(tmp, name,
                                  PYTHON_SNIPPETS if name == "python" else ENGLISH_SNIPPETS)
            tok = build_tokenizer(cfg.experts[name], files, names)
            tokenizers[name] = tok
            print(f"  {name}: vocab_size={tok.vocab_size}")

        # 2. Build model.
        print("\n[2/5] Building model...")
        model = CooperatingExperts(cfg, tokenizers)
        model.to(device)
        if cfg.shared.cross_attn:
            exp = model.expert(names[1])
            assert exp.cross_attn.residual == cfg.shared.cross_attn_residual
            ids = torch.tensor([[tokenizers[names[1]].pad_id]], device=device)
            mem0 = torch.zeros(1, cfg.shared.bridge_len, exp.cfg.d_model, device=device)
            mem1 = torch.randn_like(mem0)
            exp.eval()
            out0 = exp.encode_with_cross_attn(ids, mem0)
            out1 = exp.encode_with_cross_attn(ids, mem1)
            assert not torch.allclose(out0, out1), "cross-attention ignores memory"
            out1.sum().backward()
            assert all(p.grad is not None for p in exp.cross_attn.parameters()), \
                "cross-attention parameter missing gradient"
            model.zero_grad(set_to_none=True)
        for n in names:
            print(f"  {n}: {model.expert(n).num_params()/1e6:.2f}M params")

        # 3. Pre-train each expert.
        print("\n[3/5] Pre-training experts...")
        for name in names:
            texts = PYTHON_SNIPPETS if name == "python" else ENGLISH_SNIPPETS
            pretrain_expert(model, name, texts, tokenizers[name], cfg)

        # 4. Joint fine-tune projections.
        print("\n[4/5] Joint fine-tuning projections...")
        texts = {
            "python": PYTHON_SNIPPETS,
            "english": ENGLISH_SNIPPETS,
        }
        joint_finetune(model, tokenizers, cfg, texts=texts)

        # 4b. Interleaved mixed_loss on a synthetic switch-token session.
        print("\n[4b] Interleaved mixed_loss (synthetic switch session)...")
        # Build one interleaved example: python code -> switch:english ->
        # english text -> switch:python -> python code.
        py_ids = tokenizers["python"].tokenizer.encode(
            "def add(a, b): return a + b").ids
        en_ids = tokenizers["english"].tokenizer.encode(
            "The function adds two numbers together.").ids
        sw_en = tokenizers["python"].switch_id("english")
        sw_py = tokenizers["english"].switch_id("python")
        eos_py = tokenizers["python"].eos_id
        eos_en = tokenizers["english"].eos_id
        seg_py1 = torch.tensor([py_ids + [sw_en]], dtype=torch.long, device=device)
        seg_en = torch.tensor([en_ids + [sw_py]], dtype=torch.long, device=device)
        seg_py2 = torch.tensor([py_ids + [eos_py]], dtype=torch.long, device=device)
        segments = [
            ("python", seg_py1),
            ("english", seg_en),
            ("python", seg_py2),
        ]
        loss, info = model.mixed_loss(segments)
        assert torch.isfinite(loss), "mixed_loss produced non-finite value"
        print(f"  mixed_loss = {loss.item():.4f} over {info['n_segs']} segments")

        # 5. Generate with switching.
        print("\n[5/5] Generation samples...")
        model.eval()
        for prompt, expert in [("def square", "python"), ("The model", "english")]:
            print(f"\n  prompt={prompt!r}  start={expert}")
            out = model.generate(prompt, start_expert=expert, max_new_tokens=40,
                                 temperature=0.7, top_k=20)
            print("  " + out.replace("\n", "\n  "))


def main() -> None:
    # Run the pipeline twice: once with the legacy seed-prepend bridge, once
    # with the CALM-style cross-attention bridge, to exercise both code paths.
    _run_pipeline(_tiny_config(cross_attn=False), "LEGACY seed-prepend bridge")
    _run_pipeline(_tiny_config(cross_attn=True),  "CALM cross-attention bridge")
    print("\n[OK] Smoke test passed (both bridge modes)!")


if __name__ == "__main__":
    main()
