"""Live demo: load the trained model, generate samples, and run the RAG curator.

Usage:  python scripts/demo.py --ckpt results/fifty-m/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from zero.eval import generate, load_checkpoint  # noqa: E402
from zero.fpff import RAGCurator, SparseContextCompressor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--max-tokens", type=int, default=180)
    args = ap.parse_args()

    model, cfg, tokenizer, _ = load_checkpoint(args.ckpt)
    model.eval()
    print(f"loaded {cfg.name}: {model.n_params():,} params, {cfg.n_layer} layers, "
          f"fpff={'on' if cfg.fpff else 'off'} hash={'on' if cfg.hash_engine else 'off'}")

    print("\n" + "=" * 70)
    print("FREE GENERATION")
    print("=" * 70)
    for prompt, temp in [("CHAPTER I.", 0.8), ("The theory of", 0.8),
                         ("It was a dark and stormy", 0.7), ("= Science and", 0.8)]:
        print(f"\n>>> {prompt!r}  (T={temp})")
        print(generate(args.ckpt, prompt, args.max_tokens, temp))

    print("\n" + "=" * 70)
    print("RAG CURATOR (BM25 retrieval over the training corpus)")
    print("=" * 70)
    text = open(args.corpus, encoding="utf-8", errors="replace").read()
    curator = RAGCurator(text)
    for query in ["time machine travel future", "count of monte cristo revenge",
                  "whale captain ahab sea"]:
        ctx = curator.curate(query)
        prompt = f"[context] {ctx}\n\nQuestion: {query}\n\nAnswer:"
        print(f"\n>>> query: {query!r}")
        print(f"    retrieved: {ctx[:160]}...")
        print(f"    model: {generate(args.ckpt, prompt, args.max_tokens, 0.8)[len(prompt):][:400]}")

    # ---- sparse context compressor (session-vector demo) ----
    print("\n" + "=" * 70)
    print("SPARSE CONTEXT COMPRESSOR (long-context -> latent)")
    print("=" * 70)
    comp = SparseContextCompressor(cfg.n_embd)
    long_prompt = ("CHAPTER I. The count of monte cristo arrived at the harbour of "
                   "Marseilles " * 6)
    ids = tokenizer.encode(long_prompt)
    x = model.transformer.wte(torch.tensor([ids], dtype=torch.long))
    latent = comp(x)
    print(f"  context length {len(ids)} tokens -> latent vector shape {tuple(latent.shape)} "
          f"(norm {latent.norm().item():.2f})")


if __name__ == "__main__":
    main()
