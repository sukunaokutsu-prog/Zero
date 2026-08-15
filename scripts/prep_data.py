"""Build the BPE tokenizer + mmap token files for real and toy corpora.

Usage:  python scripts/prep_data.py [--vocab 2048]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from zero.tokenizer import BPETokenizer, save_tokenizer     # noqa: E402
from zero.streaming import tokenize_to_mmap                 # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--real", default="data/corpus.txt")
    ap.add_argument("--toy", default="data/toy.txt")
    args = ap.parse_args()

    for p in (args.real, args.toy):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    t0 = time.time()
    tok = BPETokenizer.build_streaming(args.real, args.vocab)
    save_tokenizer(tok, "data/mixed.tok.json")
    print(f"tokenizer vocab={tok.vocab_size} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    n = tokenize_to_mmap(args.real, tok, "data/mixed.ids.bin")
    print(f"real: {n:,} tokens ({time.time()-t0:.1f}s)")

    t0 = time.time()
    n = tokenize_to_mmap(args.toy, tok, "data/toy.ids.bin")
    print(f"toy:  {n:,} tokens ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
