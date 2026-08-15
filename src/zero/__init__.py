"""Zero — a from-scratch LLM training pipeline (FPFF-Z2G engine).

A complete, unified training stack for training a 50M-parameter language model
from scratch on CPU-only hardware, implementing the Fractal-Procedural
Forward-Forward (FPFF) + Zero-to-Growth (Z2G) architecture.

Modules:
  tokenizer   char / byte-level BPE
  streaming   disk-backed (memory-mapped) token corpora
  fpff        procedural hash weights, fractal attractor core, Hebbian + evolution,
              sparse context compressor, RAG curator
  model       GPT decoder with Z2G growth + FPFF hooks
  train       Z2G curriculum (Phase A/B/C) with live logging + generation
  eval        benchmark + generation harness
"""
from __future__ import annotations

import os

import torch

__version__ = "0.2.0"

torch.set_num_threads(int(os.environ.get("ZERO_THREADS", str(os.cpu_count() or 2))))
