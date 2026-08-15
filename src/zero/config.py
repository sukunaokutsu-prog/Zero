"""Configuration for Zero training runs (FPFF-Z2G pipeline)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ---- data ----
    data_path: str = "data/corpus.txt"        # Phase B/C: real corpus (raw text)
    toy_path: str = "data/toy.txt"            # Phase A: toy/curriculum corpus
    tokenizer: str = "bpe"                    # "char" | "bpe"
    tokenizer_path: str = ""                  # saved BPE tokenizer JSON
    ids_path: str = ""                        # mmap token ids for real corpus
    toy_ids_path: str = ""                    # mmap token ids for toy corpus
    block_size: int = 256
    split_ratio: float = 0.9

    # ---- model ----
    n_layer: int = 8                          # Phase A depth (doubled in Phase B)
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False

    # ---- Z2G curriculum ----
    growth: str = "net2net"                   # "net2net" | "none"
    phase_b_step: int = 60                    # step where depth doubles + data switches
    phase_c_spawn: int = -1                   # step for modular spawn (-1 = off)

    # ---- FPFF: Fractal-Procedural Forward-Forward ----
    fpff: bool = True
    fpff_loops: int = 6                       # max fractal recursion depth
    fpff_eps: float = 1e-3                    # attractor convergence threshold
    fpff_sparsity: float = 2.0 / 3.0          # fraction of zeros in ternary masks
    fpff_seed: int = 42
    fpff_scale: float = 0.1                   # weight of FPFF features in residual stream
    fpff_lr: float = 1e-3                     # Hebbian local learning rate
    fpff_pop: int = 8                         # neuroevolution population
    fpff_hebb_interval: int = 10
    fpff_evo_interval: int = 50

    # ---- Hash-weight engine (procedural ternary weights, O(1) storage) ----
    hash_engine: bool = True
    hash_seed: int = 7
    hash_scale: float = 0.05

    # ---- optimizer ----
    lr: float = 3e-3
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 40
    min_lr: float = 0.0
    grad_clip: float = 1.0

    # ---- loop ----
    max_steps: int = 200
    batch_size: int = 32
    eval_interval: int = 25
    eval_iters: int = 20
    log_interval: int = 5
    gen_interval: int = 50
    gen_max_tokens: int = 200
    gen_prompt: str = "CHAPTER I."
    seed: int = 1337

    # ---- long-run / resume / progress ----
    target_tokens: int = 1_000_000_000   # progress target ("1 billion tokens")
    resume_from: str = ""                # path to a resume checkpoint (or "")
    checkpoint_interval: int = 100       # steps between resume checkpoints

    # ---- runtime ----
    name: str = "run"
    out_dir: str = "results"
    device: str = "cpu"
    vocab_size: int = 0
