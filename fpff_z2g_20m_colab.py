#!/usr/bin/env python3
"""
================================================================================
 FPFF-Z2G — train a ~20M-parameter language model from scratch (single file)
================================================================================
 A complete, self-contained implementation of the Fractal-Procedural
 Forward-Forward + Zero-to-Growth (FPFF-Z2G) method, tuned for a **T4 GPU**
 (Colab: Runtime -> Change runtime type -> T4 GPU).

 WHAT IT DOES
   * Z2G curriculum:
       Phase A — train a SHALLOW model (3 layers, ~10M params) on a toy corpus
                 (TinyShakespeare) to embed base syntax/geometry.
       Phase B — function-preserving depth doubling (3 -> 6 layers, ~20M params)
                 with zero loss-spike, then an instant switch to a REAL corpus
                 (20+ public-domain classic novels).
   * FPFF (Fractal-Procedural Forward-Forward):
       - HashWeightEngine   : procedural ternary weights W[i,j] in {-1,0,1}
                              generated on the fly from a hash (no stored matrix).
       - FractalDynamicalCore: ONE fixed 512x512 matrix, recursively re-masked
                              each loop, with a dynamic "attractor" exit
                              (||h_t - h_{t-1}|| < eps).
       - Hebbian Forward-Forward: calculus-free local update (positive pass
                              strengthens, negative pass weakens; Oja-normalized).
       - Seed neuroevolution : seeds scored by validation loss ("lowest error
                              on the data"), worst half deleted, fittest mutated.
   * A GPT decoder backbone is the generative engine (backprop + AdamW).
   * fp16 autocast + GradScaler for T4 tensor cores.

 LOGS
   * Live per-step logs: loss, lr, tok/s, tokens seen, % of target, ETA, phase.
   * Evaluation (train/val loss + perplexity) every `eval_interval` steps.
   * **Text generation printed + logged every 100 steps.**

 USAGE (Colab, one cell):
   !pip install -q tokenizers        # only if not already installed
   %run fpff_z2g_20m_colab.py        # or paste the whole file into a cell

 OPTIONS
   --steps N      total steps (default 3000)
   --smoke        tiny 40-step CPU run to verify the script works end-to-end
   --device cuda|cpu
================================================================================
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Tuple

# --------------------------------------------------------------------------- #
# 0. Dependencies (Colab has torch; `tokenizers` gives a fast byte-level BPE)
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_tokenizers():
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers"])
        import tokenizers  # noqa: F401


# --------------------------------------------------------------------------- #
# 1. Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # model geometry (final ~20M params at 6 layers / 512 dim)
    n_layer: int = 6
    n_layer_start: int = 3          # Phase A depth
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 256
    dropout: float = 0.0
    bias: bool = False
    vocab_size: int = 2048

    # Z2G curriculum
    phase_b_step: int = 300         # step where depth doubles + data switches

    # FPFF
    fpff: bool = True
    fpff_loops: int = 4
    fpff_eps: float = 1e-3
    fpff_sparsity: float = 2.0 / 3.0
    fpff_seed: int = 42
    fpff_scale: float = 0.1
    fpff_lr: float = 1e-3
    fpff_pop: int = 8
    fpff_hebb_interval: int = 10
    fpff_evo_interval: int = 200
    hash_engine: bool = True
    hash_seed: int = 7
    hash_scale: float = 0.05

    # optimizer
    lr: float = 1e-3
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 100
    min_lr: float = 0.0
    grad_clip: float = 1.0

    # loop
    max_steps: int = 3000
    batch_size: int = 64
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    gen_interval: int = 100          # <-- generate text every 100 steps
    gen_max_tokens: int = 160
    gen_prompt: str = "It was a dark and stormy night"
    seed: int = 1337
    target_tokens: int = 1_000_000_000

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "/content/zero_ckpt"


# --------------------------------------------------------------------------- #
# 2. FPFF — procedural hash weights + fractal attractor core + evolution
# --------------------------------------------------------------------------- #
def hash64(seed: int, ctx: int = 0) -> int:
    h = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h ^= ctx * 0xC2B2AE3D27D4EB4F
    h &= 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (h ^ (h >> 31)) & 0xFFFFFFFFFFFFFFFF


def ternary_matrix(rows: int, cols: int, seed: int, sparsity: float,
                   device: str, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randint(-1, 2, (rows, cols), generator=g, dtype=dtype, device="cpu")
    if sparsity > 0:
        z = torch.rand(rows, cols, generator=g, dtype=dtype, device="cpu")
        m = torch.where(z < sparsity, torch.zeros_like(m), m)
    return m.to(device)


class HashWeightEngine(nn.Module):
    """A linear layer whose weight matrix is generated procedurally (O(1) storage)."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 7, sparsity: float = 2 / 3):
        super().__init__()
        self.in_dim, self.out_dim, self.seed, self.sparsity = in_dim, out_dim, seed, sparsity
        self._cache = {}

    def weights(self, ctx: int, device: str, dtype) -> torch.Tensor:
        key = (self.seed, ctx, device)
        w = self._cache.get(key)
        if w is None or w.dtype != dtype:
            w = ternary_matrix(self.out_dim, self.in_dim, hash64(self.seed, ctx),
                               self.sparsity, device, dtype)
            self._cache[key] = w
        return w

    def forward(self, x: torch.Tensor, ctx: int = 0) -> torch.Tensor:
        W = self.weights(ctx, x.device, x.dtype)
        return x @ W.t()


class FractalDynamicalCore(nn.Module):
    """h_{t+1} = LayerNorm(relu(h_t @ (W_core * mask(t)))) with attractor exit."""

    def __init__(self, dim: int, seed: int = 42, max_loops: int = 4,
                 eps: float = 1e-3, sparsity: float = 2 / 3):
        super().__init__()
        self.dim, self.seed = dim, seed
        self.max_loops, self.eps, self.sparsity = max_loops, eps, sparsity
        core = torch.empty(dim, dim)
        nn.init.orthogonal_(core, gain=1.0)
        self.register_buffer("core", core)
        self._mask_cache = {}
        self._cache_seed = seed

    def mask(self, loop: int) -> torch.Tensor:
        if self.seed != self._cache_seed:
            self._mask_cache.clear()
            self._cache_seed = self.seed
        key = (self.seed, loop)
        m = self._mask_cache.get(key)
        if m is None:
            m = ternary_matrix(self.dim, self.dim, hash64(self.seed, loop),
                               self.sparsity, self.core.device)
            self._mask_cache[key] = m
        return m

    @torch.no_grad()
    def run(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for t in range(self.max_loops):
            W = (self.core * self.mask(t)).to(h.dtype)
            h_new = F.layer_norm(F.relu(h @ W), (self.dim,))
            if t > 0 and (h_new - h).norm(dim=-1).mean().item() < self.eps:
                h = h_new          # settled into an attractor — break early
                break
            h = h_new
        return h

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(x)

    @torch.no_grad()
    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        return self.run(x).square().mean()

    @torch.no_grad()
    def forward_forward_update(self, pos: torch.Tensor, neg: torch.Tensor, lr: float):
        """Oja-normalized Hebbian FF: strengthen for real, weaken for noise."""
        b = max(1, pos.shape[0])
        pos_h = self.run(pos.half()).float()
        neg_h = self.run(neg.half()).float()
        delta = (pos_h.t() @ pos_h - neg_h.t() @ neg_h) / b
        self.core.add_(lr * delta)
        self.core.mul_(math.sqrt(self.dim) / (self.core.norm() + 1e-12))


class SeedPopulation:
    """Neuroevolution of procedural seeds (survival of the fittest)."""

    def __init__(self, size: int = 8, base_seed: int = 42):
        self.seeds = [base_seed + i * 101 for i in range(size)]

    def mutate(self, seed: int) -> int:
        import random
        for _ in range(random.randint(1, 4)):
            seed ^= (1 << random.randint(0, 63))
        return seed & 0xFFFFFFFFFFFFFFFF

    def evolve(self, fitness_fn) -> int:
        keep = max(1, len(self.seeds) // 2)
        scored = sorted((fitness_fn(s), s) for s in self.seeds)
        best = [s for _, s in scored[:keep]]
        new = list(best)
        while len(new) < len(self.seeds):
            parent = best[len(new) % len(best)]
            new.append(self.mutate(parent))
        self.seeds = new
        return best[0]


# --------------------------------------------------------------------------- #
# 3. GPT decoder backbone (with Z2G growth + FPFF integration)
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.has_flash = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self.has_flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1),
                                  float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer_start)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight   # weight tying

        self.fpff = None
        self.fpff_pop = None
        if cfg.fpff:
            self.fpff = FractalDynamicalCore(cfg.n_embd, seed=cfg.fpff_seed,
                                             max_loops=cfg.fpff_loops,
                                             eps=cfg.fpff_eps,
                                             sparsity=cfg.fpff_sparsity)
            self.fpff_pop = SeedPopulation(size=cfg.fpff_pop, base_seed=cfg.fpff_seed)
        self.hash_engine = None
        if cfg.hash_engine:
            self.hash_engine = HashWeightEngine(cfg.n_embd, cfg.n_embd, seed=cfg.hash_seed)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- Z2G: function-preserving depth growth ----
    def double_depth(self):
        new_blocks = []
        for blk in self.transformer.h:
            new_blocks.append(blk)
            clone = copy_block(blk)
            zero_outputs(clone)          # copied block starts as identity
            new_blocks.append(clone)
        self.transformer.h = nn.ModuleList(new_blocks)
        self.cfg.n_layer = len(new_blocks)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok = self.transformer.wte(idx)                 # fp32 (embeddings)
        x = tok + self.transformer.wpe(pos)
        if self.fpff is not None:
            x = x + self.cfg.fpff_scale * self.fpff(tok.detach().half()).float()
        if self.hash_engine is not None:
            x = x + self.cfg.hash_scale * self.hash_engine(tok.detach().half()).float()
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx

    # ---- FPFF hooks ----
    def fpff_hebbian(self, pos, neg):
        if self.fpff is not None:
            self.fpff.forward_forward_update(pos, neg, lr=self.cfg.fpff_lr)

    def fpff_evolve(self, fitness_fn) -> int:
        if self.fpff is not None and self.fpff_pop is not None:
            return self.fpff_pop.evolve(fitness_fn)
        return self.cfg.fpff_seed

    def fpff_seed(self) -> int:
        return self.fpff.seed if self.fpff is not None else self.cfg.fpff_seed

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def copy_block(blk):
    import copy
    return copy.deepcopy(blk)


def zero_outputs(block):
    for name, m in block.named_modules():
        if isinstance(m, nn.Linear) and name.endswith("c_proj"):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# --------------------------------------------------------------------------- #
# 4. Data — real public-domain corpus + toy curriculum corpus
# --------------------------------------------------------------------------- #
GUTENBERG_BOOKS = [
    (84, "Frankenstein"), (1342, "Pride and Prejudice"), (11, "Alice in Wonderland"),
    (2701, "Moby Dick"), (1661, "Sherlock Holmes"), (98, "A Tale of Two Cities"),
    (345, "Dracula"), (1260, "Jane Eyre"), (768, "Wuthering Heights"),
    (174, "Dorian Gray"), (1400, "Great Expectations"), (730, "Oliver Twist"),
    (35, "The Time Machine"), (36, "War of the Worlds"), (2554, "Crime and Punishment"),
    (1184, "Count of Monte Cristo"), (1399, "Anna Karenina"), (2600, "War and Peace"),
    (76, "Huckleberry Finn"), (74, "Tom Sawyer"), (5200, "Metamorphosis"),
    (1257, "Three Musketeers"), (215, "Call of the Wild"), (910, "White Fang"),
]

TOY_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

FALLBACK_TEXT = (
    "CHAPTER I. Call me Ishmael. Some years ago, never mind how long precisely, "
    "having little or no money in my purse, and nothing particular to interest me "
    "on shore, I thought I would sail about a little and see the watery part of the "
    "world. It is a way I have of driving off the spleen, and regulating the "
    "circulation. Whenever I find myself growing grim about the mouth; whenever it "
    "is a damp, drizzly November in my soul; then I account it high time to get to "
    "sea as soon as I can. "
)


def fetch(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def strip_book(text: str) -> str:
    s, e = text.find("*** START OF"), text.find("*** END OF")
    return text[s:e].strip() if (s != -1 and e != -1 and e > s) else text.strip()


def download_books(device="cpu", quiet=False):
    parts, ok, fail = [], [], []
    for pid, name in GUTENBERG_BOOKS:
        for url in (f"https://www.gutenberg.org/cache/epub/{pid}/pg{pid}.txt",
                    f"https://www.gutenberg.org/files/{pid}/{pid}-0.txt"):
            try:
                txt = fetch(url)
                txt = strip_book(txt)
                if len(txt) > 50_000:
                    parts.append(txt)
                    ok.append(name)
                    break
            except Exception:
                continue
        else:
            fail.append(name)
        if not quiet:
            print(f"  [data] {name:<24s} {'OK' if name in ok else 'MISS'} "
                  f"({len(parts[-1]) if parts else 0:,} chars)", flush=True)
    real_text = "\n\n".join(parts)
    return real_text, ok, fail


def download_toy():
    try:
        return fetch(TOY_URL)
    except Exception:
        return FALLBACK_TEXT


# --------------------------------------------------------------------------- #
# 5. Tokenizer (byte-level BPE via `tokenizers`, char fallback)
# --------------------------------------------------------------------------- #
class CharTok:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def build_tokenizer(real_text: str, vocab_size: int):
    try:
        ensure_tokenizers()
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[],
                                      min_frequency=1)

        def chunks():
            for i in range(0, len(real_text), 1_000_000):
                yield real_text[i:i + 1_000_000]

        tok.train_from_iterator(chunks(), trainer=trainer)
        return tok, "bpe"
    except Exception:
        return CharTok(real_text[:2_000_000]), "char"


def encode_all(tok, text: str) -> torch.Tensor:
    if hasattr(tok, "encode"):  # both have encode; tokenizers returns .ids
        ids = tok.encode(text)
        if hasattr(ids, "ids"):
            ids = ids.ids
        return torch.tensor(ids, dtype=torch.long)
    return torch.tensor(tok.encode(text), dtype=torch.long)


def encode_text(tok, s: str) -> List[int]:
    ids = tok.encode(s)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return list(ids)


def decode(tok, ids) -> str:
    return tok.decode(ids)


def vocab_size(tok) -> int:
    if hasattr(tok, "get_vocab_size"):
        return tok.get_vocab_size()
    return tok.vocab_size


# --------------------------------------------------------------------------- #
# 6. Training loop
# --------------------------------------------------------------------------- #
def get_batch(data: torch.Tensor, block: int, batch: int, device: str):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(device), y.to(device)


def pos_neg_embeddings(model, ids):
    B, T = ids.shape
    pos = model.transformer.wte(ids).detach().view(B * T, -1)
    neg_ids = torch.randint(0, model.cfg.vocab_size, (B, T), device=ids.device)
    neg = model.transformer.wte(neg_ids).detach().view(B * T, -1)
    return pos, neg


def cosine_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    d = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * d)) * (cfg.lr - cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, amp_ctx):
    model.eval()
    losses = {}
    for split, data in (("train", train_data), ("val", val_data)):
        ls = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device)
            with amp_ctx():
                _, loss = model(x, y)
            ls[k] = loss.item()
        losses[split] = ls.mean().item()
    model.train()
    return losses


def train(cfg: Config, download: bool = True):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    use_amp = device == "cuda"
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if use_amp else (lambda: _null())
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if use_amp:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- data ----
    print("=" * 78)
    print("FPFF-Z2G — gathering real training data")
    print("=" * 78)
    if download:
        real_text, ok, fail = download_books(device, quiet=False)
        if len(real_text) < 200_000:
            print("[data] WARNING: download failed, using bundled fallback text")
            real_text = FALLBACK_TEXT * 2000
        toy_text = download_toy()
    else:
        real_text = FALLBACK_TEXT * 200
        toy_text = FALLBACK_TEXT * 50
    print(f"[data] real corpus: {len(real_text):,} chars; toy: {len(toy_text):,} chars")

    tok, kind = build_tokenizer(real_text, cfg.vocab_size)
    cfg.vocab_size = vocab_size(tok)
    real_ids = encode_all(tok, real_text)
    toy_ids = encode_all(tok, toy_text)
    n_val = int(0.05 * len(real_ids))
    train_ids, val_ids = real_ids[:-n_val], real_ids[-n_val:]
    print(f"[data] tokenizer={kind} vocab={cfg.vocab_size} | "
          f"real tokens={len(real_ids):,} | toy tokens={len(toy_ids):,}")

    # ---- model ----
    model = GPT(cfg).to(device)
    print(f"[model] Phase A: {cfg.n_layer_start} layers | Phase B: {cfg.n_layer} layers")
    print(f"[model] final param count: {model.n_params():,}")

    def make_optimizer():
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay, betas=cfg.betas)
    optimizer = make_optimizer()

    os.makedirs(cfg.out_dir, exist_ok=True)
    best_val = float("inf")
    tokens_per_step = cfg.batch_size * cfg.block_size
    total_tokens = 0
    t0 = time.time()

    def log(msg):
        print(msg, flush=True)

    log("=" * 78)
    log(f"[run] steps={cfg.max_steps} | batch={cfg.batch_size} | block={cfg.block_size} "
        f"| device={device} | amp={use_amp}")
    log(f"[run] Phase A (toy) -> step {cfg.phase_b_step}; Phase B (real, "
        f"{cfg.n_layer} layers) -> step {cfg.max_steps}")
    log("=" * 78)

    for step in range(1, cfg.max_steps + 1):
        # ---- Phase B: function-preserving growth + data switch ----
        if step == cfg.phase_b_step:
            model.double_depth()
            optimizer = make_optimizer()
            log(f"[z2g] Phase B: depth {cfg.n_layer_start} -> {len(model.transformer.h)} "
                f"| params {model.n_params():,} | data switch toy -> real")

        dataset = toy_ids if step < cfg.phase_b_step else train_ids
        lr = cosine_lr(step - 1, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(dataset, cfg.block_size, cfg.batch_size, device)
        with amp_ctx():
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += tokens_per_step

        # ---- FPFF Hebbian positive/negative pass ----
        if cfg.fpff and step % cfg.fpff_hebb_interval == 0:
            pos, neg = pos_neg_embeddings(model, x)
            model.fpff_hebbian(pos, neg)

        # ---- live log ----
        if step % cfg.log_interval == 0 or step == 1:
            elapsed = time.time() - t0
            tok_s = total_tokens / max(elapsed, 1e-9)
            pct = 100.0 * total_tokens / cfg.target_tokens
            eta_h = (cfg.target_tokens - total_tokens) / max(tok_s, 1e-9) / 3600
            log(f"[step {step:>5}/{cfg.max_steps}] loss {loss.item():.4f} | "
                f"lr {lr:.2e} | {tok_s:,.0f} tok/s | tok {total_tokens:,} "
                f"({pct:.4f}%) | ETA {eta_h:.1f}h | phase "
                f"{'A' if step < cfg.phase_b_step else 'B'} | t {elapsed:.0f}s")

        # ---- evaluation ----
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, train_ids, val_ids, cfg, amp_ctx)
            val = losses["val"]
            log(f"[eval  step {step:>5}] train {losses['train']:.4f} | "
                f"val {val:.4f} | ppl {math.exp(val):.2f}")
            if val < best_val:
                best_val = val
                torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                            "step": step, "val": val}, os.path.join(cfg.out_dir, "best.pt"))
                log(f"[ckpt] best saved (val {val:.4f})")

        # ---- FPFF neuroevolution ----
        if cfg.fpff and step % cfg.fpff_evo_interval == 0:
            @torch.no_grad()
            def fitness(seed):
                model.fpff.seed = seed
                tot = 0.0
                for _ in range(2):
                    vx, vy = get_batch(val_ids, cfg.block_size, cfg.batch_size, device)
                    with amp_ctx():
                        _, l = model(vx, vy)
                    tot += l.item()
                return tot / 2.0
            best_seed = model.fpff_evolve(fitness)
            model.fpff.seed = best_seed
            log(f"[fpff step {step:>5}] neuroevolution -> deployed seed {best_seed}")

        # ---- text generation every 100 steps ----
        if step % cfg.gen_interval == 0 or step == 1:
            prompt_ids = torch.tensor([encode_text(tok, cfg.gen_prompt)],
                                      dtype=torch.long, device=device)
            model.eval()
            with amp_ctx(), torch.no_grad():
                out = model.generate(prompt_ids, cfg.gen_max_tokens, temperature=0.8, top_k=50)
            model.train()
            sample = decode(tok, out[0].tolist())
            log("\n" + "-" * 78)
            log(f"[generation step {step}]")
            log("-" * 78)
            log(sample)
            log("-" * 78 + "\n")

    # ---- finish ----
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                "step": cfg.max_steps, "val": best_val},
               os.path.join(cfg.out_dir, "final.pt"))
    elapsed = time.time() - t0
    log("=" * 78)
    log(f"[done] final val {best_val:.4f} | ppl {math.exp(best_val):.2f} | "
        f"{total_tokens:,} tokens | {elapsed:.0f}s | "
        f"{total_tokens / max(elapsed, 1e-9):,.0f} tok/s")
    log(f"[done] checkpoints in: {cfg.out_dir}")
    log("=" * 78)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- #
# 7. Entry point
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny 40-step CPU run to verify the script works")
    args = ap.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    if args.smoke:
        cfg.n_embd, cfg.n_head, cfg.n_layer, cfg.n_layer_start = 128, 4, 4, 2
        cfg.block_size, cfg.batch_size = 64, 8
        cfg.max_steps, cfg.phase_b_step = 40, 20
        cfg.eval_interval = cfg.gen_interval = 20
        cfg.log_interval = 5
        cfg.fpff_loops = 3
        cfg.fpff_evo_interval = 10
        cfg.gen_max_tokens = 40
        cfg.device = "cpu"
        cfg.out_dir = "/tmp/zero_smoke"
    else:
        cfg.max_steps = args.steps

    train(cfg, download=not args.smoke)


if __name__ == "__main__":
    main()
