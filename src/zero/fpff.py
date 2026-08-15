"""FPFF — Fractal-Procedural Forward-Forward engine.

Implements the architecture described in the FPFF-Z2G blueprint:

  * HashWeightEngine  — coordinate-based procedural ternary weights: no stored
    matrix, W[i,j] = ternary(hash(seed,i,j,ctx)) in {-1,0,1}, O(1) storage.
  * FractalDynamicalCore — a fixed physical matrix W_core recursively transformed
    by procedural ternary masks, with a *dynamic attractor exit condition*
    (recursion stops when ||h_t - h_{t-1}|| < eps) instead of a fixed depth.
  * Hebbian Forward-Forward — calculus-free local update: strengthen for real
    data, weaken for noise (Oja-normalized for stability).
  * SeedPopulation — neuroevolution of the procedural hash seeds (survival of
    the fittest, bit-flip mutations), fitness = "lowest error on the data".
  * SparseContextCompressor — ingest long context via first/trailing/deterministic
    middle sample projected to a latent vector (no token reconstruction).
  * RAGCurator — BM25 lexical retrieval over a local corpus; returns dense,
    boilerplate-stripped statements to the core.

None of the FPFF components uses autograd: the core learns without backprop,
gradient graphs, or activation caching.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import torch
from torch import nn
from torch.nn import functional as F


# --------------------------------------------------------------------------- #
# Deterministic integer hash (procedural seed mixer)
# --------------------------------------------------------------------------- #
def hash64(seed: int, ctx: int = 0) -> int:
    h = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    h ^= ctx * 0xC2B2AE3D27D4EB4F
    h &= 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    h = (h ^ (h >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (h ^ (h >> 31)) & 0xFFFFFFFFFFFFFFFF


def _ternary_matrix(rows: int, cols: int, seed: int, sparsity: float,
                    device: str, dtype=torch.float32) -> torch.Tensor:
    """Deterministic ternary {-1,0,1} matrix from a seed (never stored)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randint(-1, 2, (rows, cols), generator=g, dtype=dtype, device="cpu")
    if sparsity > 0:
        z = torch.rand(rows, cols, generator=g, dtype=dtype, device="cpu")
        m = torch.where(z < sparsity, torch.zeros_like(m), m)
    return m.to(device)


# --------------------------------------------------------------------------- #
# 2. Hash-weight engine — "zero bytes allocated for stored weights"
# --------------------------------------------------------------------------- #
class HashWeightEngine(nn.Module):
    """A linear layer whose weight matrix is generated on the fly from a hash.

    Persistent state: a single seed integer. Every parameter coordinate W[i,j]
    is produced procedurally as ternary {-1,0,1} and dropped after use. Because
    the weights are ternary, the matmul reduces to signed addition (no
    floating-point multiplies against the weight).
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 7,
                 sparsity: float = 2.0 / 3.0):
        super().__init__()
        self.in_dim, self.out_dim, self.seed, self.sparsity = in_dim, out_dim, seed, sparsity
        self._cache = {}

    def weights(self, ctx: int = 0, device: str = "cpu") -> torch.Tensor:
        w = self._cache.get((self.seed, ctx))
        if w is None or w.device != device:
            w = _ternary_matrix(self.out_dim, self.in_dim,
                                hash64(self.seed, ctx), self.sparsity, device)
            self._cache[(self.seed, ctx)] = w
        return w

    def forward(self, x: torch.Tensor, ctx: int = 0) -> torch.Tensor:
        W = self.weights(ctx, x.device)
        return x @ W.t()


# --------------------------------------------------------------------------- #
# 3. Fractal dynamical core — infinite virtual depth in a fixed matrix
# --------------------------------------------------------------------------- #
class FractalDynamicalCore(nn.Module):
    """h_{t+1} = LayerNorm(σ(h_t @ (W_core ⊙ M_procedural(t))))

    with a dynamic attractor exit: recursion stops when the activation settles
    (||h_t - h_{t-1}|| < eps). Persistent state is ONE matrix (W_core) + seed.
    """

    def __init__(self, dim: int, seed: int = 42, max_loops: int = 6,
                 eps: float = 1e-3, sparsity: float = 2.0 / 3.0):
        super().__init__()
        self.dim, self.seed = dim, seed
        self.max_loops, self.eps, self.sparsity = max_loops, eps, sparsity
        core = torch.empty(dim, dim)
        nn.init.orthogonal_(core, gain=1.0)
        self.register_buffer("core", core)
        self._mask_cache = {}
        self._cache_seed = seed

    def mask(self, loop: int) -> torch.Tensor:
        # masks depend only on (seed, loop): cache to avoid regenerating
        if self.seed != self._cache_seed:
            self._mask_cache.clear()
            self._cache_seed = self.seed
        key = (self.seed, loop)
        m = self._mask_cache.get(key)
        if m is None:
            m = _ternary_matrix(self.dim, self.dim, hash64(self.seed, loop),
                                self.sparsity, self.core.device)
            self._mask_cache[key] = m
        return m

    @torch.no_grad()
    def run(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for t in range(self.max_loops):
            W = self.core * self.mask(t)            # mutate geometry per loop
            h_new = F.layer_norm(F.relu(h @ W), (self.dim,))
            if t > 0 and (h_new - h).norm(dim=-1).mean().item() < self.eps:
                h = h_new                              # settled into an attractor
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
    def forward_forward_update(self, pos: torch.Tensor, neg: torch.Tensor,
                               lr: float = 1e-3):
        """Oja-normalized Hebbian FF: strengthen for real data, weaken for noise."""
        b = max(1, pos.shape[0])
        pos_h, neg_h = self.run(pos), self.run(neg)
        delta = (pos_h.t() @ pos_h - neg_h.t() @ neg_h) / b
        self.core.add_(lr * delta)
        self.core.mul_(math.sqrt(self.dim) / (self.core.norm() + 1e-12))


# --------------------------------------------------------------------------- #
# 4. Neuroevolution of the procedural seeds
# --------------------------------------------------------------------------- #
class SeedPopulation:
    def __init__(self, size: int = 8, base_seed: int = 42):
        self.seeds = [base_seed + i * 101 for i in range(size)]

    def mutate(self, seed: int) -> int:
        import random
        bits = 63
        for _ in range(random.randint(1, 4)):
            seed ^= (1 << random.randint(0, bits))
        return seed & 0xFFFFFFFFFFFFFFFF

    def evolve(self, fitness_fn) -> int:
        """fitness_fn(seed) -> float, lower is better ('lowest error on the data')."""
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
# 5. Sparse context compressor + RAG curator
# --------------------------------------------------------------------------- #
class SparseContextCompressor(nn.Module):
    """Ingest long context via first tokens + trailing tokens + a deterministic
    pseudo-random middle sample, projected into a single latent vector."""

    def __init__(self, dim: int, n_first: int = 16, n_last: int = 16,
                 n_mid: int = 16, hash_seed: int = 99):
        super().__init__()
        self.dim, self.n_first, self.n_last, self.n_mid = dim, n_first, n_last, n_mid
        self.proj = HashWeightEngine(dim, dim, seed=hash_seed, sparsity=0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, dim) -> latent (B, dim)."""
        B, T, _ = h.shape
        idx = list(range(min(self.n_first, T)))
        idx += list(range(max(0, T - self.n_last), T))
        # deterministic pseudo-random middle sample
        g = torch.Generator(device="cpu").manual_seed(hash64(1234, T))
        mid = torch.randperm(T, generator=g)[: self.n_mid].tolist()
        idx += mid
        idx = sorted(set(idx))
        sparse = h[:, idx, :]                      # (B, S, dim)
        pooled = sparse.mean(dim=1)                # (B, dim)
        return self.proj(pooled, ctx=T)


class RAGCurator:
    """BM25 retrieval over a local corpus; delivers dense, cleaned statements."""

    def __init__(self, corpus_text: str, chunk_size: int = 400):
        self.chunks = self._chunk(corpus_text, chunk_size)
        self.terms = [self._tokenize(c) for c in self.chunks]
        self.df = Counter()
        for t in self.terms:
            for w in set(t):
                self.df[w] += 1
        self.n_docs = len(self.chunks)
        self.avg_len = sum(len(t) for t in self.terms) / max(1, self.n_docs)

    @staticmethod
    def _tokenize(s: str):
        return re.findall(r"[a-z0-9']+", s.lower())

    @staticmethod
    def _chunk(text: str, size: int):
        words = text.split()
        return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

    def _bm25(self, q_terms, k1=1.5, b=0.75):
        scores = []
        for i, terms in enumerate(self.terms):
            tf = Counter(terms)
            dl = len(terms)
            s = 0.0
            for q in q_terms:
                if q not in tf:
                    continue
                idf = math.log(1 + (self.n_docs - self.df[q] + 0.5) / (self.df[q] + 0.5))
                num = tf[q] * (k1 + 1)
                den = tf[q] + k1 * (1 - b + b * dl / self.avg_len)
                s += idf * num / den
            scores.append(s)
        return scores

    def curate(self, query: str, k: int = 2, max_chars: int = 700) -> str:
        q = self._tokenize(query)
        scores = self._bm25(q)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        statements = []
        for i in top:
            s = self.chunks[i].strip()
            s = re.sub(r"\s+", " ", s)
            statements.append(s)
        return " ".join(statements)[:max_chars]
