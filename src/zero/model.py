"""GPT decoder with Z2G growth and FPFF integration.

The backbone is a standard pre-LayerNorm GPT decoder (backprop-trained). The
FPFF components — the fractal dynamical core and the procedural hash-weight
engine — enrich the token embeddings *without any gradient flowing through
them* (calculus-free), implementing the "reasoning core + generative engine"
division of labour from the FPFF-Z2G blueprint.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .fpff import FractalDynamicalCore, HashWeightEngine, SeedPopulation


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
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1), float("-inf"))
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
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

        # ---- FPFF components (calculus-free) ----
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
            self.hash_engine = HashWeightEngine(cfg.n_embd, cfg.n_embd,
                                                seed=cfg.hash_seed)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---- Z2G: function-preserving depth growth ----
    def double_depth(self):
        new_blocks = []
        for blk in self.transformer.h:
            new_blocks.append(blk)
            clone = copy.deepcopy(blk)
            self._zero_outputs(clone)          # copied block starts as identity
            new_blocks.append(clone)
        self.transformer.h = nn.ModuleList(new_blocks)
        self.cfg.n_layer = len(new_blocks)

    @staticmethod
    def _zero_outputs(block):
        for name, m in block.named_modules():
            if isinstance(m, nn.Linear) and name.endswith("c_proj"):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---- Phase C: modular spawn (zero-gated) ----
    def spawn_module(self, factory):
        """Append a zero-gated module (modular growth without forgetting)."""
        m = factory()
        nn.init.zeros_(getattr(m, "gate", None) if hasattr(m, "gate") else
                       torch.zeros(1))  # placeholder; gates are linear params
        self.transformer["extra"] = nn.ModuleList([m]) if "extra" not in self.transformer \
            else self.transformer["extra"] + nn.ModuleList([m])

    # ---- forward ----
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok = self.transformer.wte(idx)
        x = tok + self.transformer.wpe(pos)
        if self.fpff is not None:
            x = x + self.cfg.fpff_scale * self.fpff(tok.detach())
        if self.hash_engine is not None:
            x = x + self.cfg.hash_scale * self.hash_engine(tok.detach())
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
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

    def fpff_goodness_gap(self, pos, neg) -> float:
        if self.fpff is None:
            return 0.0
        return (self.fpff.goodness(pos) - self.fpff.goodness(neg)).item()

    # ---- accounting ----
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def flops_per_step(self, tokens: int) -> float:
        return 6.0 * self.n_params() * tokens
