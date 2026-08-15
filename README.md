# Zero — the FPFF-Z2G engine

A from-scratch pipeline for training a **50M-parameter language model on CPU-only
hardware**, implementing the **Fractal-Procedural Forward-Forward + Zero-to-Growth
(FPFF-Z2G)** architecture.

| component | what it does |
|---|---|
| **Z2G curriculum** | Phase A: shallow model on toy data → Phase B: function-preserving depth doubling (8→16 layers) + instant switch to the real corpus → Phase C: modular spawn |
| **Hash-weight engine** | coordinate-based procedural ternary weights `W[i,j] ∈ {-1,0,1}` — zero stored weight bytes, `O(1)` memory |
| **Fractal dynamical core** | one fixed matrix recursively masked, with a **dynamic attractor exit** (`‖h_t − h_{t-1}‖ < ε`) — infinite virtual depth in fixed SRAM |
| **Calculus-free learning** | Hebbian forward-forward (positive pass strengthens, negative pass weakens) + seed **neuroevolution** |
| **Context + RAG** | sparse context compressor (first/trailing/middle → latent) + BM25 RAG curator |

The generative engine (a GPT decoder) is backprop-trained; the FPFF core learns
*without backprop* and enriches the token stream. Full writeup: `paper/method.md`.

---

## The model (50M parameters)

| | Phase A | Phase B |
|---|---|---|
| layers | 8 (~26M params) | **16 (~51.5M params)** |
| dim / heads | 512 / 8 | 512 / 8 |
| data | toy (TinyShakespeare) | **real corpus (36 novels)** |
| context / batch | 256 / 32 | 256 / 32 |

Fits and trains in ~1 GB of RAM on 2 vCPU.

---

## Quickstart

```bash
bash scripts/setup.sh                                  # venv + CPU torch
.venv/bin/python scripts/fetch_corpus.py --max-mb 32   # 36 novels (+ TinyShakespeare)
.venv/bin/python scripts/prep_data.py                  # BPE + mmap token files
PYTHONPATH=src .venv/bin/python -c \
  "from zero.methods import fifty_m; from zero.train import train; train(fifty_m(200))"
```

Then:

```bash
PYTHONPATH=src .venv/bin/python scripts/demo.py --ckpt results/fifty-m/best.pt
```

Outputs: `results/fifty-m/train.log` (per-step loss/lr/tok/s/FLOPs),
`results/fifty-m/generations.txt` (**text generated live every 50 steps**),
`results/fifty-m/best.pt` (checkpoint, saved locally).

---

## Honest limits

This machine (2 vCPU / 3.8 GB RAM) trains the 50M model at a few hundred tokens/sec
— a real model, genuinely trained from scratch, but not converged to a frontier
scale. The techniques (progressive growth, procedural weights, forward-forward,
neuroevolution) are the same ones used to make large-model training cheaper; they
transfer to bigger hardware. `paper/method.md` documents what worked, what didn't,
and why.
