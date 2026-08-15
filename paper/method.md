# The FPFF-Z2G Engine — training a 50M-parameter model from scratch on CPU

A complete, from-scratch training pipeline implementing the **Fractal-Procedural
Forward-Forward + Zero-to-Growth** architecture, and its first real deployment:
a **~51.5M-parameter** language model trained entirely on a 2-vCPU / 3.8 GB-RAM
machine with no GPU.

---

## 1. Architecture

### 1.1 Z2G curriculum (training & scaling)

| phase | model | data | mechanism |
|---|---|---|---|
| **A — curriculum init** | 8 layers (~26.3M params) | toy corpus (TinyShakespeare) | shallow, fast-calculating core learns base syntactic geometry |
| **B — temporal expansion + data switch** | **16 layers (~51.5M params)** | real corpus (36 novels) | function-preserving depth doubling (each block is copied with zeroed residual outputs → no loss spike) + instant dataset switch |
| **C — modular spawn** | + modules | — | zero-gated module spawning (mechanism provided; not triggered in this run) |

Depth doubling is *function-preserving*: the copied blocks compute the identity,
so the loss does not spike at the growth boundary (visible in `train.log`).

### 1.2 Hash-weight engine (procedural ternary weights)

Every weight coordinate `W[i,j]` is generated on the fly from a deterministic
integer hash of `(master_seed, i, j, ctx)` into `{-1, 0, 1}` and dropped after
use. Persistent storage = one seed integer (**zero bytes of stored weights**).
Because the weights are ternary, the matmul reduces to signed addition.
(Measured honestly: the masks are materialized transiently per forward, so this
is "O(1) *persistent* storage", not zero transient memory.)

### 1.3 Fractal dynamical core (infinite depth in fixed SRAM)

One persistent `d×d` matrix `W_core` is recursively transformed:

```
h_{t+1} = LayerNorm( σ( h_t · (W_core ⊙ M_procedural(t)) ) )
```

with a **dynamic attractor exit condition** — recursion stops when
`‖h_t − h_{t−1}‖ < ε`, not at a fixed depth. For this run `d = 512`, so the
core is a single 512×512 buffer (262,144 floats ≈ 1 MB).

### 1.4 Calculus-free learning

* **Hebbian forward-forward** — positive pass strengthens the core toward real
  data (`ΔW ∝ h_posᵀh_pos`), negative pass weakens it (`ΔW ∝ −h_negᵀh_neg`),
  Oja-normalized for stability. No `loss.backward()` through the core.
* **Seed neuroevolution** — each seed is scored by *validation loss* ("lowest
  error on the data"), the worst half is deleted, the fittest is mutated by
  bit-flips, and the fittest is deployed.

### 1.5 Context compression + RAG curator

* **Sparse context compressor** — long context → first tokens + trailing tokens
  + a deterministic pseudo-random middle sample → projected to one latent vector.
* **RAG curator** — BM25 lexical retrieval over the training corpus, returning
  dense, boilerplate-stripped statements to the model.

---

## 2. Division of labour

The FPFF core and hash engine enrich the token-embedding stream
(`x = tok + wpe + 0.1·FPFF(tok) + 0.05·Hash(tok)`) **with no gradient flowing
through them** — they are calculus-free feature extractors. The GPT backbone is
the generative engine, backprop-trained. This is the honest realisation of the
spec: forward-forward produces *representations*, a language model needs
*logits*, so the two are combined rather than conflated.

---

## 3. The 50M model

| | Phase A | Phase B |
|---|---|---|
| layers | 8 | 16 |
| parameters | 26,297,344 | **51,479,552** |
| dim / heads | 512 / 8 | 512 / 8 |
| context / batch | 128 / 16 | 128 / 16 |
| tokenizer | byte-level BPE, vocab 2048 | ← |
| data | TinyShakespeare (405k tokens) | 36 novels (9.4M tokens) |

Memory: ~2.1 GB peak (model 206 MB + Adam state ~412 MB + grads 206 MB +
activations) — comfortably inside 3.8 GB.

Throughput: ~1,000 tok/s in Phase A, ~600 tok/s in Phase B (2 vCPU).

---

## 4. Results

Run: `fifty-m`, 300 steps (Phase A 100 + Phase B 200), 2 vCPU, 19.5 min.

| metric | value |
|---|---|
| final model | **51,479,552 params** (16 layers, 512-dim, 8 heads) |
| Phase A → B growth | 8 → 16 layers, function-preserving (no loss spike) |
| training tokens | 204,800 toy + 409,600 real = 614,400 |
| total FLOPs | 1.59e14 |
| throughput | ~1,000 tok/s (8L) / ~600 tok/s (16L) |
| final train loss | 5.163 |
| **final val loss** | **5.243** (ppl 189, BPE vocab 2048) |

Learning trajectory (val loss on the real corpus):

| step | phase | train loss | val loss |
|---|---|---|---|
| 80 | A (toy) | 6.446 | 6.439 |
| 120 | B | 5.692 | 5.770 |
| 200 | B | 5.295 | 5.333 |
| 280 | B | 5.173 | 5.203 |
| 300 | B | 5.163 | 5.243 |

**Observations (honest).**

1. The Z2G curriculum ran exactly as specified: shallow toy init → function-
   preserving 8→16 depth doubling (no loss spike at step 100) → real-corpus
   training. The Phase-A model visibly learned the toy data's structure
   (Shakespeare `CHARACTER:` stage directions in its step-50 sample).
2. Generation progressed from gibberish → word fragments → coherent quoted
   dialogue, and the step-300 sample recovers a real corpus name ("Cosette",
   *Les Misérables*). See `results/fifty-m/generations.txt`.
3. The model is **under-trained**, not converged: 614k tokens is ~3 orders of
   magnitude short of what a 50M model needs (~1B tokens). Generations remain
   word-salad-heavy — that is the honest state of a 50M model after 19 minutes
   on 2 CPU cores, not a bug.
4. FPFF's seed neuroevolution showed **no measurable fitness signal** at this
   scale (goodness gap ≈ 0; the fittest seed stayed 42). Reported as-is: the
   FPFF core's value here is its memory/compute profile (one 512×512 buffer,
   zero gradient memory, Hebbian-only), not a loss improvement.
5. The **RAG curator works**: BM25 retrieves the correct corpus passages for a
   query (e.g. "time machine travel future" → the Time Traveller dialogue).

Loss curve: `results/fifty-m/loss_curve.png`.

---

## 5. Reproducibility

```bash
bash scripts/setup.sh
.venv/bin/python scripts/fetch_corpus.py --max-mb 32
.venv/bin/python scripts/prep_data.py
PYTHONPATH=src .venv/bin/python -c "from zero.methods import fifty_m; from zero.train import train; train(fifty_m(300))"
PYTHONPATH=src .venv/bin/python scripts/demo.py --ckpt results/fifty-m/best.pt
```

---

## 6. Honest limitations

* A 50M model needs ~billions of tokens to converge; this run shows the full
  pipeline and real learning (loss falls, generations improve) but is not a
  converged frontier model. Throughput on 2 vCPU is ~600–1000 tok/s.
* Procedural ternary weights are random (untrained) projections — they cannot
  *learn* the way backprop weights do; only their seeds evolve. This is an
  inherent property of "zero stored weights", stated plainly.
* The fractal core learns only via Hebbian + neuroevolution, which is weaker
  than backprop for generative quality at this scale — its contribution is the
  *memory/compute* profile (1 MB core, zero gradient memory), not a loss win.
