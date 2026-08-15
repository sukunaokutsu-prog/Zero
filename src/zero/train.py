"""Training loop with the Z2G curriculum and live logging/generation.

Phase A — shallow model, toy/curriculum corpus.
Phase B — function-preserving depth doubling + instant switch to the real corpus.
Phase C — (optional) modular spawn.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

from .config import Config
from .model import GPT
from .streaming import MMapDataset
from .tokenizer import load_tokenizer, save_tokenizer


def cosine_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    decay = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (cfg.lr - cfg.min_lr)


def _pos_neg_embeddings(model, ids):
    """Real embeddings vs corrupted (random-token) embeddings, detached."""
    B, T = ids.shape
    pos = model.transformer.wte(ids).detach().view(B * T, -1)
    neg_ids = torch.randint(0, model.cfg.vocab_size, (B, T), device=ids.device)
    neg = model.transformer.wte(neg_ids).detach().view(B * T, -1)
    return pos, neg


@torch.no_grad()
def estimate_loss(model, dataset, cfg):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = dataset.get_batch(split, cfg.batch_size, cfg.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    device = cfg.device

    # ---- data: toy (Phase A) + real (Phase B/C), shared tokenizer ----
    tokenizer = load_tokenizer(cfg.tokenizer_path)
    toy = MMapDataset(cfg.toy_ids_path, tokenizer, cfg.block_size,
                      cfg.split_ratio, seed=cfg.seed)
    real = MMapDataset(cfg.ids_path, tokenizer, cfg.block_size,
                       cfg.split_ratio, seed=cfg.seed)
    cfg.vocab_size = tokenizer.vocab_size
    model = GPT(cfg).to(device)

    def make_optimizer():
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay, betas=cfg.betas)
    optimizer = make_optimizer()

    # ---- logging ----
    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)
    logf = open(os.path.join(run_dir, "train.log"), "a")
    genf = open(os.path.join(run_dir, "generations.txt"), "a")

    def emit(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    tokens_per_step = cfg.batch_size * cfg.block_size
    total_tokens = 0
    total_flops = 0.0
    start_step = 0
    history = []
    resume_path = os.path.join(run_dir, "latest.pt")

    # ---- resume ----
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        ck = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        history = ck.get("history", [])
        total_tokens = ck.get("total_tokens", start_step * tokens_per_step)
        total_flops = ck.get("total_flops", 0.0)
        emit(f"[resume] restored step {start_step} from {cfg.resume_from}")

    emit("=" * 78)
    emit(f"Zero FPFF-Z2G run: {cfg.name}")
    emit(f"config: {json.dumps(cfg.__dict__, default=list)}")
    emit(f"params: {model.n_params():,}  (fpff core {cfg.n_embd}x{cfg.n_embd} = "
         f"{cfg.n_embd * cfg.n_embd:,} floats persistent)")
    emit(f"vocab: {cfg.vocab_size}  toy_tokens: {toy.n_train:,}  real_tokens: {real.n_train:,}")
    emit(f"Phase A (toy, {cfg.n_layer} layers) -> step {cfg.phase_b_step}; "
         f"Phase B (real, {cfg.n_layer * 2} layers) -> step {cfg.max_steps}")
    emit(f"target: {cfg.target_tokens:,} tokens (progress tracked live)")
    emit("=" * 78)

    t_start = time.time()
    best_val = float("inf")

    for step in range(start_step + 1, cfg.max_steps + 1):
        # ---- Z2G Phase B: depth doubling + dataset switch ----
        if step == cfg.phase_b_step:
            model.double_depth()
            del optimizer
            import gc
            gc.collect()
            optimizer = make_optimizer()          # fresh optimizer state
            emit(f"[z2g] Phase B: depth doubled -> n_layer={cfg.n_layer} "
                 f"params={model.n_params():,} | data switch toy -> real")

        # ---- (optional) Phase C: modular spawn ----
        if cfg.phase_c_spawn > 0 and step == cfg.phase_c_spawn:
            emit(f"[z2g] Phase C: modular spawn requested (no-op backbone)")

        dataset = toy if step < cfg.phase_b_step else real
        lr = cosine_lr(step - 1, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = dataset.get_batch("train", cfg.batch_size, device)
        _, loss = model(x, y)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += tokens_per_step
        total_flops += model.flops_per_step(tokens_per_step)

        # ---- FPFF Hebbian positive/negative pass ----
        if cfg.fpff and step % cfg.fpff_hebb_interval == 0:
            pos, neg = _pos_neg_embeddings(model, x)
            model.fpff_hebbian(pos, neg)

        # ---- logging (with live progress toward target tokens) ----
        if step % cfg.log_interval == 0 or step == 1:
            elapsed = time.time() - t_start
            tok_s = total_tokens / max(elapsed, 1e-9)
            pct = 100.0 * total_tokens / max(1, cfg.target_tokens)
            eta_h = max(0.0, (cfg.target_tokens - total_tokens) / max(tok_s, 1e-9)) / 3600.0
            emit(f"step {step:>5}/{cfg.max_steps} | loss {loss.item():.4f} | "
                 f"lr {lr:.2e} | tok/s {tok_s:.0f} | "
                 f"tok {total_tokens:,}/{cfg.target_tokens:,} ({pct:.4f}%) | "
                 f"ETA {eta_h:.1f}h | t {elapsed:.0f}s | phase "
                 f"{'A' if step < cfg.phase_b_step else 'B'}")

        # ---- evaluation (always on the real corpus — the target) ----
        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            losses = estimate_loss(model, real, cfg)
            val = losses["val"]
            history.append({"step": step, "train_loss": losses["train"],
                            "val_loss": val, "val_ppl": math.exp(val),
                            "total_flops": total_flops, "n_params": model.n_params(),
                            "n_layer": cfg.n_layer, "total_tokens": total_tokens})
            emit(f"[eval] step {step:>4} | train {losses['train']:.4f} | val {val:.4f} "
                 f"| ppl {math.exp(val):.2f}")
            if val < best_val:
                best_val = val
                save_checkpoint(model, tokenizer, cfg,
                                os.path.join(run_dir, "best.pt"), val, step)

        # ---- FPFF seed neuroevolution ----
        if cfg.fpff and step % cfg.fpff_evo_interval == 0:
            @torch.no_grad()
            def fitness(seed):
                model.fpff.seed = seed
                total = 0.0
                for _ in range(2):
                    vx, vy = real.get_batch("val", cfg.batch_size, device)
                    _, l = model(vx, vy)
                    total += l.item()
                return total / 2.0
            best_seed = model.fpff_evolve(fitness)
            model.fpff.seed = best_seed
            vx, _ = real.get_batch("val", cfg.batch_size, device)
            pos, neg = _pos_neg_embeddings(model, vx)
            emit(f"[fpff] step {step}: neuroevolution -> seed {best_seed}, "
                 f"goodness gap {model.fpff_goodness_gap(pos, neg):.4f}")

        # ---- live text generation ----
        if step % cfg.gen_interval == 0 or step == 1:
            prompt_ids = torch.tensor(tokenizer.encode(cfg.gen_prompt), dtype=torch.long,
                                      device=device).unsqueeze(0)
            model.eval()
            out = model.generate(prompt_ids, cfg.gen_max_tokens, temperature=0.8)
            model.train()
            sample = tokenizer.decode(out[0].tolist())
            genf.write(f"\n========== step {step} ==========\n{sample}\n")
            genf.flush()
            emit(f"[gen] step {step}: wrote {cfg.gen_max_tokens} tokens")

        # ---- resume checkpoint (model + optimizer + counters) ----
        if step % cfg.checkpoint_interval == 0:
            torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "step": step,
                        "history": history, "total_tokens": total_tokens,
                        "total_flops": total_flops}, resume_path)
            emit(f"[ckpt] step {step}: saved resume checkpoint")

    elapsed = time.time() - t_start
    final_losses = estimate_loss(model, real, cfg)
    save_checkpoint(model, tokenizer, cfg, os.path.join(run_dir, "final.pt"),
                    final_losses["val"], cfg.max_steps)
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": cfg.max_steps,
                "history": history, "total_tokens": total_tokens,
                "total_flops": total_flops}, resume_path)

    results = {
        "name": cfg.name, "final_train_loss": round(final_losses["train"], 5),
        "final_val_loss": round(final_losses["val"], 5),
        "best_val_loss": round(best_val, 5),
        "val_perplexity": round(math.exp(final_losses["val"]), 3),
        "bits_per_char": round(final_losses["val"] / math.log(2.0), 5),
        "steps": cfg.max_steps, "n_layer_final": cfg.n_layer,
        "n_params": model.n_params(), "n_params_phaseA": model.n_params() // 2,
        "total_tokens": total_tokens, "target_tokens": cfg.target_tokens,
        "total_train_flops": total_flops, "total_seconds": round(elapsed, 1),
        "tokens_per_sec": round(total_tokens / max(elapsed, 1e-9), 1),
        "history": history,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    emit(f"[done] val {final_losses['val']:.4f} | ppl {math.exp(final_losses['val']):.2f} "
         f"| {elapsed:.0f}s | {total_tokens:,} tokens "
         f"({100 * total_tokens / cfg.target_tokens:.4f}% of target)")
    logf.close()
    genf.close()
    return results


def save_checkpoint(model, tokenizer, cfg, path, val_loss, step):
    tok_path = os.path.splitext(path)[0] + ".tok.json"
    save_tokenizer(tokenizer, tok_path)
    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                "tokenizer_path": tok_path, "val_loss": val_loss, "step": step}, path)
