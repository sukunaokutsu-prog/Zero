"""Benchmark + generation harness for trained checkpoints."""
from __future__ import annotations

import argparse
import math

import torch

from .config import Config
from .model import GPT
from .streaming import MMapDataset
from .tokenizer import load_tokenizer


def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    fields = {k: v for k, v in ckpt["cfg"].items() if k in Config.__dataclass_fields__}
    cfg = Config(**fields)
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tokenizer = load_tokenizer(ckpt["tokenizer_path"])
    return model, cfg, tokenizer, ckpt


@torch.no_grad()
def evaluate(path: str, eval_iters: int = 50, device: str = "cpu"):
    model, cfg, tokenizer, ckpt = load_checkpoint(path, device)
    model.eval()
    dataset = MMapDataset(cfg.ids_path, tokenizer, cfg.block_size,
                          cfg.split_ratio, seed=cfg.seed)
    losses = torch.zeros(eval_iters)
    accs = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = dataset.get_batch("val", cfg.batch_size, device)
        logits, loss = model(x, y)
        losses[k] = loss.item()
        preds = logits.view(-1, logits.size(-1)).argmax(-1)
        accs[k] = (preds == y.view(-1)).float().mean().item()
    return {
        "name": cfg.name, "val_loss": round(losses.mean().item(), 5),
        "perplexity": round(math.exp(losses.mean().item()), 3),
        "next_token_accuracy": round(accs.mean().item(), 4),
        "n_params": model.n_params(), "n_layer": cfg.n_layer,
    }


@torch.no_grad()
def generate(path: str, prompt: str, max_tokens: int = 200, temperature: float = 0.8,
             top_k: int = None, seed: int = 0, device: str = "cpu") -> str:
    model, cfg, tokenizer, _ = load_checkpoint(path, device)
    model.eval()
    torch.manual_seed(seed)
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--prompt", default="CHAPTER I.")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    r = evaluate(args.ckpt, args.eval_iters)
    print(f"{r['name']}: val {r['val_loss']} ppl {r['perplexity']} "
          f"acc {r['next_token_accuracy']} params {r['n_params']:,} layers {r['n_layer']}")
    print("\n=== generation ===")
    print(generate(args.ckpt, args.prompt, args.max_tokens, args.temperature))


if __name__ == "__main__":
    main()
