"""Method recipes — including the 50M-parameter FPFF-Z2G model."""
from __future__ import annotations

from .config import Config


def fifty_m(max_steps: int = 200, name: str = "fifty-m") -> Config:
    """The 50M-parameter FPFF-Z2G model.

    Geometry: n_embd=512, n_head=8, block=256. Phase A = 8 layers (~26M params)
    on the toy corpus; Phase B doubles depth to 16 layers (~51.5M params) and
    switches to the real corpus. FPFF fractal core + procedural hash weights are
    active (calculus-free), Hebbian + neuroevolution every N steps.
    """
    cfg = Config()
    cfg.name = name
    cfg.tokenizer = "bpe"
    cfg.tokenizer_path = "data/mixed.tok.json"
    cfg.ids_path = "data/mixed.ids.bin"
    cfg.toy_path = "data/toy.txt"
    cfg.toy_ids_path = "data/toy.ids.bin"

    cfg.n_layer = 8
    cfg.n_head = 8
    cfg.n_embd = 512
    cfg.block_size = 128
    cfg.batch_size = 16

    cfg.growth = "net2net"
    cfg.phase_b_step = max(1, max_steps // 3)

    cfg.fpff = True
    cfg.fpff_loops = 6
    cfg.fpff_eps = 1e-3
    cfg.fpff_scale = 0.1
    cfg.fpff_lr = 1e-3
    cfg.fpff_pop = 8
    cfg.fpff_hebb_interval = 10
    cfg.fpff_evo_interval = 50
    cfg.hash_engine = True
    cfg.hash_scale = 0.05

    cfg.max_steps = max_steps
    cfg.warmup_steps = 40
    cfg.lr = 3e-3
    cfg.weight_decay = 0.1
    cfg.grad_clip = 1.0
    cfg.eval_interval = 40
    cfg.eval_iters = 10
    cfg.log_interval = 5
    cfg.gen_interval = 50
    cfg.gen_max_tokens = 200
    cfg.gen_prompt = "CHAPTER I."
    cfg.seed = 1337
    return cfg


def tiny(max_steps: int = 30, name: str = "tiny") -> Config:
    """A small smoke-test config (same pipeline, tiny model)."""
    cfg = Config()
    cfg.name = name
    cfg.tokenizer = "bpe"
    cfg.tokenizer_path = "data/mixed.tok.json"
    cfg.ids_path = "data/mixed.ids.bin"
    cfg.toy_path = "data/toy.txt"
    cfg.toy_ids_path = "data/toy.ids.bin"
    cfg.n_layer = 2
    cfg.n_head = 4
    cfg.n_embd = 128
    cfg.block_size = 128
    cfg.batch_size = 16
    cfg.phase_b_step = max_steps // 2
    cfg.max_steps = max_steps
    cfg.fpff = True
    cfg.fpff_loops = 4
    cfg.fpff_pop = 4
    cfg.fpff_hebb_interval = 5
    cfg.fpff_evo_interval = 10
    cfg.hash_engine = True
    cfg.eval_interval = 10
    cfg.eval_iters = 5
    cfg.log_interval = 5
    cfg.gen_interval = 10
    cfg.gen_max_tokens = 40
    return cfg


METHODS = {"fifty-m": fifty_m, "tiny": tiny}
