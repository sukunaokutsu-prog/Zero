"""Disk-backed (memory-mapped) token corpora — train on data larger than RAM."""
from __future__ import annotations

import numpy as np
import torch


def tokenize_to_mmap(text_path: str, tokenizer, out_bin: str,
                     chunk_chars: int = 2_000_000) -> int:
    """Tokenize a text file to a memory-mapped int32 file in bounded chunks."""
    parts = []
    n = 0
    with open(text_path, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            ids = np.asarray(tokenizer.encode(chunk), dtype=np.int32)
            parts.append(ids)
            n += len(ids)
    arr = np.concatenate(parts) if len(parts) > 1 else parts[0]
    mm = np.memmap(out_bin, dtype=np.int32, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    del mm
    return n


class MMapDataset:
    """Random-access sequence dataset over a memory-mapped token file."""

    def __init__(self, bin_path: str, tokenizer, block_size: int,
                 split_ratio: float = 0.9, seed: int = 0):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.block_size = block_size
        ids = np.memmap(bin_path, dtype=np.int32, mode="r")
        n = int(split_ratio * len(ids))
        self.train = ids[:n]
        self.val = ids[n:]
        self.n_train = len(self.train)
        self.n_val = len(self.val)
        self.rng = np.random.default_rng(seed)

    def get_batch(self, split: str, batch_size: int, device: str = "cpu"):
        data = self.train if split == "train" else self.val
        ix = self.rng.integers(0, len(data) - self.block_size - 1, size=batch_size)
        x = np.stack([data[i:i + self.block_size] for i in ix])
        y = np.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return (torch.from_numpy(x).long().to(device),
                torch.from_numpy(y).long().to(device))
