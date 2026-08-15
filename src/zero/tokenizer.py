"""Tokenizers: character-level and fast byte-level BPE (via `tokenizers`)."""
from __future__ import annotations

import json
from typing import List


class CharTokenizer:
    def __init__(self, stoi: dict, itos: dict):
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)

    @classmethod
    def build(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for i, c in enumerate(chars)}
        return cls(stoi, itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharTokenizer":
        return cls({k: int(v) for k, v in state["stoi"].items()},
                   {int(k): v for k, v in state["itos"].items()})


def _has_tokenizers() -> bool:
    try:
        import tokenizers  # noqa: F401
        return True
    except Exception:
        return False


class BPETokenizer:
    """Byte-level BPE wrapper over the Rust `tokenizers` library."""

    def __init__(self, tok):
        self._t = tok
        self.vocab_size = tok.get_vocab_size()

    @classmethod
    def build(cls, text: str, vocab_size: int = 2048) -> "BPETokenizer":
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[],
                                      min_frequency=1)
        tok.train_from_iterator([text], trainer=trainer)
        return cls(tok)

    @classmethod
    def build_streaming(cls, text_path: str, vocab_size: int = 2048,
                        chunk_chars: int = 1_000_000) -> "BPETokenizer":
        """Train BPE by streaming a file in chunks (bounded RAM)."""
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[],
                                      min_frequency=1)

        def chunks():
            with open(text_path, "r", encoding="utf-8", errors="replace") as f:
                while True:
                    c = f.read(chunk_chars)
                    if not c:
                        break
                    yield c

        tok.train_from_iterator(chunks(), trainer=trainer)
        return cls(tok)

    def save(self, path: str):
        self._t.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(path))

    def encode(self, s: str):
        return self._t.encode(s).ids

    def decode(self, ids) -> str:
        return self._t.decode(ids)


def build_tokenizer(text_or_path: str, kind: str, vocab_size: int = 2048):
    if kind == "bpe":
        if not _has_tokenizers():
            raise RuntimeError("`tokenizers` library required for BPE")
        return BPETokenizer.build_streaming(text_or_path, vocab_size)
    with open(text_or_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return CharTokenizer.build(text)


def save_tokenizer(tok, path: str):
    if hasattr(tok, "save"):
        tok.save(path)
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "char", "state": tok.state_dict()}, f)


def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(64)
    if head.lstrip().startswith("{"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if obj.get("type") == "char":
                return CharTokenizer.from_state_dict(obj["state"])
        except Exception:
            pass
        return BPETokenizer.load(path)
    raise ValueError(f"unrecognized tokenizer file: {path}")
