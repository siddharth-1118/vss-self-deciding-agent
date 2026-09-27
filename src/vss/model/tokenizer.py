"""VSS tokenizer.

A deliberately simple, fully deterministic word tokenizer:
  - learned vocabulary loaded from a checkpoint or built from a corpus
  - any OOV word is mapped by stable FNV-1a hashing into `hash_buckets`
    pseudo-tokens appended after the base vocab [vss]

Why not subword BPE? The canonical serializer (serialize.py) breaks state
into short typed lines whose lexical surface is highly repetitive. A word
vocab + hash fallback trains on tiny corpora and adds zero runtime deps.
`vocab.json` inside a checkpoint carries the mapping.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# Structural tags are atomic, case-preserved tokens [vss]: they delimit the
# state and question blocks and must never collide with word tokens.
_TAG_RE_PART = r"</?(?:STATE|QUESTION)>"
_TOKEN_RE = re.compile(rf"{_TAG_RE_PART}|\w+|[^\w\s]", re.UNICODE)

RESERVED_TAGS = ["<STATE>", "</STATE>", "<QUESTION>", "</QUESTION>"]


def fnv1a(data: bytes) -> int:
    """Stable 32-bit FNV-1a — identical across runs, processes and machines."""
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class VSSTokenizer:
    """Word-level tokenizer with reserved special tokens and OOV hashing."""

    pad_token: str = "<pad>"

    def __init__(self, vocab_size: int = 24576, hash_buckets: int = 8192) -> None:
        self.vocab_size = int(vocab_size)
        self.hash_buckets = int(hash_buckets)
        self.token_to_id: dict[str, int] = {self.pad_token: 0}
        for i, tag in enumerate(RESERVED_TAGS, start=1):
            self.token_to_id.setdefault(tag, i)
        self._built = False

    # ------------------------------------------------------------------ fit
    def fit(self, texts: Iterable[str]) -> "VSSTokenizer":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(self._words(t))
        most_common = counter.most_common(
            max(0, self.vocab_size - len(self.token_to_id))
        )
        for word, _ in most_common:
            self.token_to_id.setdefault(word, len(self.token_to_id))
        self._built = True
        return self

    def require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Tokenizer has no vocabulary; call fit() or load() first")

    # ---------------------------------------------------------------- encode
    def _words(self, text: str) -> list[str]:
        out: list[str] = []
        for tok in _TOKEN_RE.findall(text):
            if tok in RESERVED_TAGS:
                out.append(tok)  # case-preserved structural tag
            else:
                out.append(tok.lower())
        return out

    def encode(self, text: str) -> list[int]:
        self.require_built()
        ids: list[int] = []
        n_base = len(self.token_to_id)
        for word in self._words(text):
            tid = self.token_to_id.get(word)
            if tid is None:
                # deterministic OOV bucket [vss]
                tid = n_base + fnv1a(word.encode("utf-8")) % self.hash_buckets
            ids.append(tid)
        return ids

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"vocab": self.token_to_id, "hash_buckets": self.hash_buckets}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VSSTokenizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=1, hash_buckets=int(raw.get("hash_buckets", 8192)))
        tok.token_to_id = dict(raw["vocab"])
        tok._built = True
        return tok

    def __len__(self) -> int:
        return self.vocab_size + self.hash_buckets
