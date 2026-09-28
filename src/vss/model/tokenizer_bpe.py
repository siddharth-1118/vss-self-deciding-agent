"""Byte-pair-encoding tokenizer for VSS (drop-in VSSTokenizer alternative).

Same fit/encode/save/load interface as VSSTokenizer so the model, trainer
and checkpoints are tokenizer-agnostic. Words are split into character
pieces (plus an end-of-word marker) and merged by learned pair frequency;
reserved structural tags stay atomic. Unknown characters are skipped (a
byte-level fallback can replace this later).

Motivation: the robustness suite quantified the word+hash path's
surface-form brittleness (one typo = -8.8 pts, one dropped word = -10.4 pts
on CLINC150). Subword pieces should absorb character-level noise; this
module exists to test that hypothesis with an otherwise-identical recipe.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .tokenizer import VSSTokenizer

_END = "</w>"


class BPETokenizer(VSSTokenizer):
    """Word-fragment BPE with learned merge table and atomic reserved tags."""

    def __init__(self, vocab_size: int = 16384, hash_buckets: int = 0) -> None:
        super().__init__(vocab_size=vocab_size, hash_buckets=0)
        self.merges: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ fit
    def fit(self, texts) -> "BPETokenizer":  # type: ignore[override]
        # incremental BPE: pair counts + pair->word index, so each merge only
        # touches words that actually contain the merged pair [std]
        word_counter: Counter[tuple[str, ...]] = Counter()
        for t in texts:
            for tok in self._words(t):
                word_counter[tuple(tok) + (_END,)] += 1
        words: list[tuple[tuple[str, ...], int]] = list(word_counter.items())

        # seed vocab with observed single characters + end marker
        piece_counts: Counter[str] = Counter()
        for word, c in words:
            for p in word:
                piece_counts[p] += c
        for p, _ in piece_counts.most_common(
            max(0, self.vocab_size - len(self.token_to_id))
        ):
            self.token_to_id.setdefault(p, len(self.token_to_id))

        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[int]] = {}
        for wi, (w, c) in enumerate(words):
            for ab in zip(w, w[1:]):
                pair_counts[ab] += c
                pair_words.setdefault(ab, set()).add(wi)

        target_merges = max(0, self.vocab_size - len(self.token_to_id))
        for _ in range(target_merges):
            if not pair_counts:
                break
            # deterministic: highest count, lexicographically smallest pair
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if pair_counts[best] < 2:
                break
            self.merges.append(best)
            new_sym = best[0] + best[1]
            if new_sym not in self.token_to_id:
                self.token_to_id[new_sym] = len(self.token_to_id)
            affected = sorted(pair_words.pop(best, ()))
            for wi in affected:
                w, c = words[wi]
                if best not in zip(w, w[1:]):
                    continue  # stale index entry
                for ab in zip(w, w[1:]):
                    pair_counts[ab] -= c
                    if pair_counts[ab] <= 0:
                        del pair_counts[ab]
                    pair_words.get(ab, set()).discard(wi)
                out: list[str] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        out.append(new_sym)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                words[wi] = (tuple(out), c)
                for ab in zip(out, out[1:]):
                    pair_counts[ab] += c
                    pair_words.setdefault(ab, set()).add(wi)
        self._built = True
        return self

    # ---------------------------------------------------------------- encode
    def _pieces(self, word: str) -> list[str]:
        w = list(word) + [_END]
        for a, b in self.merges:
            i = 0
            while i < len(w) - 1:
                if w[i] == a and w[i + 1] == b:
                    w[i:i + 2] = [a + b]
                else:
                    i += 1
        return w

    def encode(self, text: str) -> list[int]:  # type: ignore[override]
        self.require_built()
        ids: list[int] = []
        for word in self._words(text):
            for piece in self._pieces(word):
                tid = self.token_to_id.get(piece)
                if tid is not None:  # unseen chars skipped (no hash fallback)
                    ids.append(tid)
        return ids

    # --------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "type": "bpe",
            "vocab": self.token_to_id,
            "merges": [list(m) for m in self.merges],
            "hash_buckets": 0,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=1, hash_buckets=0)
        if raw.get("type") != "bpe":
            raise ValueError(f"{path} is not a BPE tokenizer file")
        tok.token_to_id = dict(raw["vocab"])
        tok.merges = [tuple(m) for m in raw["merges"]]
        tok._built = True
        return tok
