"""Data-leakage audit for VSS JSONL datasets.

Reproducible checks:

  1. exact duplicates       - normalized-content hash, within and across splits
  2. near duplicates        - token-Jaccard pairs above thresholds
  3. generator leakage      - best-match similarity of every test example
                              against the whole train split
  4. question-template audit - which (question id, type, options) rows and
                              question-id-only leakage is present

Usage:
    python benchmarks/leakage/audit_leakage.py \
        --train data/generated/train.jsonl --test data/generated/eval.jsonl \
        --out benchmarks/leakage/synthetic_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vss.data.schema import TrainingExample  # noqa: E402
from vss.data.schema import load_jsonl  # noqa: E402

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation -> leakage-proof text."""
    return " ".join(_WORD.findall(text.lower()))


def example_text(ex: TrainingExample) -> str:
    parts = []
    if isinstance(ex.state, dict):
        for k in sorted(ex.state):
            v = ex.state[k]
            parts.append(str(v) if not isinstance(v, (dict, list)) else json.dumps(v, sort_keys=True))
    else:
        parts.append(json.dumps(ex.state, sort_keys=True))
    return " ".join(parts)


def example_signature(ex: TrainingExample) -> str:
    """Hash of (normalized state text, question schema, answer)."""
    qsig = json.dumps(
        [
            {
                "id": q.id,
                "type": q.type,
                "options": q.options or [],
                "answer": str(q.answer),
            }
            for q in ex.questions
        ],
        sort_keys=True,
    )
    raw = normalize_text(example_text(ex)) + "||" + qsig
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def soft_signature(ex: TrainingExample) -> str:
    """Hash ignoring answers (detects same-input-different-split duplicates)."""
    qsig = json.dumps(
        [{"id": q.id, "type": q.type, "options": q.options or []} for q in ex.questions],
        sort_keys=True,
    )
    raw = normalize_text(example_text(ex)) + "||" + qsig
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tokens(ex: TrainingExample) -> set[str]:
    return set(normalize_text(example_text(ex)).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cross_split_near_duplicates(
    train: list[TrainingExample],
    test: list[TrainingExample],
    j_threshold: float = 0.7,
) -> dict[str, Any]:
    """For each test example, best Jaccard vs train. O(N*M) token Jaccard —
    fine for tens of thousands; inverted index used to keep it fast."""
    train_toks = [_tokens(t) for t in train]
    index: dict[str, set[int]] = {}
    for i, toks in enumerate(train_toks):
        for tok in toks:
            index.setdefault(tok, set()).add(i)
    best: list[float] = []
    over: list[tuple[int, float]] = []
    for j, ex in enumerate(test):
        toks = _tokens(ex)
        cand: set[int] = set()
        for tok in toks:
            cand |= index.get(tok, set())
        best_j = 0.0
        for i in cand:
            s = _jaccard(toks, train_toks[i])
            if s > best_j:
                best_j = s
        best.append(best_j)
        if best_j >= j_threshold:
            over.append((j, best_j))
    return {
        "j_threshold": j_threshold,
        "n_test": len(test),
        "mean_best_match_jaccard": sum(best) / max(1, len(best)),
        "p50_best": sorted(best)[len(best) // 2],
        "p95_best": sorted(best)[min(len(best) - 1, int(0.95 * len(best)))],
        "fraction_above_threshold": len(over) / max(1, len(test)),
        "n_above_threshold": len(over),
        "examples": [
            {"test_index": j, "jaccard": round(s, 3)} for j, s in over[:20]
        ],
    }


def question_template_audit(
    train: list[TrainingExample], test: list[TrainingExample]
) -> dict[str, Any]:
    """Question-side leakage: identical (id, type, options) tuples across splits."""
    def rows(exs: list[TrainingExample]) -> list[tuple[str, str, tuple]]:
        out = []
        for ex in exs:
            for q in ex.questions:
                out.append((q.id, q.type, tuple(q.options or [])))
        return out

    tr = rows(train)
    te = rows(test)
    tr_set = set(tr)
    overlap = sum(1 for r in te if r in tr_set)
    ids_train = {r[0] for r in tr}
    ids_test = {r[0] for r in te}
    return {
        "question_id_overlap": sorted(ids_train & ids_test),
        "question_id_overlap_fraction": len(ids_train & ids_test) / max(1, len(ids_test)),
        "exact_schema_rows_in_both": overlap,
        "exact_schema_rows_in_both_fraction": overlap / max(1, len(te)),
        "note": (
            "shared (id,type,options) tuples are EXPECTED for a decision API "
            "(the deployed API declares the same options at train and test time); "
            "the leakage that matters is STATE-side template sharing"
        ),
    }


def exact_duplicate_audit(
    train: list[TrainingExample], test: list[TrainingExample]
) -> dict[str, Any]:
    """Exact (answer-aware) and soft (answer-agnostic) duplicate counts."""
    from collections import Counter

    tr_hard = Counter(example_signature(t) for t in train)
    te_hard = Counter(example_signature(t) for t in test)
    tr_soft = Counter(soft_signature(t) for t in train)
    te_soft = Counter(soft_signature(t) for t in test)
    hard_cross = len(set(te_hard) & set(tr_hard))
    soft_cross = len(set(te_soft) & set(tr_soft))
    return {
        "train_exact_internal_dups": sum(c - 1 for c in tr_hard.values() if c > 1),
        "test_exact_internal_dups": sum(c - 1 for c in te_hard.values() if c > 1),
        "train_soft_internal_dups": sum(c - 1 for c in tr_soft.values() if c > 1),
        "test_soft_internal_dups": sum(c - 1 for c in te_soft.values() if c > 1),
        "cross_split_exact_dup_test_examples": hard_cross,
        "cross_split_answer_agnostic_dup_test_examples": soft_cross,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True, help="output JSON report path")
    ap.add_argument("--j-threshold", type=float, default=0.7)
    args = ap.parse_args()

    train = load_jsonl(args.train)
    test = load_jsonl(args.test)
    report = {
        "train_file": args.train,
        "test_file": args.test,
        "n_train": len(train),
        "n_test": len(test),
        "exact_duplicates": exact_duplicate_audit(train, test),
        "near_duplicates_cross_split": cross_split_near_duplicates(
            train, test, args.j_threshold
        ),
        "question_templates": question_template_audit(train, test),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
