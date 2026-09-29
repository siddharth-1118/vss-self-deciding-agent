"""Per-question metrics for the multi-question value benchmark.

Every metric operates on flat per-question records (never per-request
aggregates) so that per-question gold scoring is the default. Records are
dicts produced by the harness extractors:

    {
      "qid": str, "type": "choice"|"noul"|"score",
      "correct": bool,
      "conf": float,            # confidence used for selective metrics
      "max_prob": float,        # probability of the predicted answer
      "probs": dict|None,       # choice: {option: p} over DECLARED options
      "p_true": float|None,     # noul: P(answer true)
      "value": float|None,      # score: predicted value
      "gold": ..., "pred": ..., # for failure analysis
    }

Definitions match the repository's existing calibration audit
(benchmarks/audit_calibration.py) and OOD tooling where applicable.
"""
from __future__ import annotations

import math
from collections import defaultdict

SCORE_TOL_FRAC = 0.10  # score correct iff |pred - gold| <= 10% of range


def correct_score(pred: float, gold: float, lo: float, hi: float) -> bool:
    tol = SCORE_TOL_FRAC * (hi - lo)
    return abs(pred - gold) <= tol


# ------------------------------------------------------------------ accuracy

def accuracy(records: list[dict]) -> float:
    if not records:
        return float("nan")
    return sum(1 for r in records if r["correct"]) / len(records)


def per_type_accuracy(records: list[dict]) -> dict[str, float]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["type"]].append(r)
    return {t: accuracy(rs) for t, rs in sorted(by_type.items())}


def macro_f1(records: list[dict]) -> float:
    """Macro-F1 over discrete labels (choice options, noul classes).

    Score questions contribute MAE separately (see score_mae); they have no
    discrete labels, so they are excluded from F1 by construction.
    """
    tp: dict[str, float] = defaultdict(float)
    fp: dict[str, float] = defaultdict(float)
    fn: dict[str, float] = defaultdict(float)
    n_used = 0
    for r in records:
        if r["type"] == "score":
            continue
        n_used += 1
        gold, pred = str(r["gold"]), str(r["pred"])
        if pred == gold:
            tp[gold] += 1
        else:
            fn[gold] += 1
            fp[pred] += 1
    if n_used == 0:
        return float("nan")
    f1s = []
    for label in set(tp) | set(fp) | set(fn):
        prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def request_accuracy(records_by_request: list[list[dict]]) -> float:
    """Fraction of requests where EVERY question was answered correctly."""
    if not records_by_request:
        return float("nan")
    ok = 0
    for rows in records_by_request:
        ok += 1 if rows and all(r["correct"] for r in rows) else 0
    return ok / len(records_by_request)


# --------------------------------------------------------------- calibration

def ece(confidences: list[float], corrects: list[bool], n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    if not confidences:
        return float("nan")
    edges = [i / n_bins for i in range(n_bins + 1)]
    total = len(confidences)
    err = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == n_bins - 1 and c == hi)]
        if not idx:
            continue
        acc = sum(1.0 for i in idx if corrects[i]) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        err += (len(idx) / total) * abs(acc - conf)
    return err


def brier(records: list[dict]) -> float:
    """Multi-class Brier score, averaged per question.

    choice: sum_i (p_i - y_i)^2 over the DECLARED options (y one-hot at gold).
    noul:   (p_true - y)^2.
    score:  one-hot over 64 bins: sum_i (p_i - y_i)^2 (bins normalized to
            [min, max]); the harness stores score probs already.
    """
    if not records:
        return float("nan")
    tot = 0.0
    for r in records:
        if r["type"] == "choice" and r.get("probs"):
            gold = str(r["gold"])
            tot += sum((p - (1.0 if k == gold else 0.0)) ** 2
                       for k, p in r["probs"].items())
        elif r["type"] == "noul":
            y = float(r["gold"])
            tot += (float(r["p_true"]) - y) ** 2
        elif r["type"] == "score" and r.get("probs"):
            # probs over normalized bins, gold bin index stored by harness
            gi = int(r.get("gold_bin", -1))
            tot += sum((p - (1.0 if i == gi else 0.0)) ** 2
                       for i, p in enumerate(r["probs"]))
        else:
            tot += float("nan")
    return tot / len(records)


def nll(records: list[dict]) -> float:
    """Mean negative log-likelihood of the gold answer."""
    vals: list[float] = []
    for r in records:
        if r["type"] == "choice" and r.get("probs"):
            p = r["probs"].get(str(r["gold"]), 0.0)
            vals.append(-math.log(max(p, 1e-9)))
        elif r["type"] == "noul":
            p = min(max(float(r["p_true"]), 1e-9), 1 - 1e-9)
            y = float(r["gold"])
            vals.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
        elif r["type"] == "score" and r.get("probs"):
            gi = int(r.get("gold_bin", -1))
            p = r["probs"][gi] if 0 <= gi < len(r["probs"]) else 0.0
            vals.append(-math.log(max(p, 1e-9)))
    return sum(vals) / len(vals) if vals else float("nan")


def score_mae(records: list[dict]) -> float:
    vals = [abs(float(r["value"]) - float(r["gold"])) for r in records
            if r["type"] == "score" and r.get("value") is not None]
    return sum(vals) / len(vals) if vals else float("nan")


# ------------------------------------------------------------ OOD / selective

def auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC via rank statistics (labels: 1 = positive/OOD)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    r_pos = sum(ranks[i] for i, l in enumerate(labels) if l == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def risk_coverage(records: list[dict], coverage_fractions: list[float]) -> dict:
    """Selective accuracy at fixed coverage levels (lowest-confidence
    questions dropped first)."""
    out: dict[str, float] = {}
    ordered = sorted(records, key=lambda r: r["conf"], reverse=True)
    n = len(ordered)
    for cov in coverage_fractions:
        k = max(1, int(round(cov * n)))
        sub = ordered[:k]
        out[f"{cov:.2f}"] = accuracy(sub)
    return out


# ------------------------------------------------------------------ aggregate

def summarize(records: list[dict]) -> dict:
    """Standard per-question summary block for one (mode, Q, seed) cell."""
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "accuracy": accuracy(records),
        "macro_f1": macro_f1(records),
        "ece": ece([r["conf"] for r in records], [r["correct"] for r in records]),
        "brier": brier(records),
        "nll": nll(records),
        "per_type_accuracy": per_type_accuracy(records),
        "score_mae": score_mae(records),
    }
