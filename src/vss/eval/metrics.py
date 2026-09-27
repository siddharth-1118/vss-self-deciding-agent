"""Evaluation metrics [public: accuracy, probability quality, performance].

No external ML libraries: metric implementations are self-contained and
unit-tested (accuracy, macro/micro-F1, AUROC, AUPRC, latency percentiles).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


def accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(preds)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def f1_scores(preds: list[str], golds: list[str]) -> dict[str, float]:
    """Return {"macro": ..., "micro": ...} plus per-class F1."""
    labels = sorted(set(golds) | set(preds))
    per_class: dict[str, float] = {}
    tp_total = fp_total = fn_total = 0
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        per_class[lab] = _prf(tp, fp, fn)[2]
    micro = _prf(tp_total, fp_total, fn_total)[2]
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    out = {"macro": macro, "micro": micro}
    out.update(per_class)
    return out


def _rank_stats(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """AUROC and AUPRC via rank statistics / step integration."""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan"), float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks: dict[int, int] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    auc = (sum(ranks[i] for i, l in enumerate(labels) if l) - pos * (pos + 1) / 2) / (pos * neg)
    # AUPRC by descending-score step integration
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    auprc = 0.0
    prev_recall = 0.0
    for s, l in pairs:
        if l:
            tp += 1
            recall = tp / pos
            prec = tp / (tp + fp)
            auprc += (recall - prev_recall) * prec
            prev_recall = recall
        else:
            fp += 1
    return auc, auprc


def roc_pr(scores: list[float], labels: list[int]) -> dict[str, float]:
    auroc, auprc = _rank_stats(scores, labels)
    return {"auroc": auroc, "auprc": auprc}


@dataclass
class LatencyReport:
    counts: dict[int, dict[str, float]] = field(default_factory=dict)

    def add(self, n_questions: int, latencies_ms: list[float]) -> None:
        lat = sorted(latencies_ms)
        n = len(lat)
        self.counts[n_questions] = {
            "p50": lat[n // 2],
            "p95": lat[min(n - 1, int(n * 0.95))],
            "p99": lat[min(n - 1, int(n * 0.99))],
            "mean": sum(lat) / n,
            "rps": 1000.0 / (sum(lat) / n) if n else 0.0,
        }

    def table(self) -> str:
        lines = ["questions | p50(ms) | p95(ms) | p99(ms) | req/s"]
        for n in sorted(self.counts):
            c = self.counts[n]
            lines.append(
                f"{n:>9} | {c['p50']:>7.2f} | {c['p95']:>7.2f} | {c['p99']:>7.2f} | {c['rps']:>5.1f}"
            )
        return "\n".join(lines)


def time_decide(fn: Any, repeat: int = 30, warmup: int = 3) -> list[float]:
    """Time fn() in ms; returns latency list (warmup excluded)."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def resource_usage() -> dict[str, float]:
    """RAM (RSS MB) and, when available, torch GPU memory [std]."""
    import os

    rss = 0.0
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = float(line.split()[1]) / 1024.0
        except Exception:
            rss = 0.0
    gpu = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.max_memory_allocated() / (1024**2)
    except Exception:
        pass
    return {"ram_mb": rss, "gpu_mb": gpu, "pid": float(os.getpid())}
