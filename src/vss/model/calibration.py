"""Calibration metrics and fitting [public: calibration emphasis; std: TS/ECE].

Implements:
  - temperature scaling (fit on a held-out split by NLL)
  - Expected Calibration Error (ECE), Brier score, NLL
  - reliability-diagram data (confidence buckets)

The calibration is applied at the head-logit level; the auxiliary correctness
head complements it (see model/questions.py CalibrationHead).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CalibrationReport:
    ece: float
    brier: float
    nll: float
    accuracy: float
    reliability: list[dict[str, float]]  # [{bucket, conf, acc, count}, ...]


def _bucket_reliability(
    confidences: list[float], correct: list[int], n_buckets: int = 10
) -> list[dict[str, float]]:
    buckets = [0] * n_buckets
    acc_sum = [0.0] * n_buckets
    conf_sum = [0.0] * n_buckets
    for c, ok in zip(confidences, correct):
        b = min(n_buckets - 1, int(c * n_buckets))
        buckets[b] += 1
        acc_sum[b] += ok
        conf_sum[b] += c
    rows = []
    for b in range(n_buckets):
        if buckets[b]:
            rows.append(
                {
                    "bucket": (b + 0.5) / n_buckets,
                    "confidence": conf_sum[b] / buckets[b],
                    "accuracy": acc_sum[b] / buckets[b],
                    "count": float(buckets[b]),
                }
            )
    return rows


def expected_calibration_error(confidences: list[float], correct: list[int], n_buckets: int = 10) -> float:
    rows = _bucket_reliability(confidences, correct, n_buckets)
    total = sum(r["count"] for r in rows) or 1.0
    return sum(r["count"] / total * abs(r["accuracy"] - r["confidence"]) for r in rows)


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    if not probabilities:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes)) / len(probabilities)


def nll(probabilities: list[float], outcomes: list[int]) -> float:
    eps = 1e-12
    if not probabilities:
        return 0.0
    return -sum(math.log(max(p, eps)) if y else math.log(max(1 - p, eps))
                for p, y in zip(probabilities, outcomes)) / len(probabilities)


def binary_metrics_report(
    probabilities: list[float], outcomes: list[int], n_buckets: int = 10
) -> CalibrationReport:
    """Full calibration report for binary-style correctness data."""
    acc = sum(outcomes) / max(1, len(outcomes))
    return CalibrationReport(
        ece=expected_calibration_error(probabilities, outcomes, n_buckets),
        brier=brier_score(probabilities, outcomes),
        nll=nll(probabilities, outcomes),
        accuracy=acc,
        reliability=_bucket_reliability(probabilities, outcomes, n_buckets),
    )


@torch.no_grad()
def collect_choice_confidence(
    model: Any,
    examples: list[Any],
    device: str = "cpu",
) -> tuple[list[float], list[int], dict[int, torch.Tensor], list[int]]:
    """Collect max-softmax confidences + correctness on choice questions.

    Also returns the raw pre-softmax logits per row so temperature scaling
    can be fitted. Returns (confidences, correct, logits_by_row, row_index_of_example_question)
    """
    from ..model.questions import QuestionSpec

    confs: list[float] = []
    correct: list[int] = []
    logits_by_row: dict[int, torch.Tensor] = {}
    rows_meta: list[int] = []
    bs = 32
    row_id = 0
    for start in range(0, len(examples), bs):
        batch = examples[start : start + bs]
        states = [ex.state for ex in batch]
        qs = [[q.as_request() for q in ex.questions] for ex in batch]
        out = model(states, qs, device=device)
        for row_group, ex in zip(out["per_example_rows"], batch):
            for row, q in zip(row_group, ex.questions):
                if q.type != "choice":
                    continue
                logits = row["logits"].float().cpu()
                if "abstain_logit" in row:
                    # match inference semantics: confidence includes abstain mass
                    full = torch.cat([logits, row["abstain_logit"].float().cpu().view(1)])
                    probs = torch.softmax(full, dim=-1)
                    conf, idx = probs.max(dim=-1)
                    n_opts = logits.shape[-1]
                    logits_by_row[row_id] = full  # temperature fits over options+abstain
                else:
                    probs = torch.softmax(logits, dim=-1)
                    conf, idx = probs.max(dim=-1)
                    n_opts = logits.shape[-1]
                    logits_by_row[row_id] = logits
                confs.append(float(conf))
                if q.answer == "ABSTAIN":
                    target_idx = n_opts  # abstain class index
                elif q.answer in (q.options or []):
                    target_idx = q.options.index(q.answer)
                else:
                    target_idx = -1
                correct.append(1 if int(idx) == target_idx else 0)
                rows_meta.append(target_idx)
                row_id += 1
    return confs, correct, logits_by_row, rows_meta


def fit_temperature(
    logits_by_row: dict[int, torch.Tensor],
    targets: list[int],
    max_iter: int = 200,
) -> float:
    """Fit a single temperature on held-out logits by minimizing NLL [std].

    Rows may have different option counts, so the loss is summed per row
    (each row keeps its own logits tensor) with one shared temperature.
    """
    valid = [
        (logits_by_row[i], max(0, targets[i]))
        for i in sorted(logits_by_row)
        if i < len(targets) and targets[i] >= 0
    ]
    if not valid:
        return 1.0
    t = torch.tensor([1.0], requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=max_iter)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.zeros(())
        for logits, y in valid:
            loss = loss + torch.nn.functional.cross_entropy(
                logits.unsqueeze(0) / t.clamp(min=0.05), torch.tensor([y])
            )
        loss = loss / len(valid)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.clamp(min=0.05).item())
