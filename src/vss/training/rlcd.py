"""VSS-RLCD: reinforcement learning for calibrated decisions.

EXPERIMENTAL research track [vss]. Inspired by the public RLCD concept
(reinforcement from contrastive calibration feedback); this is NOT a
reproduction of any proprietary implementation.

Objective: after SFT, optimize a shaped reward

    R = alpha*correctness + beta*calibration_quality
        + gamma*abstention_quality - delta*overconfidence

on top of the supervised model, and compare against SFT-only and
SFT + temperature scaling. The harness keeps whichever method wins on
held-out calibration metrics — the simpler method wins ties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from ..data.schema import TrainingExample
from ..model.vss_model import VSSModel


@dataclass
class RLCDConfig:
    alpha_correct: float = 1.0
    beta_calibration: float = 0.5
    gamma_abstain: float = 0.3
    delta_overconf: float = 0.5
    abstain_threshold: float = 0.55
    lr: float = 1e-4
    epochs: int = 1
    batch_size: int = 32


def row_reward(
    row: dict[str, Any],
    target: dict[str, Any],
    cfg: RLCDConfig,
) -> dict[str, float]:
    """Compute the shaped reward components for one question row.

    R = alpha*correctness + beta*calibration_quality
        + gamma*abstention_quality - delta*overconfidence
    """
    t = row["type"]
    out = {"correct": 0.0, "calibration": 0.0, "abstain": 0.0, "overconf": 0.0}

    if t == "choice":
        logits = torch.cat(
            [row["logits"].detach(), row["abstain_logit"].detach().view(1)]
        )
        p = torch.softmax(logits, -1)
        pred = int(p.argmax())
        conf = float(p[pred])
        gold = logits.shape[-1] - 1 if target.get("abstain") else target["answer_index"]
        correct = pred == gold
    elif t == "noul":
        p = float(row["prob"].detach())
        pred = 1 if p >= 0.5 else 0
        conf = max(p, 1 - p)
        correct = pred == int(target["answer"])
    else:  # score
        value = float(row["value"].detach())
        lo, hi = target["min"], target["max"]
        tol = 0.1 * (hi - lo)
        conf = float(row["calibration"].detach())
        correct = abs(value - float(target["answer"])) <= tol

    correct_f = 1.0 if correct else 0.0
    out["correct"] = correct_f
    calib = float(row["calibration"].detach())
    # calibration quality: stated P(correct) vs empirical outcome [vss]
    out["calibration"] = 1.0 - abs(calib - correct_f)
    # abstention quality: abstaining on what would have been wrong is good;
    # abstaining on what would have been right is neutral (no bonus)
    abstained = conf < cfg.abstain_threshold
    out["abstain"] = 1.0 if (abstained and not correct) else 0.0
    # overconfidence: confident AND wrong
    out["overconf"] = max(0.0, conf - correct_f)

    out["total"] = (
        cfg.alpha_correct * out["correct"]
        + cfg.beta_calibration * out["calibration"]
        + cfg.gamma_abstain * out["abstain"]
        - cfg.delta_overconf * out["overconf"]
    )
    return out


def rlcd_step(
    model: VSSModel,
    batch: list[TrainingExample],
    cfg: RLCDConfig,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """One policy-gradient-style step: REINFORCE on the shaped reward.

    The per-row reward R is treated as a weight on the supervised log-prob of
    the model's own sampled/argmax decision (self-critical baseline = 0),
    reinforcing decisions that were correct AND well calibrated, penalizing
    confident wrong answers.
    """
    from .losses import build_targets

    model.train()
    states = [ex.state for ex in batch]
    qs = [[q.as_request() for q in ex.questions] for ex in batch]
    out = model(states, qs, device=device)
    rows_flat = [r for g in out["per_example_rows"] for r in g]
    targets_flat = [t for ex in batch for t in build_targets(ex)]

    losses: list[torch.Tensor] = []
    stats = {"reward": 0.0, "correct": 0.0, "overconf": 0.0, "n": 0.0}
    for row, tgt in zip(rows_flat, targets_flat):
        r = row_reward(row, tgt, cfg)
        stats["reward"] += r["total"]
        stats["correct"] += r["correct"]
        stats["overconf"] += r["overconf"]
        stats["n"] += 1

        t = row["type"]
        if t == "choice":
            logits = torch.cat([row["logits"], row["abstain_logit"].view(1)])
            logp = torch.log_softmax(logits.float(), dim=-1)
            idx = logits.shape[-1] - 1 if tgt.get("abstain") else tgt["answer_index"]
            losses.append(-r["total"] * logp[idx])
        elif t == "noul":
            prob = row["prob"].float().clamp(1e-6, 1 - 1e-6)
            y = int(tgt["answer"])
            logp = torch.log(prob if y else 1 - prob)
            losses.append(-r["total"] * logp)
        else:
            # score: reinforce calibrated expected value via Huber weighted by reward
            value = row["value"].float()
            huber = F.huber_loss(
                value, torch.tensor(float(tgt["answer"]), device=value.device),
                delta=(tgt["max"] - tgt["min"]) / 8.0, reduction="none",
            )
            losses.append(r["total"] * huber)

    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    n = max(1.0, stats["n"])
    return {k: stats[k] / n for k in ("reward", "correct", "overconf")}


def run_rlcd(
    model: VSSModel,
    train: list[TrainingExample],
    eval_examples: list[TrainingExample],
    cfg: RLCDConfig,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fine-tune with the shaped reward and report SFT vs SFT+RLCD metrics."""
    from ..model.calibration import binary_metrics_report

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    for _ in range(cfg.epochs):
        for start in range(0, len(train), cfg.batch_size):
            rlcd_step(model, train[start : start + cfg.batch_size], cfg, device, optimizer)

    confs, correct, _, _ = collect_confidences(model, eval_examples, device)
    report = binary_metrics_report(confs, correct)
    return {
        "ece": report.ece,
        "brier": report.brier,
        "accuracy": report.accuracy,
        "note": "compare against SFT-only eval; keep the simpler method on ties",
    }


def collect_confidences(
    model: VSSModel, examples: list[TrainingExample], device: str
) -> tuple[list[float], list[int], dict[int, torch.Tensor], list[int]]:
    from ..model.calibration import collect_choice_confidence

    return collect_choice_confidence(model, examples, device)
