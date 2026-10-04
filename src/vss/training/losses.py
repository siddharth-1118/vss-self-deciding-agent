"""Multi-task training losses [public: typed decisions; std: CE/BCE/Huber].

    L = w_choice * L_choice + w_noul * L_noul + w_score * L_score
        + w_ordinal * L_score_ordinal

L_score_ordinal is an auxiliary cross-entropy over the score's ordinal bins
(declared in the prompt's "Support ordinal scoring internally" requirement).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def choice_loss(logits: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    """logits [N, C], target_index [N] long -> CE loss."""
    return F.cross_entropy(logits.float(), target_index)


def noul_loss(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """prob [N] in (0,1), target [N] in {0,1} -> BCE."""
    prob = prob.float().clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(prob, target.float())


def score_losses(
    probs: torch.Tensor,
    centers: torch.Tensor,
    target_value: torch.Tensor,
    lo: float,
    hi: float,
) -> dict[str, torch.Tensor]:
    """Score losses: Huber on the expected value + ordinal CE over bins.

    probs [N, bins]; centers [bins]; target_value [N] float.
    """
    expected = (probs.float() * centers).sum(dim=-1)
    huber = F.huber_loss(expected, target_value.float().reshape(expected.shape), delta=(hi - lo) / 8.0)
    # soft target: place a small triangular kernel around the target bin so
    # neighboring bins share label mass (ordinal smoothness) [vss]
    bin_width = (hi - lo) / (probs.shape[-1] - 1) if probs.shape[-1] > 1 else 1.0
    tgt = target_value.float().reshape(-1).clamp(lo, hi)
    tri = (1.0 - (centers.unsqueeze(0) - tgt.unsqueeze(1)).abs() / (2 * bin_width)).clamp(min=0.0)
    tri = tri / tri.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    log_probs = torch.log(probs.float().clamp(min=1e-9))
    ordinal = -(tri * log_probs).sum(dim=-1).mean()  # soft-target CE [std]
    return {"huber": huber, "ordinal": ordinal}


def correctness_targets(rows: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[float]:
    """Empirical per-row correctness used to train the calibration head [vss].

    choice/noul: 1.0 iff argmax matches the label; score: 1.0 iff the
    expected value is within 10% of the declared range.
    """
    out = []
    for row, tgt in zip(rows, targets):
        t = row["type"]
        if t == "choice":
            logits = torch.cat(
                [row["logits"].detach(), row["abstain_logit"].detach().view(1)]
            )
            pred = int(torch.argmax(logits))
            gold = logits.shape[-1] - 1 if tgt.get("abstain") else tgt["answer_index"]
            out.append(1.0 if pred == gold else 0.0)
        elif t == "noul":
            pred = 1 if float(row["prob"]) >= 0.5 else 0
            out.append(1.0 if pred == int(tgt["answer"]) else 0.0)
        else:
            lo, hi = tgt["min"], tgt["max"]
            tol = 0.1 * (hi - lo)
            ok = abs(float(row["value"]) - float(tgt["answer"])) <= tol
            out.append(1.0 if ok else 0.0)
    return out


def combined_loss(
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    weights: dict[str, float],
    score_ordinal_weight: float = 0.25,
    calibration_weight: float = 1.0,
    calibration_targets: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum weighted per-type losses across all question rows in the batch.

    rows: model output rows (with 'logits' / 'prob' / score dict)
    targets: aligned per-row dicts: {"answer_index" | "answer" | "answer"}.
    """
    device = next((r["calibration"].device for r in rows if "calibration" in r), "cpu")
    parts: dict[str, list[torch.Tensor]] = {
        "choice": [], "noul": [], "score": [], "ordinal": [], "calibration": []
    }
    for row, tgt in zip(rows, targets):
        t = row["type"]
        if t == "choice":
            # One softmax over options + learned abstain class [vss]: normal
            # rows SUPPRESS the abstain logit via CE; ABSTAIN rows select it.
            logits = torch.cat(
                [row["logits"], row["abstain_logit"].view(1)], dim=0
            ).unsqueeze(0)
            if tgt.get("abstain"):
                idx = torch.tensor([logits.shape[-1] - 1], device=device).long()
            else:
                idx = torch.tensor([tgt["answer_index"]], device=device).long()
            parts["choice"].append(choice_loss(logits, idx))
        elif t == "noul":
            y = torch.tensor(tgt["answer"], device=device).float()
            parts["noul"].append(noul_loss(row["prob"].reshape(-1), y.reshape(-1)))
        elif t == "score":
            y = torch.tensor(tgt["answer"], device=device).float()
            lo, hi = tgt["min"], tgt["max"]
            s = score_losses(row["probs"], row["centers"], y, lo, hi)
            parts["score"].append(s["huber"])
            parts["ordinal"].append(s["ordinal"])

    # Calibration head: BCE against empirical correctness. When the caller
    # supplies `calibration_targets`, those come from an external source (an
    # EMA teacher / previous pass) rather than this forward pass -- see D28,
    # where deriving the target from the current pass drove the head to
    # saturation (mean 0.9916, std 0.0134) because the target and the thing
    # being pushed toward it are the same distribution.
    if calibration_weight > 0 and rows:
        if calibration_targets is not None:
            if len(calibration_targets) != len(rows):
                raise ValueError(
                    f"calibration_targets has {len(calibration_targets)} entries "
                    f"but there are {len(rows)} rows"
                )
            corr = list(calibration_targets)
        else:
            corr = correctness_targets(rows, targets)
        calib_probs = torch.stack([r["calibration"] for r in rows]).float().clamp(1e-6, 1 - 1e-6)
        calib_target = torch.tensor(corr, device=device).float()
        parts["calibration"].append(F.binary_cross_entropy(calib_probs, calib_target))

    total = rows[0]["calibration"].new_zeros(())
    report: dict[str, float] = {}
    named = [
        ("choice", weights.get("choice", 1.0)),
        ("noul", weights.get("noul", 1.0)),
        ("score", weights.get("score", 1.0)),
        ("ordinal", score_ordinal_weight),
        ("calibration", weights.get("calibration", calibration_weight)),
    ]
    for key, w in named:
        if parts[key]:
            val = torch.stack(parts[key]).mean()
            total = total + w * val
            report[key] = float(val.detach())
    report["total"] = float(total.detach())
    return total, report
