"""Schema-constrained inference engine.

Guarantees [public]:
  - Choice answers contain EXACTLY the declared options (masking happens in
    the head; invalid options are impossible at the API layer).
  - Noul answers are 0/1 with an explicit probability.
  - Score answers lie within [min, max].
  - Abstention (`ABSTAIN`) when confidence < threshold and enabled.
"""
from __future__ import annotations

from typing import Any

import torch

from ..model.questions import QuestionSpec


def _answer_from_row(
    row: dict[str, Any], spec: QuestionSpec, enable_abstention: bool
) -> tuple[Any, float]:
    """Extract (value, base_confidence) from a model output row.

    Base confidence semantics per type [vss]:
      choice -> probability of the selected option (the distribution IS the
      uncertainty measure); noul -> max(p, 1-p); score -> 1.0 placeholder
      because a bin distribution over arbitrary bins is not a class
      probability — score confidence comes from the calibration head's
      P(prediction correct) instead.
    """
    if spec.type == "choice":
        logits = row["logits"].float().detach()
        idx = int(logits.argmax())
        # Confidence includes the trained abstain logit [vss]: the model's
        # "none of these options" mass deflates confidence on out-of-scope
        # inputs, which then triggers threshold abstention. The emitted
        # distribution stays over the DECLARED options only.
        if "abstain_logit" in row:
            full = torch.softmax(
                torch.cat([logits, row["abstain_logit"].float().detach().view(1)])
            , dim=-1)
            abstain_p = float(full[-1])
            if int(full.argmax()) == len(full) - 1:
                # trained abstain class wins: the model says "none of these".
                # Per the output contract, the reported confidence is the
                # best DECLARED option's support — i.e. how weak the evidence
                # for any concrete answer is (mission example: 0.31).
                if enable_abstention:
                    return "ABSTAIN", float(full[:-1].max())
                # forced concrete answer: best option, deflated confidence
                opt_idx = int(logits.argmax())
                return spec.options[opt_idx], float(full[opt_idx])
            return spec.options[idx], float(full[idx])
        probs = torch.softmax(logits, dim=-1)
        return spec.options[idx], float(probs[idx])
    if spec.type == "noul":
        p = float(row["prob"].detach())
        return (1 if p >= 0.5 else 0), max(p, 1 - p)
    if spec.type == "score":
        return float(row["value"].detach()), 1.0
    raise ValueError(f"unsupported type {spec.type}")


def build_answer_rows(
    rows: list[dict[str, Any]],
    specs: list[QuestionSpec],
    abstain_threshold: float,
    enable_abstention: bool,
    confidence_mode: str,
) -> list[dict[str, Any]]:
    """Convert head outputs into typed answer entries, ALIGNED with `rows`.

    Returning a list (not an id-keyed dict) keeps batch inference correct
    when multiple examples reuse the same question ids [vss].
    """
    entries: list[dict[str, Any]] = []
    for row, spec in zip(rows, specs):
        value, base = _answer_from_row(row, spec, enable_abstention)
        calib = float(row["calibration"].detach()) if "calibration" in row else base

        if spec.type == "score":
            # score confidence = P(prediction within tolerance) [vss]
            conf = calib
        elif confidence_mode == "max_prob":
            conf = base
        elif confidence_mode == "calibrated_head":
            conf = calib
        else:  # blend [vss]
            conf = min(1.0, max(0.0, base * calib))

        abstained = enable_abstention and conf < abstain_threshold

        if spec.type == "choice":
            if value == "ABSTAIN":
                # trained abstain class won the argmax [vss]. `confidence`
                # follows the output contract (best declared option's
                # support); `abstain_probability` is the model's P(that
                # abstaining is right), used by calibration metrics.
                abstain_p = float(torch.cat([
                    row["logits"].float().detach(),
                    row["abstain_logit"].float().detach().view(1),
                ]).softmax(-1)[-1])
                entries.append({
                    "value": "ABSTAIN",
                    "confidence": round(conf, 4),
                    "abstain_probability": round(abstain_p, 4),
                })
                continue
            probs = torch.softmax(row["logits"].float().detach(), dim=-1)
            distribution = {opt: float(p) for opt, p in zip(spec.options, probs)}
            s = sum(distribution.values()) or 1.0
            distribution = {k: v / s for k, v in distribution.items()}
            if abstained:
                entries.append({"value": "ABSTAIN", "confidence": round(conf, 4)})
            else:
                entries.append({
                    "value": value,
                    "probabilities": {k: round(v, 6) for k, v in distribution.items()},
                    "confidence": round(conf, 4),
                })
        elif spec.type == "noul":
            p_true = float(row["prob"].detach())
            if abstained:
                entries.append({"value": "ABSTAIN", "confidence": round(conf, 4)})
            else:
                entries.append({
                    "value": value,
                    "probability": round(p_true, 6),
                    "confidence": round(max(p_true, 1 - p_true), 4),
                })
        else:  # score
            if abstained:
                entries.append({"value": "ABSTAIN", "confidence": round(conf, 4)})
            else:
                v = min(spec.max, max(spec.min, float(row["value"].detach())))
                entries.append({"value": round(v, 4), "confidence": round(conf, 4)})
    return entries


def group_answers_by_example(
    entries: list[dict[str, Any]],
    questions_list: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Un-flatten row-aligned entries into per-example {id: entry} dicts."""
    out: list[dict[str, Any]] = []
    idx = 0
    for qs in questions_list:
        per: dict[str, Any] = {}
        for q in qs:
            per[q["id"]] = entries[idx]
            idx += 1
        out.append({"answers": per})
    return out


def decide_once(
    model: Any,
    state: dict[str, Any] | list[Any],
    questions: list[dict[str, Any]],
    *,
    abstain_threshold: float,
    enable_abstention: bool,
    confidence_mode: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """One decision pass: state + questions -> typed answers. No per-question
    model calls — a single forward [public]."""
    from ..model.vss_model import VSSModel

    if not isinstance(model, VSSModel):
        raise TypeError("model must be a VSSModel")
    # Defensive: an empty question list cannot produce a question row set and
    # previously surfaced as an opaque torch tensor-size RuntimeError. Reject it
    # as the client error it is. (DecisionRequest also enforces min_length=1.)
    if not questions:
        raise ValueError("questions must contain at least one question")
    specs = [QuestionSpec.from_dict(q) for q in questions]
    model.eval()
    with torch.no_grad():
        out = model([state], [questions], device=device)
    entries = build_answer_rows(
        out["rows"], specs, abstain_threshold, enable_abstention, confidence_mode
    )
    answers = group_answers_by_example(entries, [questions])[0]["answers"]
    return {"answers": answers}
