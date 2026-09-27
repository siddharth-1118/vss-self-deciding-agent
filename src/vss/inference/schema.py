"""Request/response validation for the inference contract.

The engine never trusts raw dicts: requests go through pydantic
(`DecisionRequest`), and outputs are re-validated against the same schema
before being returned. Invalid values are impossible, not just discouraged.
"""
from __future__ import annotations

from typing import Any

from ..data.schema import DecisionRequest, QuestionIn


def validate_request(raw: dict[str, Any]) -> DecisionRequest:
    """Validate a raw request dict. Raises pydantic ValidationError."""
    return DecisionRequest.model_validate(raw)


def questions_from_request(req: DecisionRequest) -> list[dict[str, Any]]:
    """Convert validated questions into engine-ready dicts."""
    out = []
    for q in req.questions:
        d: dict[str, Any] = {"id": q.id, "type": q.type}
        if q.options is not None:
            d["options"] = q.options
        if q.min is not None:
            d["min"] = q.min
        if q.max is not None:
            d["max"] = q.max
        out.append(d)
    return out


def validate_output(answers: dict[str, Any], req: DecisionRequest) -> None:
    """Re-check the engine's output against the request schema. Raises
    AssertionError if any contract invariant is violated (choice values must
    be declared options; scores within range; noul values in {0,1})."""
    for q in req.questions:
        a = answers.get(q.id)
        assert a is not None, f"missing answer for question {q.id!r}"
        if a.get("value") == "ABSTAIN":
            continue
        if q.type == "choice":
            opts = q.resolved_options()
            assert a["value"] in opts, f"{q.id}: value not in declared options"
            probs = a.get("probabilities", {})
            assert set(probs) == set(opts), f"{q.id}: probability keys != options"
            assert abs(sum(probs.values()) - 1.0) < 1e-3, f"{q.id}: probs must sum to 1"
        elif q.type == "noul":
            assert a["value"] in (0, 1), f"{q.id}: noul value must be 0 or 1"
            lo, hi = q.resolved_min_max()  # score
        else:
            lo, hi = q.resolved_min_max()
            assert lo <= a["value"] <= hi, f"{q.id}: score outside [{lo}, {hi}]"
        assert 0.0 <= a.get("confidence", 0.0) <= 1.0, f"{q.id}: confidence out of range"
