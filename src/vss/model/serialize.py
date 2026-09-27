"""Canonical serialization of state and questions.

VSS flattens arbitrary JSON state into sorted, typed `<STATE>` lines, and
emits each question as its own `<QUESTION>` block with a stable field order.
Questions are rendered once per question (never repeated per state token) —
they enter the sequence as sibling blocks and are retrieved by learned
question embeddings at the attention layer. [vss]

Deterministic ordering guarantees identical serialization for identical
input state, which is a requirement for reproducible training and
deterministic inference.
"""
from __future__ import annotations

import json
import math
from typing import Any


def _flatten(obj: Any, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten nested JSON into (dotted.path, scalar-repr) pairs. Deterministic."""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            out.extend(_flatten(obj[key], f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            out.extend(_flatten(item, f"{prefix}.{i}" if prefix else str(i)))
    else:
        out.append((prefix or "value", _scalar_repr(obj)))
    return out


def _scalar_repr(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return repr(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def serialize_state(state: Any) -> str:
    """Render state as canonical `<STATE>` text. Raises on non-JSON roots."""
    if not isinstance(state, (dict, list)):
        raise TypeError("state must be a JSON object or array")
    lines = [f"{k} = {v}" for k, v in _flatten(state)]
    body = "\n".join(lines)
    return f"<STATE>\n{body}\n</STATE>"


def serialize_question(question: dict[str, Any]) -> str:
    """Render one question as its canonical `<QUESTION>` block.

    Field order is fixed (id, type, options, min, max) so identical questions
    always serialize identically regardless of input key order.

    With `header_only_choice=True`, choice blocks omit the option list from
    the SERIALIZED TEXT: the head enforces schema membership via its slot
    mask, so emitting hundreds of option tokens into the sequence only
    dilutes the question's mean-pooled representation [vss — added after the
    full-schema dilution experiment, see benchmarks/clinc150].
    """
    qtype = question.get("type")
    if qtype not in {"choice", "noul", "score"}:
        raise ValueError(f"unknown question type: {qtype!r}")
    fields = [
        f"id={question.get('id')}",
        f"type={qtype}",
    ]
    if qtype == "choice":
        options = question.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError("choice question requires a non-empty options list")
        if not question.get("header_only_choice"):
            fields.append("options=[" + ",".join(str(o) for o in options) + "]")
    if qtype == "score":
        if "min" not in question or "max" not in question:
            raise ValueError("score question requires min and max")
        fields.append(f"min={_scalar_repr(question['min'])}")
        fields.append(f"max={_scalar_repr(question['max'])}")
    body = "\n".join(fields)
    return f"<QUESTION>\n{body}\n</QUESTION>"


def serialize_example(state: Any, questions: list[dict[str, Any]]) -> str:
    """Full input text for one decision request (state once, then questions)."""
    parts = [serialize_state(state)]
    parts.extend(serialize_question(q) for q in questions)
    return "\n".join(parts)


def dumps_compact(obj: Any) -> str:
    """Stable compact JSON (sorted keys) — used in dataset fingerprints."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
