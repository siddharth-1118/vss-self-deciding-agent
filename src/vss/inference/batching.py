"""Batched inference utilities.

Padding and batching live here so the model itself stays shape-agnostic.
"""
from __future__ import annotations

from typing import Any

import torch

from ..model.vss_model import VSSModel
from .engine import build_answer_rows, group_answers_by_example


def run_batch(
    model: VSSModel,
    states: list[dict[str, Any] | list[Any]],
    questions_list: list[list[dict[str, Any]]],
    *,
    abstain_threshold: float,
    enable_abstention: bool,
    confidence_mode: str,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """One padded forward per mini-batch -> list of answer dicts."""
    if len(states) != len(questions_list):
        raise ValueError("states and questions_list length mismatch")
    model.eval()
    with torch.no_grad():
        out = model(states, questions_list, device=device)
    from ..model.questions import QuestionSpec

    specs = [QuestionSpec.from_dict(q) for qs in questions_list for q in qs]
    entries = build_answer_rows(
        out["rows"], specs, abstain_threshold, enable_abstention, confidence_mode
    )
    return group_answers_by_example(entries, questions_list)
