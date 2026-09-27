"""Curriculum training [public: staged training; vss: stage definitions].

Stages follow the mission's curriculum: basic -> noul -> score -> multi ->
dynamic -> hard -> calibration. Each stage is a subset of examples; the
trainer simply trains sequentially over stage datasets (fine-tune style),
which keeps the implementation simple and measurable.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..data.schema import TrainingExample
from ..data.synthetic import SyntheticGenerator

STAGE_ORDER = ("basic", "noul", "score", "multi", "dynamic", "hard")


@dataclass
class CurriculumStage:
    name: str
    examples: list[TrainingExample]


def build_curriculum(
    counts: dict[str, int] | None = None,
    seed: int = 13,
) -> list[CurriculumStage]:
    """Generate stage-ordered synthetic datasets."""
    gen = SyntheticGenerator(seed=seed)
    counts = counts or {s: 400 for s in STAGE_ORDER}
    gens = {
        "basic": gen.stage_basic,
        "noul": gen.stage_noul,
        "score": gen.stage_score,
        "multi": gen.stage_multi,
        "dynamic": gen.stage_dynamic,
        "hard": gen.stage_hard,
    }
    stages = []
    for name in STAGE_ORDER:
        n = int(counts.get(name, 0))
        if n > 0:
            stages.append(CurriculumStage(name=name, examples=gens[name](n)))
    return stages


def flatten_stages(stages: list[CurriculumStage]) -> list[TrainingExample]:
    out: list[TrainingExample] = []
    for s in stages:
        out.extend(s.examples)
    return out
