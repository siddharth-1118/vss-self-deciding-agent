"""Dataset and request schema validation [public: typed question schema].

Every training example and every inference request is validated here.
Invalid examples fail BEFORE training or inference — never silently.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VALID_TYPES = {"choice", "noul", "score"}


class QuestionIn(BaseModel):
    """A typed question (request side)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str
    options: list[str] | None = None
    min: float | None = None
    max: float | None = None

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        if v not in VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(VALID_TYPES)}")
        return v

    @model_validator(mode="after")
    def _check_type_fields(self) -> "QuestionIn":
        if self.type == "choice" and not self.options:
            raise ValueError(f"question {self.id!r}: choice requires options")
        if self.type == "score":
            lo = self.min if self.min is not None else 0.0
            hi = self.max if self.max is not None else 10.0
            if not hi > lo:
                raise ValueError(f"question {self.id!r}: max must exceed min")
        return self

    def resolved_options(self) -> list[str]:
        if self.type == "choice":
            if not self.options:
                raise ValueError(f"question {self.id!r}: choice requires options")
            return self.options
        return []

    def resolved_min_max(self) -> tuple[float, float]:
        if self.type == "score":
            lo = self.min if self.min is not None else 0.0
            hi = self.max if self.max is not None else 10.0
            if not hi > lo:
                raise ValueError(f"question {self.id!r}: max must exceed min")
            return lo, hi
        return 0.0, 1.0


class AnsweredQuestion(BaseModel):
    """A question carrying its ground-truth answer (training data)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str
    options: list[str] | None = None
    min: float | None = None
    max: float | None = None
    answer: float | int | str

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        if v not in VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(VALID_TYPES)}")
        return v

    def as_request(self) -> dict[str, Any]:
        q = {"id": self.id, "type": self.type}
        if self.options is not None:
            q["options"] = self.options
        if self.min is not None:
            q["min"] = self.min
        if self.max is not None:
            q["max"] = self.max
        return q

    def validate_answer(self) -> None:
        """Type-check the answer against the question schema. Raises on error.

        The literal string "ABSTAIN" is a valid choice answer: it marks
        examples where no declared option is defensible [vss].
        """
        if self.type == "choice":
            opts = self.options or []
            if not opts:
                raise ValueError(f"question {self.id!r}: choice requires options")
            if self.answer != "ABSTAIN" and self.answer not in opts:
                raise ValueError(
                    f"question {self.id!r}: answer {self.answer!r} not in options"
                )
        elif self.type == "noul":
            if self.answer not in (0, 1, True, False):
                raise ValueError(f"question {self.id!r}: noul answer must be 0 or 1")
        else:  # score
            lo = self.min if self.min is not None else 0.0
            hi = self.max if self.max is not None else 10.0
            a = float(self.answer)
            if not (lo <= a <= hi):
                raise ValueError(
                    f"question {self.id!r}: score answer {a} outside [{lo}, {hi}]"
                )


class DecisionRequest(BaseModel):
    """Inference request: state + questions."""

    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any] | list[Any]
    # At least one question: a decision model asked zero questions cannot build a
    # question row set, and previously reached the encoder and died with an opaque
    # tensor-size RuntimeError instead of a client-facing validation error.
    questions: list[QuestionIn] = Field(..., min_length=1)


class TrainingExample(BaseModel):
    """One training example: state + answered questions."""

    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any] | list[Any]
    questions: list[AnsweredQuestion]

    def validate_all(self) -> None:
        if not self.questions:
            raise ValueError("example must contain at least one question")
        for q in self.questions:
            q.validate_answer()


def validate_jsonl_file(path: str) -> tuple[int, list[str]]:
    """Validate a JSONL training file. Returns (valid_count, errors).

    Never raises for invalid examples; collects errors so data prep can
    report the full list at once.
    """
    errors: list[str] = []
    valid = 0
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                ex = TrainingExample.model_validate(raw)
                ex.validate_all()
                valid += 1
            except Exception as e:  # collect, do not raise
                errors.append(f"line {lineno}: {e}")
    return valid, errors


def load_jsonl(path: str, strict: bool = True) -> list[TrainingExample]:
    """Load a JSONL training file into validated TrainingExamples."""
    out: list[TrainingExample] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = TrainingExample.model_validate(json.loads(line))
                ex.validate_all()
                out.append(ex)
            except Exception as e:
                if strict:
                    raise ValueError(f"{path} line {lineno}: {e}") from e
    return out


def dump_jsonl(examples: list[TrainingExample], path: str) -> None:
    """Write examples as JSONL (compact, sorted keys)."""
    import json as _json

    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(
                _json.dumps(
                    ex.model_dump(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
