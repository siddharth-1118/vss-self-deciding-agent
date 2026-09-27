"""Typed question definitions, head routing, and parallel decision heads.

Architecture [vss]: every question is encoded as a learned query vector that
attends (through the shared encoder) to the state representation. A small
per-question MLP adapter refines the query before routing to its typed head.
All heads run in ONE forward pass — no per-question model calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuestionSpec:
    """Validated question descriptor from the request schema."""

    id: str
    type: str  # choice | noul | score
    options: list[str] = field(default_factory=list)
    min: float = 0.0
    max: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuestionSpec":
        qtype = d.get("type")
        if qtype not in {"choice", "noul", "score"}:
            raise ValueError(f"question {d.get('id')!r}: unknown type {qtype!r}")
        qid = d.get("id")
        if not isinstance(qid, str) or not qid:
            raise ValueError("question requires a non-empty string 'id'")
        options = [str(o) for o in d.get("options", [])]
        if qtype == "choice" and not options:
            raise ValueError(f"question {qid!r}: choice requires options")
        lo = float(d.get("min", 0.0))
        hi = float(d.get("max", 10.0))
        if qtype == "score" and not hi > lo:
            raise ValueError(f"question {qid!r}: score requires max > min")
        return cls(id=qid, type=qtype, options=options, min=lo, max=hi)


class QuestionAdapter(nn.Module):
    """Per-question refinement MLP over the contextual question vector [vss].

    Also injects learned type embeddings so the adapter knows which head will
    consume its output.
    """

    def __init__(self, dim: int, num_types: int = 3) -> None:
        super().__init__()
        self.type_emb = nn.Embedding(num_types, dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, qvec: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        """qvec: [N, D]; type_idx: [N] long tensor of head-type ids."""
        t = self.type_emb(type_idx)  # [N, D]
        return self.net(torch.cat([qvec, t], dim=-1))


class ChoiceHead(nn.Module):
    """Scores a shared option vocabulary; dynamic masking selects the subset
    of logits corresponding to the question's declared options [vss].

    Options map to vocabulary slots by deterministic hashing so any option
    set works without retraining; an option-embedding table provides semantic
    smoothing for frequently seen options.
    """

    def __init__(self, dim: int, num_slots: int = 1024) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.proj = nn.Linear(dim, num_slots)
        self.option_emb = nn.Embedding(num_slots, dim)
        self.option_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        # learned abstain logit [vss]: trained against ABSTAIN targets so the
        # model can express "none of the declared options" during training;
        # at inference abstention is threshold-driven (inference config).
        self.abstain_proj = nn.Linear(dim, 1)

    def logits_with_abstain(self, qvec: torch.Tensor, options: list[str]) -> torch.Tensor:
        base = self.logits_for_options(qvec, options)
        return torch.cat([base, self.abstain_proj(qvec)], dim=-1)

    def option_slot(self, option: str) -> int:
        from .tokenizer import fnv1a

        return fnv1a(("opt:" + option).encode("utf-8")) % self.num_slots

    def logits_for_options(self, qvec: torch.Tensor, options: list[str]) -> torch.Tensor:
        """Return [B?, num_options] logits for exactly the declared options."""
        slots = torch.tensor([self.option_slot(o) for o in options], device=qvec.device)
        slot_logits = self.proj(qvec)  # [N, num_slots]
        shared = slot_logits[:, slots]  # [N, num_options]
        emb = self.option_emb(slots)  # [num_options, D]
        n = qvec.shape[0]
        pair = torch.cat(
            [qvec.unsqueeze(1).expand(n, len(options), -1), emb.unsqueeze(0).expand(n, -1, -1)],
            dim=-1,
        )
        refined = self.option_mlp(pair).squeeze(-1)  # [N, num_options]
        return shared + refined

    def forward(self, qvec: torch.Tensor, options: list[str]) -> torch.Tensor:
        return self.logits_for_options(qvec, options)


class NoulHead(nn.Module):
    """Binary probabilistic statement head."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, 1)

    def forward(self, qvec: torch.Tensor) -> torch.Tensor:
        """Return probability that the statement is true, shape [N]."""
        return torch.sigmoid(self.proj(qvec)).squeeze(-1)


class ScoreHead(nn.Module):
    """Ordinal-binned score head over a declared [min, max] scale [vss].

    Predicts a discretized distribution over `bins`; the expected value under
    the bin centers gives the continuous score, and cumulative soft-BIN
    probabilities give an ordinal signal during training.
    """

    def __init__(self, dim: int, bins: int = 64) -> None:
        super().__init__()
        self.bins = bins
        self.proj = nn.Linear(dim, bins)

    def forward(self, qvec: torch.Tensor, lo: float, hi: float) -> dict[str, torch.Tensor]:
        logits = self.proj(qvec)  # [N, bins]
        probs = F.softmax(logits, dim=-1)
        centers = torch.linspace(lo, hi, self.bins, device=qvec.device)
        value = (probs * centers).sum(dim=-1)  # expected score [N]
        # confidence = 1 - normalized entropy of the bin distribution [vss]:
        # a peaked distribution -> ~1.0, a uniform one -> ~0.0. Mass-near-mode
        # is structurally too strict with many bins (it almost never exceeds
        # 0.55, forcing constant abstention).
        ent = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1)
        confidence = 1.0 - ent / math.log(self.bins)
        return {
            "logits": logits,
            "probs": probs,
            "value": value,
            "confidence": confidence,
            "centers": centers,
        }


class CalibrationHead(nn.Module):
    """Auxiliary correctness head: P(this answer is correct | inputs) [vss].

    Trained with BCE against whether the sampled/argmax decision matched the
    label. Used as an explicit confidence signal, not just max-softmax.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def forward(self, qvec: torch.Tensor, answer_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([qvec, answer_vec], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class DecisionHeads(nn.Module):
    """Container routing each question to its typed head in one pass."""

    TYPE_INDEX = {"choice": 0, "noul": 1, "score": 2}

    def __init__(self, dim: int, num_option_slots: int, score_bins: int) -> None:
        super().__init__()
        self.adapter = QuestionAdapter(dim)
        self.choice = ChoiceHead(dim, num_option_slots)
        self.noul = NoulHead(dim)
        self.score = ScoreHead(dim, score_bins)
        self.calibration = CalibrationHead(dim)
        # head-specific value embeddings for the calibration head input
        self.value_emb = nn.Linear(dim, dim)

    def forward(self, qvecs: torch.Tensor, specs: list[QuestionSpec]) -> dict[str, Any]:
        """qvecs: [N, D] stacked question vectors; specs aligned by row."""
        type_idx = torch.tensor(
            [self.TYPE_INDEX[s.type] for s in specs], device=qvecs.device, dtype=torch.long
        )
        adapted = self.adapter(qvecs, type_idx)  # [N, D] per-row type conditioning

        out: dict[str, Any] = {"rows": []}
        choice_rows = [i for i, s in enumerate(specs) if s.type == "choice"]
        noul_rows = [i for i, s in enumerate(specs) if s.type == "noul"]
        score_rows = [i for i, s in enumerate(specs) if s.type == "score"]

        choice_out: dict[int, torch.Tensor] = {}
        for i in choice_rows:
            choice_out[i] = self.choice(adapted[i : i + 1], specs[i].options)[0]
        abstain_out: dict[int, torch.Tensor] = {}
        for i in choice_rows:
            abstain_out[i] = self.choice.abstain_proj(adapted[i : i + 1])[0]
        noul_out = self.noul(adapted[noul_rows]) if noul_rows else adapted.new_zeros(0)
        score_out: dict[int, dict[str, torch.Tensor]] = {}
        for i in score_rows:
            score_out[i] = self.score(adapted[i : i + 1], specs[i].min, specs[i].max)

        calib = self.calibration(adapted, self.value_emb(adapted))

        for i, s in enumerate(specs):
            row: dict[str, Any] = {"type": s.type}
            if s.type == "choice":
                row["logits"] = choice_out[i]
                row["abstain_logit"] = abstain_out[i]
            elif s.type == "noul":
                row["prob"] = noul_out[noul_rows.index(i)]
            else:
                row.update(score_out[i])
            row["calibration"] = calib[i]
            out["rows"].append(row)
        return out
