"""Synthetic decision-data generator.

Deterministic (seeded) generator producing hard examples: negation,
irrelevant context, conflicting signals, typos, ambiguous and OOD states —
per the VSS dataset strategy. Labels are produced by *programmatic rules*,
never by an LLM at generation time (an optional LLM teacher can later
generate additional states, but labels here remain rule-based).

Variation axes [vss]:
  wording, sentence structure, spelling noise, negation, irrelevant
  information, conflicting information, long/short context, ambiguity,
  out-of-domain text.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .schema import TrainingExample

# --------------------------------------------------------------------------
# Domain banks. Templates contain {slot} placeholders filled by bank words.
# Programmatic labelers decide the answer from the *semantic key* used, so
# surface wording can vary freely while labels stay correct.
# --------------------------------------------------------------------------

DEPARTMENT_POSITIVE = {
    # key -> surface phrasings (all semantically map to the department key)
    "billing": [
        "I was charged twice on my {card} card and need a refund adjustment",
        "my invoice shows a duplicate payment from last {month}",
        "the subscription billed me {times} times this {month}",
        "there is an extra {amount} dollar charge on my statement",
    ],
    "technical": [
        "the app crashes every time I open the {feature} screen",
        "I cannot log in, the {feature} page shows error {code}",
        "the {feature} sync has been failing since yesterday",
        "checkout throws a {code} exception when I pay",
    ],
    "sales": [
        "do you offer a discount for annual plans with {feature}",
        "I want to upgrade to the team tier, who can help with pricing",
        "can I get a quote for {times} seats",
        "is there an enterprise plan with a dedicated {feature}",
    ],
    "shipping": [
        "my order with the {feature} has not arrived after {days} days",
        "the package says delivered but I never got the {feature}",
        "I need to change the delivery address of my order",
        "tracking for order {code} has not updated in {days} days",
    ],
}

IRRELEVANT_FILLERS = [
    "by the way I also love your newsletter",
    "my neighbor recommended you in {month}",
    "I have three cats and a dog at home",
    "the weather here has been rainy all week",
    "happy birthday to your support team",
    "I had pasta for lunch today",
]

NEGATIONS = [
    "I do NOT want a refund, just fix the {feature} bug",
    "no refund needed, please do not credit my account",
    "I am not asking for money back",
    "do not refund anything, this is about the {feature}",
]

TYPO_PAIRS = [
    ("refund", "refnud"), ("charged", "chagred"), ("invoice", "invioce"),
    ("error", "eror"), ("please", "plese"), ("account", "acount"),
    ("payment", "paymet"), ("delivery", "delievery"),
]

CONFUSERS = {
    # keyword that naive keyword-matching would get wrong
    "invoice": "other",      # mentions refund-adjacent words but is sales/billing doc request
    "cancel": "sales",       # "cancel my subscription" -> sales retention, not refund
    "password": "technical", # "refund my password"? no - technical
}


def _apply_typos(text: str, rng: random.Random, prob: float = 0.08) -> str:
    out = text
    for good, bad in TYPO_PAIRS:
        if good in out and rng.random() < prob:
            out = out.replace(good, bad, 1)
    return out


def _maybe_filler(rng: random.Random, prob: float = 0.25) -> str:
    if rng.random() < prob:
        return " " + rng.choice(IRRELEVANT_FILLERS)
    return ""


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        card=rng.choice(["credit", "debit", "corporate"]),
        month=rng.choice(["January", "March", "July", "October"]),
        times=rng.choice(["two", "three"]),
        amount=rng.choice(["10", "25", "60"]),
        feature=rng.choice(["dashboard", "export", "mobile app", "checkout"]),
        code=rng.choice(["500", "403", "E12", "ERR_9"]),
        days=rng.choice(["5", "9", "12"]),
    )


class SyntheticGenerator:
    """Seeded generator of VSS TrainingExamples across difficulty stages."""

    STAGES = ("basic", "noul", "score", "multi", "dynamic", "hard")

    def __init__(self, seed: int = 13) -> None:
        self.rng = random.Random(seed)
        self.seed = seed

    # ------------------------------------------------------------ helpers
    def _departments(self) -> list[str]:
        return sorted(DEPARTMENT_POSITIVE.keys())

    def _dept_message(self, dept: str) -> str:
        t = self.rng.choice(DEPARTMENT_POSITIVE[dept])
        msg = _fill(t, self.rng)
        msg = _apply_typos(msg, self.rng)
        msg += _maybe_filler(self.rng)
        return msg

    # ------------------------------------------------------------ stages
    def stage_basic(self, n: int) -> list[TrainingExample]:
        """Stage 1: state -> single choice over the department taxonomy."""
        out = []
        for _ in range(n):
            dept = self.rng.choice(self._departments())
            out.append(
                TrainingExample.model_validate(
                    {
                        "state": {"message": self._dept_message(dept)},
                        "questions": [
                            {
                                "id": "department",
                                "type": "choice",
                                "options": self._departments(),
                                "answer": dept,
                            }
                        ],
                    }
                )
            )
        return out

    def stage_noul(self, n: int) -> list[TrainingExample]:
        """Stage 2: refund_requested noul with hard negatives (negation)."""
        out = []
        for _ in range(n):
            want_refund = self.rng.random() < 0.5
            if want_refund:
                dept = self.rng.choice(["billing", "shipping"])
                msg = self._dept_message(dept) + " Please refund the duplicate charge."
                answer = 1
            else:
                if self.rng.random() < 0.6:
                    # negation trap: refund words present, semantics refuse
                    msg = _fill(self.rng.choice(NEGATIONS), self.rng)
                    msg = _apply_typos(msg, self.rng)
                else:
                    dept = self.rng.choice(["technical", "sales"])
                    msg = self._dept_message(dept)
                answer = 0
            msg += _maybe_filler(self.rng, 0.2)
            out.append(
                TrainingExample.model_validate(
                    {
                        "state": {
                            "message": msg,
                            "customer_age_days": self.rng.randint(1, 900),
                        },
                        "questions": [
                            {"id": "refund_requested", "type": "noul", "answer": answer}
                        ],
                    }
                )
            )
        return out

    def stage_score(self, n: int) -> list[TrainingExample]:
        """Stage 3: urgency score 0..10 from programmatic signals."""
        out = []
        for _ in range(n):
            signals = 0
            parts: list[str] = []
            if self.rng.random() < 0.5:
                parts.append("I cannot access my account at all")
                signals += 2
            if self.rng.random() < 0.5:
                parts.append("this has been broken for days")
                signals += 1
            if self.rng.random() < 0.4:
                parts.append("I am switching to a competitor if this is not fixed")
                signals += 2
            if self.rng.random() < 0.5:
                parts.append(self._dept_message(self.rng.choice(self._departments())))
            if not parts:
                parts.append("quick question about your pricing page")
            urgency = min(10.0, float(signals) + self.rng.uniform(0.0, 1.5))
            out.append(
                TrainingExample.model_validate(
                    {
                        "state": {"message": ". ".join(parts)},
                        "questions": [
                            {
                                "id": "urgency",
                                "type": "score",
                                "min": 0,
                                "max": 10,
                                "answer": round(urgency, 2),
                            }
                        ],
                    }
                )
            )
        return out

    def stage_multi(self, n: int) -> list[TrainingExample]:
        """Stage 4: 2-4 questions per example, mixed types, one forward pass."""
        out = []
        for _ in range(n):
            dept = self.rng.choice(self._departments())
            msg = self._dept_message(dept)
            refund = 1 if ("refund" in msg or "charged twice" in msg) else 0
            urgency = min(10.0, float(2 if refund else 0) + self.rng.uniform(0, 2))
            qs: list[dict[str, Any]] = [
                {
                    "id": "department",
                    "type": "choice",
                    "options": self._departments(),
                    "answer": dept,
                }
            ]
            if self.rng.random() < 0.8:
                qs.append({"id": "refund_requested", "type": "noul", "answer": refund})
            if self.rng.random() < 0.8:
                qs.append(
                    {
                        "id": "urgency",
                        "type": "score",
                        "min": 0,
                        "max": 10,
                        "answer": round(urgency, 2),
                    }
                )
            out.append(
                TrainingExample.model_validate(
                    {"state": {"message": msg, "customer_age_days": self.rng.randint(1, 900)}, "questions": qs}
                )
            )
        return out

    def stage_dynamic(self, n: int) -> list[TrainingExample]:
        """Stage 5: varying option sets — model must respect declared schema."""
        out = []
        for _ in range(n):
            dept = self.rng.choice(self._departments())
            options = self._departments()[:]
            if self.rng.random() < 0.5:
                self.rng.shuffle(options)
            if self.rng.random() < 0.4:
                options = options[: self.rng.randint(2, len(options))]
            if dept not in options:
                options.append(dept)
            out.append(
                TrainingExample.model_validate(
                    {
                        "state": {"message": self._dept_message(dept)},
                        "questions": [
                            {
                                "id": "department",
                                "type": "choice",
                                "options": options,
                                "answer": dept,
                            }
                        ],
                    }
                )
            )
        return out

    def stage_hard(self, n: int) -> list[TrainingExample]:
        """Stage 6: ambiguity, contradiction, long context, OOD, unknown-schema.

        ABSTAIN targets teach calibrated refusal when evidence is absent.
        """
        out = []
        depts = self._departments()
        for _ in range(n):
            kind = self.rng.random()
            if kind < 0.2:
                # OOD: none of the known departments apply -> ABSTAIN target
                msg = self.rng.choice(
                    [
                        "the sky is blue today and my garden grows well",
                        "reminds me of a song lyric about mountains and rivers",
                        "random note: the coffee machine on floor 3 is fixed",
                    ]
                )
                answer: Any = "ABSTAIN"
            elif kind < 0.4:
                # ambiguous: conflicting department signals -> ABSTAIN
                a = self.rng.choice(depts)
                b = self.rng.choice([d for d in depts if d != a])
                msg = self._dept_message(a) + ". Also, " + self._dept_message(b) + "."
                answer = "ABSTAIN"
            elif kind < 0.6:
                # long context: bury the signal in irrelevant chatter
                dept = self.rng.choice(depts)
                pad = " ".join(self.rng.choice(IRRELEVANT_FILLERS) for _ in range(12))
                msg = pad + ". " + self._dept_message(dept) + ". " + pad
                answer = dept
            elif kind < 0.8:
                # contradictory customer fields vs message
                dept = self.rng.choice(depts)
                msg = self._dept_message(dept)
                state = {
                    "message": msg,
                    "prior_department": self.rng.choice(depts),
                    "confidence_note": "conflicting historical routing",
                }
                out.append(
                    TrainingExample.model_validate(
                        {
                            "state": state,
                            "questions": [
                                {
                                    "id": "department",
                                    "type": "choice",
                                    "options": depts,
                                    "answer": dept,
                                }
                            ],
                        }
                    )
                )
                continue
            else:
                # distractor keyword trap: refund vocabulary, non-billing truth
                dept = self.rng.choice(["technical", "sales"])
                msg = self._dept_message(dept)
                if "refund" not in msg.lower():
                    msg += ". (I know some people ask for a refund here, not me.)"
                answer = dept
            out.append(
                TrainingExample.model_validate(
                    {
                        "state": {"message": msg},
                        "questions": [
                            {
                                "id": "department",
                                "type": "choice",
                                "options": depts,
                                "answer": answer,
                            }
                        ],
                    }
                )
            )
        return out

    # ------------------------------------------------------------ all
    def generate(self, counts: dict[str, int] | None = None) -> list[TrainingExample]:
        """Generate examples for requested stages, in curriculum order."""
        counts = counts or {s: 400 for s in self.STAGES}
        gens = {
            "basic": self.stage_basic,
            "noul": self.stage_noul,
            "score": self.stage_score,
            "multi": self.stage_multi,
            "dynamic": self.stage_dynamic,
            "hard": self.stage_hard,
        }
        out: list[TrainingExample] = []
        for stage in self.STAGES:
            n = int(counts.get(stage, 0))
            if n:
                out.extend(gens[stage](n))
        return out
