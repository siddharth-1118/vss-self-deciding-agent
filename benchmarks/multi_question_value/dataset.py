"""Multi-question value benchmark: datasets.

Three dataset families, all scored PER QUESTION against independently
verified gold answers:

1. `synthetic` — generated states with independent facts; every question's
   answer is computed deterministically from the facts by an answer function
   (never stored prose, never model-generated). Gold is verified at generation
   time and re-verified at load time.

2. `clinc150` — real utterances + the dataset's real 151-label intent task,
   already in VSS question format under data/clinc150/. Used as-is: one real
   choice question per state. For Q>1 the question block is replicated with
   distinct ids (same gold) so that per-question accuracy at Q>1 measures
   cross-question interference on real data, not fabricated new tasks.

3. `banking77` — same conversion, 77 real labels.

Q-scaling rule (single source of truth for both systems): each eval state
carries an ORDERED pool of questions; the Q=q evaluation uses the fixed prefix
pool[:q]. Subsets are nested by construction and identical for mode A, B and
C. No state is duplicated within a Q setting; the eval-state sample is a
seeded, split-identical subsample of the test split.
"""
from __future__ import annotations

import json
import random
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT_DIR = DATA / "multi_question_value"

# --------------------------------------------------------------------- types


@dataclass(frozen=True)
class GoldQuestion:
    """One question + its independently verified gold answer."""

    id: str
    type: str  # choice | noul | score
    options: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None
    answer: str | int | float | None = None
    # serialize choice blocks WITHOUT option text (schema stays intact for
    # masking/gold/metrics).  Mirrors the repository's shipped VSS configs
    # (header_only_choice: true); keeps every system on the same token stream.
    header_only_choice: bool = False

    def as_request(self) -> dict:
        """Request-side dict for VSS/serialization (no answer)."""
        q: dict = {"id": self.id, "type": self.type}
        if self.options is not None:
            q["options"] = list(self.options)
        if self.type == "score":
            q["min"] = self.min
            q["max"] = self.max
        return q

    def validate(self) -> None:
        """Type-check the gold answer against the question schema."""
        if self.type == "choice":
            assert self.options, f"{self.id}: choice requires options"
            assert self.answer in self.options, f"{self.id}: answer not in options"
        elif self.type == "noul":
            assert self.answer in (0, 1, True, False), f"{self.id}: bad noul answer"
        elif self.type == "score":
            assert self.min is not None and self.max is not None, f"{self.id}: no range"
            assert self.min <= float(self.answer) <= self.max, f"{self.id}: out of range"
        else:
            raise AssertionError(f"{self.id}: unknown type {self.type}")


@dataclass
class MultiQuestionExample:
    state: dict
    questions: list[GoldQuestion]

    def to_training_dict(self) -> dict:
        return {
            "state": self.state,
            "questions": [
                {
                    "id": q.id,
                    "type": q.type,
                    **({"options": list(q.options)} if q.options else {}),
                    **({"min": q.min, "max": q.max} if q.type == "score" else {}),
                    "answer": q.answer,
                }
                for q in self.questions
            ],
        }

    def verify_gold(self) -> None:
        for q in self.questions:
            q.validate()


# ------------------------------------------------------------------ synthetic


@dataclass(frozen=True)
class SynthQuestionSpec:
    """Question template: deterministic answer function over state facts."""

    qid: str
    kind: str  # choice | noul | score
    answer_fn: object  # facts dict -> answer
    options: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None
    text: str = ""


DEPARTMENTS = ("billing", "fraud", "technical", "general_support")
AGE_GROUPS = ("minor", "adult", "senior")
ACCT_BUCKETS = ("new", "established", "veteran")
REGIONS = ("americas", "europe", "asia")
TIERS = ("basic", "silver", "gold", "platinum")
_REGION_OF = {"India": "asia", "Japan": "asia", "USA": "americas",
              "Brazil": "americas", "Germany": "europe"}


def sample_facts(rng: random.Random) -> dict:
    """Independent facts about one customer/service request."""
    txn_dup = rng.random() < 0.5
    return {
        "customer_age": rng.randrange(18, 81),
        "account_age_days": rng.randrange(30, 4000),
        "transaction_duplicate": txn_dup,
        "refund_requested": rng.random() < 0.5 if txn_dup else rng.random() < 0.25,
        "urgency": rng.randrange(0, 11),
        "country": rng.choice(list(_REGION_OF)),
        "channel": rng.choice(["mobile", "web", "phone", "branch"]),
        "tier": rng.choice(list(TIERS)),
        "failed_logins": rng.randrange(0, 8),
        "amount_usd": rng.randrange(5, 5000),
    }


def _base_pool() -> list[SynthQuestionSpec]:
    """16 base question templates covering all three types."""
    return [
        SynthQuestionSpec(
            "dept", "choice",
            answer_fn=lambda f: (
                "fraud" if f["failed_logins"] >= 5
                else "billing" if f["transaction_duplicate"] or f["refund_requested"]
                else "technical" if f["channel"] in ("mobile", "web")
                else "general_support"
            ),
            options=DEPARTMENTS,
            text="What department should handle this request?",
        ),
        SynthQuestionSpec(
            "refund", "noul",
            answer_fn=lambda f: int(f["refund_requested"]),
            text="Is a refund requested?",
        ),
        SynthQuestionSpec(
            "urgency", "score",
            answer_fn=lambda f: float(f["urgency"]),
            min=0.0, max=10.0,
            text="How urgent is the request (0-10)?",
        ),
        SynthQuestionSpec(
            "acct_year", "noul",
            answer_fn=lambda f: int(f["account_age_days"] > 365),
            text="Is the account older than one year?",
        ),
        SynthQuestionSpec(
            "dup_txn", "noul",
            answer_fn=lambda f: int(f["transaction_duplicate"]),
            text="Is the transaction a duplicate?",
        ),
        SynthQuestionSpec(
            "age_group", "choice",
            answer_fn=lambda f: (
                "minor" if f["customer_age"] < 18
                else "senior" if f["customer_age"] >= 65 else "adult"
            ),
            options=AGE_GROUPS,
            text="Which age group does the customer belong to?",
        ),
        SynthQuestionSpec(
            "acct_bucket", "choice",
            answer_fn=lambda f: (
                "new" if f["account_age_days"] < 180
                else "established" if f["account_age_days"] < 1825 else "veteran"
            ),
            options=ACCT_BUCKETS,
            text="How established is the account?",
        ),
        SynthQuestionSpec(
            "login_risk", "score",
            answer_fn=lambda f: float(f["failed_logins"]),
            min=0.0, max=7.0,
            text="How many failed logins were observed?",
        ),
        SynthQuestionSpec(
            "high_value", "noul",
            answer_fn=lambda f: int(f["amount_usd"] > 1000),
            text="Is the amount above 1000 USD?",
        ),
        SynthQuestionSpec(
            "region", "choice",
            answer_fn=lambda f: _REGION_OF[f["country"]],
            options=REGIONS,
            text="Which region is the customer from?",
        ),
        SynthQuestionSpec(
            "tier_ord", "score",
            answer_fn=lambda f: float(TIERS.index(f["tier"])),
            min=0.0, max=3.0,
            text="What is the numeric tier (basic=0 to platinum=3)?",
        ),
        SynthQuestionSpec(
            "mobile_web", "noul",
            answer_fn=lambda f: int(f["channel"] in ("mobile", "web")),
            text="Is the channel mobile or web?",
        ),
        SynthQuestionSpec(
            "senior", "noul",
            answer_fn=lambda f: int(f["customer_age"] >= 65),
            text="Is the customer 65 or older?",
        ),
        SynthQuestionSpec(
            "urgent_8", "noul",
            answer_fn=lambda f: int(f["urgency"] >= 8),
            text="Is urgency at least 8?",
        ),
        SynthQuestionSpec(
            "score_aux", "score",
            answer_fn=lambda f: float(
                min(10.0, f["urgency"] + (1 if f["amount_usd"] > 2000 else 0))
            ),
            min=0.0, max=10.0,
            text="Auxiliary risk score: urgency plus one if amount above 2000 USD?",
        ),
        SynthQuestionSpec(
            "senior_high_value", "noul",
            answer_fn=lambda f: int(f["customer_age"] >= 65 and f["amount_usd"] > 1000),
            text="Is the customer a senior with an amount above 1000 USD?",
        ),
    ]


VARIANT_SUFFIXES = ("_b", "_c", "_d")


def build_pool() -> list[SynthQuestionSpec]:
    """Ordered question pool: 16 base + 3 variants each = 64 instances.

    pool[:16] are the distinct base templates (used for training and for
    Q<=16 eval); pool[16:] are textual rephrasings sharing the same answer
    function, so Q in {32, 50} still has a verified gold for every slot.
    The fixed order gives the nested prefix property pool[:q].
    """
    base = _base_pool()
    pool = list(base)
    for suf in VARIANT_SUFFIXES:
        for spec in base:
            pool.append(replace(
                spec,
                qid=spec.qid + suf,
                text=spec.text.replace("?", " (rephrased)?") if suf != "_b"
                else spec.text.replace("?", "? (check)"),
            ))
    return pool


POOL: list[SynthQuestionSpec] = build_pool()
# Training states carry 8 of the 16 base templates, selected by a fixed
# rotation (start = (idx * 8) % 16) so every template appears equally often
# across a split. This is the CPU-budget knob: the plain classifier trains
# per (state, question) row, so rows/epoch = states x 8. Eval states always
# carry the full 64-instance pool, so both systems are evaluated on the
# same question counts including counts never seen in training (Q > 8),
# which is exactly the multi-question generalization the qmask architecture
# claims to provide.
N_TRAIN_QUESTIONS = 8
N_BASE = 16


def _realize(spec: SynthQuestionSpec, facts: dict) -> GoldQuestion:
    return GoldQuestion(
        id=spec.qid,
        type=spec.kind,
        options=spec.options,
        min=spec.min,
        max=spec.max,
        answer=spec.answer_fn(facts),
    )


def make_synthetic_example(rng: random.Random, split: str,
                           idx: int = 0) -> MultiQuestionExample:
    """One synthetic example: training/validation states carry 8 base
    templates chosen by deterministic rotation; test states carry the full
    ordered 64-instance pool."""
    facts = sample_facts(rng)
    if split == "test":
        specs = POOL
    else:
        start = (idx * N_TRAIN_QUESTIONS) % N_BASE
        specs = [POOL[(start + k) % N_BASE] for k in range(N_TRAIN_QUESTIONS)]
    questions = [_realize(s, facts) for s in specs]
    ex = MultiQuestionExample(state=dict(facts), questions=questions)
    ex.verify_gold()
    return ex


def synth_dataset_path(split: str) -> Path:
    return OUT_DIR / f"synthetic_{split}.jsonl"


# Per-split RNG offsets. The splits MUST be disjoint: the original generator
# called random.Random(seed) for every split, so `validation` was literally
# `train[:300]` and `test[:800]` was `train` — i.e. 100% of the validation
# split and 80% of the test split were inside the training split. Offsets are
# large and mutually non-overlapping so the draws cannot collide; the
# invariant is additionally checked by verify_splits_disjoint().
SPLIT_SEED_OFFSET: dict[str, int] = {
    "train": 0,
    "validation": 1_000_003,
    "calibration": 2_000_003,
    "test": 3_000_003,
}
ALL_SPLITS = ("train", "validation", "calibration", "test")


def state_fingerprint(ex: "MultiQuestionExample") -> str:
    """Canonical identity of a state: its full fact dict."""
    return json.dumps(ex.state, sort_keys=True, separators=(",", ":"))


def verify_splits_disjoint(splits: dict[str, list["MultiQuestionExample"]]) -> None:
    """Raise if any state appears in more than one split.

    Guards the audit finding that train/validation/test were nested prefixes of
    one another. Cheap enough to call on every load.
    """
    seen: dict[str, str] = {}
    for name in sorted(splits):
        for ex in splits[name]:
            fp = state_fingerprint(ex)
            if fp in seen:
                raise ValueError(
                    f"synthetic split leakage: state appears in both "
                    f"{seen[fp]!r} and {name!r}"
                )
            seen[fp] = name


def generate_synthetic(n_states: int, seed: int, split: str,
                       force: bool = False) -> list[MultiQuestionExample]:
    """Generate (or reload) the synthetic split; persist as auditable JSONL.

    Regeneration is deterministic in (n_states, seed, split). Existing files
    are re-verified question-by-question against the answer functions.
    """
    path = synth_dataset_path(split)
    if path.exists() and not force:
        return load_synthetic(split)
    if split not in SPLIT_SEED_OFFSET:
        # Unknown/throwaway split names (tests, scratch runs) get a stable
        # name-derived offset so they still cannot alias a real split.
        offset = 7_000_011 + (zlib.crc32(split.encode()) % 1_000_000)
    else:
        offset = SPLIT_SEED_OFFSET[split]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed + offset)
    examples = [make_synthetic_example(rng, split, idx=i) for i in range(n_states)]
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_training_dict(), separators=(",", ":")) + "\n")
    return examples


def load_synthetic(split: str) -> list[MultiQuestionExample]:
    """Load persisted synthetic split AND re-verify every gold answer."""
    path = synth_dataset_path(split)
    out: list[MultiQuestionExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            qs = [
                GoldQuestion(
                    id=q["id"], type=q["type"],
                    options=tuple(q["options"]) if q.get("options") else None,
                    min=q.get("min"), max=q.get("max"), answer=q["answer"],
                )
                for q in raw["questions"]
            ]
            ex = MultiQuestionExample(state=raw["state"], questions=qs)
            ex.verify_gold()
            out.append(ex)
    return out


# ----------------------------------------------------------------- real data


def load_real(name: str, split: str) -> list[MultiQuestionExample]:
    """Convert a real JSONL split (clinc150 | banking77) to multi-question form.

    Rows are used EXACTLY as stored by the repository's data pipeline: the
    real state, the real declared options, the real gold answer (including
    the ABSTAIN rows in CLINC training and 'oos' states in its test split).
    One real question per state; Q-scaling replicates it with distinct ids.
    """
    fname = {"clinc150": "clinc150", "banking77": "banking77"}[name]
    path = DATA / fname / f"{split}.jsonl"
    out: list[MultiQuestionExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            qs = []
            for q in raw["questions"]:
                ex_q = GoldQuestion(
                    id=q["id"], type=q["type"],
                    options=tuple(q["options"]) if q.get("options") else None,
                    min=q.get("min"), max=q.get("max"), answer=q["answer"],
                    header_only_choice=True,
                )
                qs.append(ex_q)
            ex = MultiQuestionExample(state=raw["state"], questions=qs)
            ex.verify_gold()
            out.append(ex)
    return out


# -------------------------------------------------------------- Q-scaling set


def eval_subset(examples: list[MultiQuestionExample], q: int, split_seed: int,
                cap: int) -> list[MultiQuestionExample]:
    """Deterministic, split-identical evaluation set for Q=q.

    - subsample `cap` states with a fixed seed derived from split_seed
      (same states for every Q and every system);
    - take the fixed prefix q questions of each state's ordered pool
      (synthetic: nested by construction; real: replicated question blocks
      with distinct ids and identical gold).
    """
    # NOTE: the state subsample is independent of q on purpose: the SAME
    # states are evaluated at every Q (identical across modes and counts);
    # only the question prefix depends on q.
    rng = random.Random(str(split_seed))
    if len(examples) > cap:
        idx = sorted(rng.sample(range(len(examples)), cap))
        examples = [examples[i] for i in idx]
    out: list[MultiQuestionExample] = []
    for ex in examples:
        qs = ex.questions[:q]
        if q > len(ex.questions):
            # real data: replicate the single real question. Slot 0 KEEPS the
            # canonical trained id (renaming it would push the model OOD on
            # question text and measure id sensitivity, not co-asking);
            # extra copies get unique suffixed ids because the inference
            # engine keys answers by id within a request.
            base = ex.questions
            qs = [
                base[i % len(base)] if i < len(base)
                else replace(base[i % len(base)],
                             id=f"{base[i % len(base)].id}#{i}")
                for i in range(q)
            ]
        out.append(MultiQuestionExample(state=ex.state, questions=qs))
    return out


def plain_pairs(examples: list[MultiQuestionExample]):
    """Flatten to (state, question) training pairs for the plain classifier.

    Both systems see the same question instances: VSS trains on the same
    states/questions packed per example; the plain classifier sees one
    (state, question) row per instance — identical information content.
    """
    for ex in examples:
        for q in ex.questions:
            yield ex.state, q
