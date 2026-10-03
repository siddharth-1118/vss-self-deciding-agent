"""Regression tests for the OOD-abstention probe (Blocker D).

Two separate concerns are tested here.

1. The probe's own logic -- deterministic, and actually constructing the
   contrasts it claims to construct. A probe that silently failed to destroy
   lexical content would "prove" OOD detection works.

2. The finding itself. The quick-start checkpoint does NOT detect
   out-of-distribution input: the abstain rate is ~0.103 in-distribution,
   ~0.106 word-scrambled and 0.000 foreign-topic. Destroying every lexical
   token changed nothing, so the head behaves like a fixed prior.

The second group needs a trained checkpoint and is therefore opt-in via
VSS_SLOW=1. It asserts the *measured deficiency*, so if a future change makes
OOD detection work the suite tells us to update the limitation docs rather than
silently keeping a stale caveat.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "convergence"))

import ood_probe  # noqa: E402


# --- 1. probe logic -------------------------------------------------------

def test_scramble_preserves_word_count_and_is_deterministic():
    msg = "my invoice shows a duplicate payment from last month"
    out = ood_probe.scramble(msg, random.Random(ood_probe.SCRAMBLE_SEED))
    assert len(out.split()) == len(msg.split())
    # Same seed -> same output (the probe must be reproducible).
    assert out == ood_probe.scramble(msg, random.Random(ood_probe.SCRAMBLE_SEED))
    # Different seed eventually differs; the set of words is preserved.
    assert set(out.split()) == set(msg.split())


def test_scramble_actually_destroys_local_order():
    msg = "a b c d e f g h"
    out = ood_probe.scramble(msg, random.Random(1))
    assert out != msg


class _Q:
    def __init__(self, options):
        self.type = "choice"
        self.options = options
        self.id = "department"


class _Row:
    def __init__(self, message, options):
        self.state = {"message": message}
        self.questions = [_Q(options)]


def test_build_sets_produces_the_three_contrasts():
    rows = [_Row(f"message number {i} about a payment", ["billing", "sales"])
            for i in range(5)]
    sets = ood_probe.build_sets(rows, random.Random(0))

    assert set(sets) == {"in_distribution", "word_scramble", "foreign_topic"}
    # in-distribution and scrambled come from the eval split, 1:1.
    assert len(sets["in_distribution"]) == len(sets["word_scramble"]) == 5
    assert len(sets["foreign_topic"]) == len(ood_probe.FOREIGN)


def test_in_distribution_set_is_untouched_but_scramble_is_not():
    rows = [_Row("invoice duplicate payment july", ["billing", "sales"])]
    sets = ood_probe.build_sets(rows, random.Random(0))
    orig = rows[0].state["message"]
    assert sets["in_distribution"][0]["state"]["message"] == orig
    assert sets["word_scramble"][0]["state"]["message"] != orig


def test_build_sets_ignores_non_choice_rows():
    row = _Row("hello", ["a"])
    row.questions[0].type = "noul"
    sets = ood_probe.build_sets([row], random.Random(0))
    assert sets["in_distribution"] == []


def test_options_helper_falls_back_when_no_choice_question():
    row = _Row("hello", ["a"])
    row.questions[0].type = "noul"
    assert ood_probe._options(row) == ["billing", "sales", "shipping", "technical"]


def test_score_reports_zero_for_empty_set():
    class _M:
        pass
    assert ood_probe.score(_M(), []) == {"n": 0}


# --- 2. the finding itself (opt-in; needs a checkpoint) --------------------

@pytest.mark.skipif(not os.environ.get("VSS_SLOW"),
                    reason="needs a trained checkpoint; set VSS_SLOW=1")
def test_abstain_head_is_not_an_ood_detector():
    """Pin the measured deficiency documented in docs/model_card.md.

    If this ever fails, OOD abstention started working and the limitation
    sections in the model card, benchmark report and claims ledger are stale.
    """
    from vss import VSS
    from vss.data.schema import load_jsonl

    model_dir = Path(os.environ.get("VSS_OOD_MODEL", "runs/smoke_verify/final"))
    if not model_dir.exists():
        pytest.skip(f"checkpoint not present: {model_dir}")

    rows = load_jsonl(os.environ.get("VSS_OOD_EVAL", "data/generated/eval.jsonl"))
    model = VSS.from_pretrained(str(model_dir))
    report = {name: ood_probe.score(model, rs)
              for name, rs in ood_probe.build_sets(
                  rows, random.Random(ood_probe.SCRAMBLE_SEED)).items()}

    ind = report["in_distribution"]["abstain_rate"]
    scr = report["word_scramble"]["abstain_rate"]
    foreign = report["foreign_topic"]["abstain_rate"]

    # Destroying lexical content must not move abstention materially.
    assert abs(scr - ind) < 0.05, (
        f"scramble changed abstain rate {ind} -> {scr}; OOD behaviour changed")
    # Fluent off-domain text is answered confidently rather than flagged.
    assert foreign < 0.5, (
        f"foreign abstain rate {foreign} >= 0.5; update the limitation docs")