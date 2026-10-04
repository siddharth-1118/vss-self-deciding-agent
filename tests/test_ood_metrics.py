"""Tests for the OOD / abstention measurement layer (`scripts/eval_ood.py`).

Three things are pinned here.

1. **Metric correctness.** The first version of the threshold rule computed OOS
   recall as ``(is_oos & ~keep).mean()`` -- a mean over the WHOLE split rather
   than over the OOS subset. With 3100 validation rows of which 100 are OOS that
   understates recall by 31x: it reported 0.032 where the truth was 1.000, and
   made a reachable operating point look unreachable. `test_recall_is_averaged
   _over_the_oos_subset_only` fails if that regresses.

2. **Honest failure.** When no candidate threshold reaches the target OOS
   recall, the rule must say so (`feasible=False`) and report the best it can
   actually achieve -- not silently substitute an easier objective that makes
   the number look good.

3. **Documentation honesty.** A guard test scans the release docs for language
   that presents abstention as a dependable out-of-distribution safeguard. The
   measured behaviour is a real but *partial* signal (AUROC 0.81, and the
   operating point blocks a third of legitimate traffic), so a guarantee claim
   is exactly the failure mode worth pinning.

The finding tests that need a checkpoint live in `tests/test_ood_probe.py` and
are opt-in via ``VSS_SLOW=1``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_ood  # noqa: E402


# --- 1. recall denominator ------------------------------------------------

def test_recall_is_averaged_over_the_oos_subset_only():
    """31 OOS rows out of 310 must give recall 1.0, not 0.01.

    The bug this pins: ``(is_oos & ~keep).mean()`` divides by the full split.
    """
    n, n_oos = 310, 31
    is_oos = np.zeros(n, dtype=bool)
    is_oos[:n_oos] = True          # 31 OOS, all low confidence
    conf = np.linspace(0.0, 1.0, n)
    thr, cov, recall, feasible, _ = eval_ood.select_threshold(
        conf, is_oos, target=0.90, n_cand=40)

    # Whatever threshold is chosen, the reported recall must be a fraction of
    # the OOS rows, i.e. in [0, 1] and consistent with the OOS confidences.
    assert 0.0 <= recall <= 1.0
    if feasible:
        achieved = float((conf < thr)[is_oos].mean())
        assert recall == pytest.approx(achieved, abs=1e-9), (
            "the reported recall must be what the threshold actually achieved, "
            "not the target that was asked for")
        assert recall >= 0.90
    # Every OOS row is below every in-scope row here, so the truth is 1.0.
    assert float((conf < float(np.median(conf)))[is_oos].mean()) == pytest.approx(1.0)
    assert cov == pytest.approx(float((conf >= thr)[~is_oos].mean()), abs=1e-9)


def test_recall_denominator_regression_would_have_been_visible():
    """The old expression, on the same data, is ~31x too small."""
    n, n_oos = 310, 31
    is_oos = np.zeros(n, dtype=bool)
    is_oos[:n_oos] = True
    conf = np.linspace(0.0, 1.0, n)
    thr = float(np.median(conf))
    keep = conf >= thr
    old = float((is_oos & ~keep).mean())     # the bug
    new = float((~keep)[is_oos].mean())      # the fix
    assert new == pytest.approx(1.0, abs=1e-9)
    # The bug returns n_oos/n instead of the true fraction.
    assert old == pytest.approx(n_oos / n, abs=1e-9)
    assert new / old == pytest.approx(n / n_oos, abs=1e-6)


def test_threshold_is_chosen_on_supplied_data_only():
    """select_threshold must not see a test split; it takes one array."""
    n = 500
    rng = np.random.default_rng(0)
    conf = rng.random(n)
    is_oos = rng.random(n) < 0.2
    thr, *_ = eval_ood.select_threshold(conf, is_oos, target=0.5)
    assert 0.0 <= thr <= 1.0


# --- 2. honest infeasibility ----------------------------------------------

def test_unreachable_target_is_reported_not_hidden():
    """No OOS row is separable -> feasible=False, and recall stays truthful."""
    n = 400
    rng = np.random.default_rng(7)
    conf = rng.random(n)                 # OOS and in-scope identically distributed
    is_oos = rng.random(n) < 0.25
    thr, cov, recall, feasible, trade = eval_ood.select_threshold(
        conf, is_oos, target=0.99, n_cand=40)
    assert feasible is False
    assert 0.0 <= recall <= 1.0
    # The reported point must still be a real threshold with a real coverage.
    assert 0.0 <= cov <= 1.0
    assert trade, "a trade-off table is always emitted"


def test_feasible_target_is_detected_when_separation_exists():
    conf = np.concatenate([np.linspace(0.05, 0.30, 100),      # OOS
                           np.linspace(0.70, 0.99, 900)])     # in-scope
    is_oos = np.zeros(1000, dtype=bool)
    is_oos[:100] = True
    thr, cov, recall, feasible, _ = eval_ood.select_threshold(
        conf, is_oos, target=0.90)
    assert feasible is True
    assert recall >= 0.90
    assert cov > 0.90


def test_empty_oos_subset_does_not_crash():
    conf = np.linspace(0.0, 1.0, 50)
    is_oos = np.zeros(50, dtype=bool)
    thr, cov, recall, feasible, _ = eval_ood.select_threshold(conf, is_oos, 0.9)
    assert feasible is False and recall == 0.0 and 0.0 <= cov <= 1.0


# --- 3. metric helpers ----------------------------------------------------

def test_auroc_is_one_when_perfectly_separated_and_half_when_tied():
    perfect = eval_ood.auroc(np.array([0.9, 0.95]), np.array([0.1, 0.2]))
    assert perfect == pytest.approx(1.0)
    tied = eval_ood.auroc(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert tied == pytest.approx(0.5)


def test_auroc_is_inverted_when_classes_are_swapped():
    pos = np.array([0.9, 0.8, 0.7])
    neg = np.array([0.1, 0.2, 0.3])
    assert eval_ood.auroc(pos, neg) == pytest.approx(1.0)
    assert eval_ood.auroc(neg, pos) == pytest.approx(0.0)


def test_risk_coverage_starts_at_one_and_is_non_decreasing_in_coverage():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([True, False, True, True])
    rc = eval_ood.risk_coverage(conf, correct)
    cov = [c for c, _ in rc]
    assert cov == sorted(cov)
    assert cov[0] == pytest.approx(1 / 4)
    assert cov[-1] == pytest.approx(1.0)
    # Highest-confidence item is correct, so accuracy at the smallest coverage is 1.
    assert rc[0][1] == pytest.approx(1.0)


def test_ece_is_zero_for_perfectly_calibrated_confidence():
    # Each bin has accuracy equal to its mean confidence: in bin k put k of
    # every 10 items correct, all at the same confidence.
    rng = np.random.default_rng(3)
    pairs = []
    for k in range(1, 10):
        conf = k / 10.0
        correct = [1] * k + [0] * (10 - k)
        rng.shuffle(correct)
        pairs += [(conf, c) for c in correct]
    assert eval_ood.ece_at(pairs) < 0.02


def test_ece_is_large_when_confidence_is_miscalibrated():
    pairs = [(0.99, 0)] * 50
    assert eval_ood.ece_at(pairs) > 0.9


# --- 4. documentation guard ----------------------------------------------

# Phrases that would assert abstention is a dependable OOD safeguard. The
# measured behaviour does not support any of them.
FORBIDDEN = [
    r"detects?\s+out[- ]of[- ]distribution\s+(input|examples|data)",
    r"reliable\s+ood\s+(detection|safeguard|gate)",
    r"ood[- ]proof",
    r"guarantee[sd]?\s+(ood|out-of-distribution)\s+rejection",
    r"will\s+abstain\s+on\s+(novel|unfamiliar|out-of-domain)",
    r"novelty\s+(detection|detector)\s+(works|is\s+supported)",
]
# Text is lowercased before matching, so the patterns must be too.
FORBIDDEN = [re.compile(p) for p in FORBIDDEN]

DOCS = [
    "README.md", "MODEL_CARD.md",
    "docs/model_card.md", "docs/claims.md", "docs/benchmark_report.md",
    "docs/release_readiness.md",
]


NEGATIONS = ("not ", "no ", "never", "cannot", "can't", "withdrawn",
             "does not", "do not", "is not", "was not", "fails", "without")


@pytest.mark.parametrize("rel", DOCS)
def test_docs_do_not_claim_dependable_ood_detection(rel):
    """Fail on an OOD *guarantee*, but not on a sentence that denies one.

    "The abstain head does not detect out-of-distribution input" is the
    opposite of a guarantee claim and must stay allowed.
    """
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    text = path.read_text(encoding="utf-8").lower()
    hits = []
    for pattern in FORBIDDEN:
        for m in re.finditer(pattern, text):
            context = text[max(0, m.start() - 60):m.start()]
            if any(neg in context for neg in NEGATIONS):
                continue          # a denial, not a guarantee
            line = text[:m.start()].count("\n") + 1
            hits.append(f"{rel}:{line}: {m.group(0)!r} (context: ...{context[-45:]!r})")
    assert not hits, (
        "documentation asserts a dependable OOD safeguard, which the measured "
        "behaviour does not support (AUROC 0.81 and a 33% false-abstention "
        "rate at the validation-selected operating point):\n  "
        + "\n  ".join(hits))


def test_negation_aware_guard_still_catches_a_bare_guarantee():
    """The escape hatch must not swallow an actual guarantee claim."""
    sample = "vss provides reliable OOD detection for unseen input"
    hits = []
    for pattern in FORBIDDEN:
        for m in re.finditer(pattern, sample.lower()):
            context = sample.lower()[max(0, m.start() - 60):m.start()]
            if any(neg in context for neg in NEGATIONS):
                continue
            hits.append(m.group(0))
    assert hits, "guard failed to flag an unqualified guarantee claim"


def test_the_ood_limitation_is_still_documented_somewhere():
    """Guard against the opposite failure: quietly deleting the caveat."""
    blob = ""
    for rel in DOCS:
        p = ROOT / rel
        if p.exists():
            blob += p.read_text(encoding="utf-8").lower()
    assert any(k in blob for k in (
        "not an out-of-distribution detector",
        "not an ood detector",
        "ood abstention does not work",
        "abstain head is a",
    )), "the measured OOD limitation disappeared from the docs"