"""Regression tests for checkpoint selection (audit finding: loss mis-selects).

Selecting the best checkpoint by minimum evaluation loss was measured to
mis-select, and unequally between the two architectures. On Banking77 at three
seeds, comparing the loss-selected epoch with the best-accuracy epoch inside the
same run:

    plain  s7 +0.015   s13 +0.015   s21 +0.000     mean  +0.010
    vss    s7 +0.080   s13 +0.000   s21 +0.025     mean  +0.035

VSS lost up to 8 accuracy points to its own selection rule; plain lost at most
1.5. The cause is structural: VSS's eval loss sums choice cross-entropy with a
calibration BCE and a soft-ordinal term, so it moves for reasons unrelated to
whether the choice head is right.

These tests pin the corrected rule. They also pin that both trainers share one
implementation, because the two systems drifting apart in the details is what
produced the `header_only_choice` defect (D26).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vss.training import selection  # noqa: E402


def imp(metric, *, accuracy=None, loss=None, best_accuracy=None, best_loss=None,
        min_delta=0.005):
    return selection.is_improvement(
        metric, accuracy=accuracy, loss=loss, best_accuracy=best_accuracy,
        best_loss=best_loss, min_delta=min_delta)


# --- accuracy selection (the corrected default) ---------------------------

def test_higher_accuracy_wins_even_when_loss_is_worse():
    # The exact shape that cost VSS 8 points on seed 7: accuracy improves,
    # eval loss gets slightly worse, and loss-selection throws the epoch away.
    assert imp("accuracy", accuracy=0.845, loss=1.5638,
               best_accuracy=0.765, best_loss=1.5477)


def test_lower_accuracy_never_wins_despite_lower_loss():
    assert not imp("accuracy", accuracy=0.755, loss=1.20,
                   best_accuracy=0.765, best_loss=1.5477)


def test_first_epoch_always_improves():
    assert imp("accuracy", accuracy=0.10, loss=5.0,
               best_accuracy=None, best_loss=float("inf"))


def test_tie_on_accuracy_is_broken_by_loss():
    # Same accuracy, better loss: still progress, so the early-stop counter
    # does not fire on a tie.
    assert imp("accuracy", accuracy=0.80, loss=1.10,
               best_accuracy=0.80, best_loss=1.20)


def test_tie_on_accuracy_and_worse_loss_is_not_progress():
    assert not imp("accuracy", accuracy=0.80, loss=1.30,
                   best_accuracy=0.80, best_loss=1.20)


def test_accuracy_move_smaller_than_min_delta_falls_through_to_loss():
    # 0.0001 is inside min_delta, so the tie-break decides. Better loss still
    # counts as progress; worse loss does not.
    assert imp("accuracy", accuracy=0.8001, loss=1.19,
               best_accuracy=0.8000, best_loss=1.20)
    assert not imp("accuracy", accuracy=0.8001, loss=1.30,
                   best_accuracy=0.8000, best_loss=1.20)


# --- loss selection (retained, must still behave) -------------------------

def test_loss_selection_still_selects_lower_loss():
    assert imp("loss", accuracy=0.50, loss=1.0,
               best_accuracy=0.80, best_loss=1.5)
    assert not imp("loss", accuracy=0.95, loss=2.0,
                   best_accuracy=0.80, best_loss=1.5)


def test_loss_selection_tie_breaks_on_accuracy():
    assert imp("loss", accuracy=0.85, loss=1.20,
               best_accuracy=0.80, best_loss=1.20)
    assert not imp("loss", accuracy=0.70, loss=1.20,
                   best_accuracy=0.80, best_loss=1.20)


# --- missing metrics must not crash or silently pass ----------------------

def test_missing_primary_metric_is_not_an_improvement():
    assert not imp("accuracy", accuracy=None, loss=1.0,
                   best_accuracy=0.80, best_loss=1.5)
    assert not imp("loss", accuracy=0.9, loss=None,
                   best_accuracy=0.8, best_loss=1.5)


# --- default is the corrected rule, in both trainers ----------------------

def test_default_selection_metric_is_accuracy():
    from vss.model.config import TrainingConfig
    assert TrainingConfig().selection_metric == "accuracy"


def test_both_trainers_share_one_selection_implementation():
    """The two systems must not drift apart in this detail again (cf. D26)."""
    import ast

    sources = {
        "vss": ROOT / "src" / "vss" / "training" / "trainer.py",
        "plain": ROOT / "benchmarks" / "multi_question_value" / "plain_classifier.py",
    }
    for name, path in sources.items():
        text = path.read_text(encoding="utf-8")
        assert "selection.is_improvement" in text, (
            f"{name} trainer does not use the shared selection rule")
        tree = ast.parse(text)
        # The VSS trainer imports relatively ("from . import selection"); the
        # plain trainer imports absolutely. Accept either form -- what matters is
        # that both resolve to vss.training.selection.
        imported = any(
            isinstance(n, ast.ImportFrom)
            and any(a.name == "selection" for a in n.names)
            # "from . import selection" parses with module=None; the plain
            # trainer uses "from vss.training import selection".
            and (n.module or ".") in (".", "selection", "training",
                                      "vss.training", "training.selection",
                                      "vss.training.selection")
            for n in ast.walk(tree))
        assert imported, f"{name} trainer does not import vss.training.selection"


def test_selection_module_documents_why():
    """The rationale must stay in the code, not only in a commit message."""
    doc = (selection.__doc__ or "").lower()
    assert "mis-select" in doc
    assert "calibration" in doc and "ordinal" in doc