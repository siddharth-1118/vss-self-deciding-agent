"""Shared checkpoint-selection and early-stopping rule.

Both trainers import this module so the rule is provably identical across the
two architectures. Selection by an *unshared* helper is what allowed the two
systems to drift apart in the first place (`header_only_choice`, D26).

Why this exists
---------------
Selecting the checkpoint by minimum evaluation **loss** was measured to
mis-select, badly and unevenly. On Banking77 at three seeds, comparing the
loss-selected epoch against the best-accuracy epoch in the same run:

    plain  s7 +0.015   s13 +0.015   s21 +0.000     mean  +0.010
    vss    s7 +0.080   s13 +0.000   s21 +0.025     mean  +0.035

VSS lost up to 8 accuracy points to its own selection rule while plain lost at
most 1.5. The cause is structural rather than incidental: VSS's eval loss sums a
choice cross-entropy with a calibration BCE and a soft-ordinal term, so it moves
for reasons that have nothing to do with whether the choice head is right. Its
loss curve is visibly non-monotonic while its accuracy climbs monotonically.
The early-stopping counter inherits the same noise, so runs stop while accuracy
is still improving.

The fix is to select on the metric the system is actually reported on --
validation choice accuracy -- for **both** systems, with loss as the tie-break.

This is validation-based selection only. The test split is never consulted, and
no metric definition changes: the reported number is still choice accuracy on
held-out data, now taken from the checkpoint that validation says is best for it.
The rule is applied identically to each architecture, and `min_delta` keeps the
"is this an improvement?" test symmetric.
"""
from __future__ import annotations

from typing import Literal

SelectionMetric = Literal["accuracy", "loss"]

SELECTION_METRICS: tuple[SelectionMetric, ...] = ("accuracy", "loss")


def _score(metric: SelectionMetric, *, accuracy: float | None,
           loss: float | None) -> float | None:
    """Map a metric to a single number where larger is always better."""
    if metric == "accuracy":
        return None if accuracy is None else float(accuracy)
    return None if loss is None else -float(loss)


def is_improvement(
    metric: SelectionMetric,
    *,
    accuracy: float | None,
    loss: float | None,
    best_accuracy: float | None,
    best_loss: float | None,
    min_delta: float,
) -> bool:
    """True when this epoch beats the incumbent under the selection metric.

    Ties on the primary metric fall back to the other one, so an epoch that ties
    on accuracy but improves on loss is still not treated as progress. This is
    what keeps the early-stopping counter from firing on a tie.
    """
    current = _score(metric, accuracy=accuracy, loss=loss)
    if current is None:
        return False
    incumbent = _score(metric, accuracy=best_accuracy, loss=best_loss)
    if incumbent is None:
        return True

    # min_delta is expressed in the metric's own units; it is subtracted from the
    # larger-is-better score so it means the same thing for both directions.
    if current > incumbent + min_delta:
        return True
    if current < incumbent - min_delta:
        return False

    # Within min_delta on the primary metric: break the tie on the secondary.
    if metric == "accuracy":
        if loss is None or best_loss is None:
            return False
        return -float(loss) > -float(best_loss) + min_delta
    if accuracy is None or best_accuracy is None:
        return False
    return float(accuracy) > float(best_accuracy) + min_delta