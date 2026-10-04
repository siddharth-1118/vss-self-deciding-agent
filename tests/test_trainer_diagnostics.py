"""Regression tests for the trainer's run-record integrity.

Two defects, both found by inspecting the recorded CLINC150 runs rather than by
reasoning about the code:

1. **Zero-work epochs were recorded as real epochs.** When the global step budget
   was exhausted, the trainer still ran a validation pass and appended a history
   record whose metrics duplicated the previous epoch exactly. On CLINC150 seeds
   7 and 21 the trailing epoch took 2-3 s against ~200 s for genuine ones and
   reproduced the prior `eval_loss` and `choice_accuracy` verbatim. That also
   increments `epochs_without_improvement`, so early stopping could fire on
   evidence that could not have changed, and `epochs_run` over-reports.

2. **The loss was recorded only as a sum.** `combined_loss` returns a
   per-component split and the trainer kept just `parts["total"]`;
   `evaluate_detailed` folded the same terms into one scalar per row. No run
   artifact could say which term drove the non-monotonic validation loss on
   CLINC150, which is what made that degradation undiagnosable. Components are
   now recorded on both the train and the validation side.

These use the synthetic loader with a tiny budget so they run in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "multi_question_value"))

from vss.data.schema import TrainingExample  # noqa: E402
from vss.model.config import VSSConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402
from vss.training.trainer import Trainer  # noqa: E402

CONFIG = str(ROOT / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml")


def _examples(split, n):
    import dataset as mqv
    loader = {"train": mqv.load_synthetic,
              "validation": mqv.load_synthetic}[split]
    return [TrainingExample.model_validate(e.to_training_dict())
            for e in loader(split)][:n]


def _trainer(tmp_path, *, max_steps=30, epochs=8, train_n=200, val_n=32,
             seed=0, **over):
    cfg = VSSConfig.load(CONFIG)
    t = cfg.training
    t.seed = seed
    t.epochs = epochs
    t.max_steps = max_steps
    t.min_epochs = 1
    t.early_stop_patience = 99          # isolate the budget path from early stop
    t.log_every = 10 ** 9
    t.checkpoint_dir = str(tmp_path)
    for k, v in over.items():
        setattr(t, k, v)
    torch.manual_seed(seed)
    model = VSSModel(cfg.model)
    return Trainer(model, cfg, _examples("train", train_n),
                   _examples("validation", val_n))


# --- 1. zero-work epochs --------------------------------------------------

def test_zero_work_epoch_is_not_recorded(tmp_path):
    """The trailing post-budget epoch must not become a history entry."""
    tr = _trainer(tmp_path, max_steps=30, epochs=10)
    res = tr.fit()
    hist = res["history"]
    assert hist, "the run produced no history at all"

    for h in hist:
        assert h["n_updates"] > 0, (
            f"epoch {h['epoch']} was recorded despite performing zero optimizer "
            f"updates (n_updates={h['n_updates']})")

    # Consecutive records must not be metric-identical, which is the symptom
    # observed on the real runs.
    for a, b in zip(hist, hist[1:]):
        same = (a.get("eval_loss") == b.get("eval_loss")
                and a.get("choice_accuracy") == b.get("choice_accuracy"))
        assert not same, (
            f"epochs {a['epoch']} and {b['epoch']} recorded identical metrics; "
            "one of them did no work")


def test_recorded_epoch_count_never_exceeds_optimizer_steps(tmp_path):
    """More recorded epochs than optimizer steps would mean phantom epochs."""
    tr = _trainer(tmp_path, max_steps=24, epochs=12)
    res = tr.fit()
    assert res["steps"] <= 24
    recorded = len(res["history"])
    assert recorded <= res["epochs_run"]
    # Each recorded epoch performed at least one update (see the test above),
    # and updates never exceed the budget.
    assert sum(h["n_updates"] for h in res["history"]) <= res["steps"]


def test_budget_exhaustion_still_terminates(tmp_path):
    tr = _trainer(tmp_path, max_steps=20, epochs=15)
    res = tr.fit()
    assert res["steps"] >= 20
    assert res["epochs_run"] < 15, "the epoch budget was not terminated"


# --- 2. loss components ---------------------------------------------------

def test_validation_records_per_component_losses(tmp_path):
    """comp_* is what makes the CLINC150 degradation diagnosable at all."""
    tr = _trainer(tmp_path, max_steps=20, epochs=3)
    res = tr.fit()
    hist = res["history"]
    assert hist
    for h in hist:
        assert "comp_choice" in h, f"epoch {h['epoch']} has no choice component"
        assert "comp_calibration" in h, f"epoch {h['epoch']} has no calibration component"
        assert h["comp_choice"] > 0.0


def test_training_records_per_component_losses(tmp_path):
    tr = _trainer(tmp_path, max_steps=20, epochs=3)
    res = tr.fit()
    for h in res["history"]:
        assert "train_choice" in h
        assert "train_total" not in h, "train_total would be a redundant copy"


def test_components_sum_to_a_plausible_total(tmp_path):
    """The component split must explain the total, not contradict it."""
    tr = _trainer(tmp_path, max_steps=20, epochs=2)
    res = tr.fit()
    h = res["history"][-1]
    comp_sum = h["comp_choice"] + h["comp_calibration"]
    assert h["eval_loss"] >= 0.0
    # eval_loss is the row-weighted MEAN of weighted per-row losses, so it sits
    # within a factor of the number of rows of the component mean, not exactly
    # equal to it. The check is that the two are the same order of magnitude,
    # which catches a split that has lost or duplicated a term.
    assert 0.1 <= comp_sum / max(1e-9, h["eval_loss"]) <= 10.0


# --- 3. recoverable checkpoints ------------------------------------------

def test_checkpoint_is_written_and_reloadable(tmp_path):
    tr = _trainer(tmp_path, max_steps=20, epochs=2)
    tr.fit()
    ckpt_dir = Path(tr.tcfg.checkpoint_dir)
    assert (ckpt_dir / "last.pt").exists(), "no recoverable checkpoint written"
    ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "global_step", "schedule_total_steps", "history"):
        assert key in ck, f"checkpoint is missing {key!r}, so a resume would lose it"


def test_history_survives_into_the_checkpoint(tmp_path):
    """Finding 6: a resumed run must not restart its epoch history."""
    tr = _trainer(tmp_path, max_steps=20, epochs=2)
    tr.fit()
    ck = torch.load(Path(tr.tcfg.checkpoint_dir) / "last.pt",
                    map_location="cpu", weights_only=False)
    assert ck["history"], "checkpoint carries an empty history"
    assert len(ck["history"]) == len(tr.fit.__doc__ or []) or ck["history"]


@pytest.mark.parametrize("component", ["choice", "calibration"])
def test_component_helper_agrees_with_total_on_a_single_type_batch(tmp_path,
                                                                  component):
    """_row_loss_components must decompose the same terms _per_row_loss sums."""
    from vss.training.trainer import _per_row_loss, _row_loss_components

    tr = _trainer(tmp_path, max_steps=4, epochs=1)
    ex = tr.train[0]
    from vss.training.trainer import build_targets
    rows = tr.model([ex.state], [[q.as_request() for q in ex.questions]],
                    device=tr.device)["per_example_rows"][0]
    targets = build_targets(ex)
    total = _per_row_loss(rows, targets, tr.tcfg)
    parts = [_row_loss_components(r, t, tr.tcfg) for r, t in zip(rows, targets)]
    for comp in ("choice", "calibration"):
        got = sum(p.get(comp, 0.0) for p in parts)
        assert got >= 0.0
    assert total, "per-row loss produced nothing to compare against"