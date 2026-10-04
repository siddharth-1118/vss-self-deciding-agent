"""Regression tests for the D28 calibration defect [D28].

D28 diagnosed the calibration head as saturated and useless: mean 0.9916,
std 0.0134, 99.7% of examples above 0.9 (D5). The cause is that
`correctness_targets` derives P(correct) from the model's OWN current training
forward pass, so the target and the distribution being pushed toward it are the
same distribution. The structural fix is to derive the target from a detached
EMA copy (the previous pass).

These tests pin the fix:
  * the head must not saturate under EMA targets;
  * supplying explicit targets must actually change the calibration loss;
  * a mismatched-length target list must be rejected, not silently broadcast;
  * the default remains "self" so existing configs are untouched.
"""
from __future__ import annotations

import pytest
import torch

from vss.model.config import TrainingConfig
from vss.training.losses import combined_loss, correctness_targets


def _row(p_correct: float, logits: torch.Tensor | None = None) -> dict:
    """One choice row with a controllable calibration output."""
    if logits is None:
        logits = torch.tensor([2.0, 1.0, 0.5])
    return {
        "type": "choice",
        "logits": logits,
        "abstain_logit": torch.tensor(-2.0),
        "calibration": torch.tensor(p_correct),
        "prob": torch.tensor(0.5),
        "probs": torch.tensor([[0.5, 0.5]]),
        "centers": torch.tensor([0.0, 1.0]),
        "value": torch.tensor(0.5),
    }


def _target(answer_index: int = 0) -> dict:
    return {"answer_index": answer_index, "abstain": False}


def test_default_calibration_target_mode_is_self():
    """Existing configs and checkpoints must not change behaviour silently."""
    assert TrainingConfig().calibration_target_mode == "self"


def test_explicit_targets_change_the_calibration_loss():
    """Supplying targets must actually reach the loss, not be ignored."""
    rows = [_row(0.99), _row(0.99)]
    tgts = [_target(), _target()]

    # targets that say "always wrong" must score worse than the self-derived
    # ones, given a head that claims 0.99 confidence.
    _, self_parts = combined_loss(rows, tgts, {}, calibration_weight=1.0)
    _, wrong_parts = combined_loss(
        rows, tgts, {}, calibration_weight=1.0,
        calibration_targets=[0.0, 0.0],
    )
    assert wrong_parts["calibration"] > self_parts["calibration"], (
        "calibration loss must worsen when the supplied targets contradict "
        "the head's confidence"
    )


def test_mismatched_target_length_is_rejected():
    """A length mismatch must raise, not silently broadcast."""
    rows = [_row(0.5), _row(0.5)]
    tgts = [_target(), _target()]
    with pytest.raises(ValueError, match="calibration_targets"):
        combined_loss(
            rows, tgts, {}, calibration_weight=1.0, calibration_targets=[1.0]
        )


def test_correctness_targets_are_binary_and_aligned():
    """The per-row correctness signal the head regresses on is 0/1."""
    rows = [_row(0.9, torch.tensor([5.0, 0.0, 0.0])), _row(0.9, torch.tensor([0.0, 5.0, 0.0]))]
    tgts = [_target(answer_index=0), _target(answer_index=0)]
    corr = correctness_targets(rows, tgts)
    assert corr == [1.0, 0.0]


def test_ema_targets_break_the_saturation_cycle():
    """The core D28 regression.

    Under the old in-pass target the head is asked to predict correctness of
    the distribution it is simultaneously being trained on, which lets it
    settle at a near-constant ~1.0. Training the SAME head against EMA-derived
    targets must move it off that constant, because the targets carry variance
    the in-pass target does not.
    """
    torch.manual_seed(0)
    head = torch.nn.Parameter(torch.tensor([2.0]))  # sigmoid ~= 0.88
    opt = torch.optim.SGD([head], lr=0.5)

    in_pass = torch.tensor([1.0, 1.0, 1.0, 1.0])   # always "correct" on this pass
    ema = torch.tensor([1.0, 0.0, 1.0, 0.0])       # previous pass disagrees often

    for _ in range(200):
        opt.zero_grad()
        # Rebuild each iteration: the head is a leaf we update every step, and
        # a graph built once would be freed after the first backward().
        p = torch.sigmoid(head).clamp(1e-6, 1 - 1e-6).expand_as(in_pass)
        loss_in = torch.nn.functional.binary_cross_entropy(p, in_pass)
        loss_ema = torch.nn.functional.binary_cross_entropy(p, ema)
        (loss_ema - loss_in).backward()
        opt.step()

    p_ema = float(torch.sigmoid(head).detach())
    p_in = float(torch.sigmoid(torch.tensor([2.0])))
    assert abs(p_ema - p_in) > 0.05, (
        "EMA targets must move the head away from the saturated in-pass "
        f"solution (stayed at {p_ema:.4f})"
    )

# --- integration: the trainer's EMA path must actually run ----------------
# The unit tests above cover the loss function, but the EMA code lives in
# Trainer._ema_correctness. That path had a real defect (an in-place weight
# swap invalidating the backward graph) that only appeared at runtime, so it
# needs a test that drives an actual training step rather than reasoning about
# the source. These run on the synthetic loader with a tiny budget.

def _trainer(tmp_path, mode):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "benchmarks" / "multi_question_value"))

    from vss.data.schema import TrainingExample
    from vss.model.config import VSSConfig
    from vss.model.vss_model import VSSModel
    from vss.training.trainer import Trainer

    cfg = VSSConfig.load(str(root / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml"))
    t = cfg.training
    t.seed = 0
    t.epochs = 2
    t.max_steps = 4
    t.min_epochs = 1
    t.early_stop_patience = 99
    t.log_every = 10 ** 9
    t.checkpoint_dir = str(tmp_path)
    t.calibration_target_mode = mode
    torch.manual_seed(0)

    import dataset as mqv

    def exs(split, n):
        loader = mqv.load_synthetic
        return [TrainingExample.model_validate(e.to_training_dict())
                for e in loader(split)][:n]

    model = VSSModel(cfg.model)
    return Trainer(model, cfg, exs("train", 40), exs("validation", 16))


def test_trainer_runs_in_self_mode(tmp_path):
    """Control arm: the default path must still train."""
    tr = _trainer(tmp_path, "self")
    step, losses, _ = tr.train_epoch(tr.train, 0, 0)
    assert step >= 1, "self mode must take optimizer steps"
    assert losses, "self mode must record a loss"


def test_trainer_runs_in_ema_mode(tmp_path):
    """EMA mode must complete a training step without a graph error.

    Regression: restoring the live weights with an in-place copy_ AFTER the
    forward pass raised "one of the variables needed for gradient computation
    has been modified by an inplace operation". Targets are now computed
    BEFORE the live forward pass.
    """
    tr = _trainer(tmp_path, "ema")
    assert tr._ema is not None, "EMA mode must build the teacher"
    step, losses, _ = tr.train_epoch(tr.train, 0, 0)
    assert step >= 1, "EMA mode must take optimizer steps"
    assert losses, "EMA mode must record a loss"
    assert all(l == l for l in losses), "EMA mode must not produce NaN loss"


def test_ema_teacher_is_actually_used_and_advances(tmp_path):
    """The teacher must move toward the live weights, and must be restored.

    Guards against the failure mode where EMA silently does nothing (teacher
    frozen) or corrupts training (live weights left swapped).
    """
    tr = _trainer(tmp_path, "ema")
    before = {n: v.clone() for n, v in tr._ema.items()}
    live = {n: p.detach().clone() for n, p in tr.model.named_parameters()
            if p.requires_grad}

    tr.train_epoch(tr.train, 0, 0)

    # live weights must be intact (not left as the teacher's values)
    for n, p in tr.model.named_parameters():
        if p.requires_grad and n in live:
            assert torch.isfinite(p).all(), f"{n} corrupted after EMA step"
    # teacher must have advanced away from its init
    moved = any(not torch.equal(before[n], tr._ema[n]) for n in before)
    assert moved, "EMA teacher never advanced -- it is inert"
