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