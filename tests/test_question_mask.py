"""Tests for question-masked attention [vss-qmask].

Contract: with `question_masked: true`, a question block's tokens may attend
to the state and to themselves, but never to another question block. The
state region is unaffected. Legacy behavior (mask off) is unchanged.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from vss.model.config import ModelConfig
from vss.model.tokenizer import VSSTokenizer
from vss.model.vss_model import VSSModel

STATE = {"message": "how do i locate my card"}
Q_A = {"id": "a", "type": "choice", "options": ["card_arrival", "exchange_rate"]}
Q_B = {"id": "b", "type": "noul"}
Q_C = {"id": "c", "type": "score", "min": 0, "max": 10}


def _fit_tokenizer(model: VSSModel) -> None:
    tok = VSSTokenizer(
        vocab_size=model.config.vocab_size, hash_buckets=model.config.hash_buckets
    ).fit(["<STATE> message = how do i locate my card <QUESTION> "
           "choose the best intent rate how likely from 0 to 10"])
    model.tokenizer = tok
    model.vss_encoder.tokenizer = tok


def _forward(model: VSSModel, questions: list[dict]):
    model.eval()
    with torch.no_grad():
        return model([dict(STATE)], [questions], device="cpu")


@pytest.fixture()
def qm_model(tiny_config: ModelConfig) -> VSSModel:
    cfg = dataclasses.replace(tiny_config, question_masked=True)
    torch.manual_seed(0)
    m = VSSModel(cfg)
    _fit_tokenizer(m)
    return m


@pytest.fixture()
def plain_model(tiny_config: ModelConfig) -> VSSModel:
    torch.manual_seed(0)
    m = VSSModel(tiny_config)
    _fit_tokenizer(m)
    return m


class TestQuestionMaskPresence:
    def test_mask_changes_hidden_states(self, qm_model: VSSModel, plain_model: VSSModel) -> None:
        qs = [Q_A, Q_B, Q_C]
        h_masked = _forward(qm_model, qs)["hidden"]
        h_plain = _forward(plain_model, qs)["hidden"]
        # question A rows must differ between masked and unmasked encoders
        # (Q_B/Q_C influence removed); state rows also differ since the
        # encoder sees a different attention pattern.
        assert not torch.allclose(h_masked, h_plain, atol=1e-5)

    def test_state_block_unchanged_by_other_questions(self, qm_model: VSSModel) -> None:
        """Key isolation property: a question's rows must not change when other
        questions are co-asked (each question = f(state, itself) under the
        full isolation mask)."""
        solo = _forward(qm_model, [Q_A])["hidden"]
        joint = _forward(qm_model, [Q_A, Q_B, Q_C])["hidden"]
        spans_solo = _forward(qm_model, [Q_A])["spans"][0]
        spans_joint = _forward(qm_model, [Q_A, Q_B, Q_C])["spans"][0]
        s0, e0 = spans_solo[0]
        s1, e1 = spans_joint[0]
        assert e0 - s0 == e1 - s1  # same token count for the same question
        assert torch.allclose(solo[0, s0:e0], joint[0, s1:e1], atol=1e-4), (
            "Q_A rows changed when other questions were co-asked — "
            "question-masked attention is not isolating question blocks"
        )

    def test_legacy_model_unmasked(self, plain_model: VSSModel) -> None:
        solo = _forward(plain_model, [Q_A])["hidden"]
        joint = _forward(plain_model, [Q_A, Q_B])["hidden"]
        spans_solo = _forward(plain_model, [Q_A])["spans"][0][0]
        spans_joint = _forward(plain_model, [Q_A, Q_B])["spans"][0][0]
        s0, e0 = spans_solo
        s1, e1 = spans_joint
        assert not torch.allclose(solo[0, s0:e0], joint[0, s1:e1], atol=1e-4), (
            "legacy encoder became question-invariant; the D3 interference "
            "documentation would no longer be pinned"
        )


class TestMaskConstruction:
    def test_mask_shape_and_finiteness(self, qm_model: VSSModel) -> None:
        enc = qm_model.vss_encoder.encode_batch(
            [dict(STATE)], [[Q_A, Q_B]], device="cpu")
        qm = enc["question_mask"]
        assert qm is not None
        T = enc["token_ids"].shape[1]
        assert qm.shape == (1, 1, T, T)
        assert bool(torch.isfinite(qm).all())

    def test_mask_allows_state_columns(self, qm_model: VSSModel) -> None:
        enc = qm_model.vss_encoder.encode_batch(
            [dict(STATE)], [[Q_A, Q_B]], device="cpu")
        qm = enc["question_mask"][0, 0]
        spans = enc["spans"][0]
        (qa_s, qa_e), (qb_s, qb_e) = spans[0], spans[1]
        # rows of Q_A, columns of Q_B: must be masked (-inf-ish)
        assert (qm[qa_s:qa_e, qb_s:qb_e] < -1e8).all()
        # rows of Q_A, state columns (before first span): must be open
        assert (qm[qa_s:qa_e, : max(0, qa_s - 1)] == 0).all()
        # rows of Q_A, own columns: open
        assert (qm[qa_s:qa_e, qa_s:qa_e] == 0).all()

    def test_no_questions_no_mask(self, qm_model: VSSModel) -> None:
        enc = qm_model.vss_encoder.encode_batch(
            [dict(STATE)], [[]], device="cpu")
        assert enc["question_mask"] is None
