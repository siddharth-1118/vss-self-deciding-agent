"""Option-permutation invariance test [vss].

Contract under test: a choice question's predicted logits must depend on the
*set* of options, not their *order*. With header-only serialization
(`header_only_choice: true`) the token sequence contains no option text, so
permuting options must leave the logits bit-identical up to the same
permutation. With legacy serialization (option text embedded in the
<QUESTION> block) the sequence itself changes under permutation, which is
exactly the dilution failure mode measured in the validation pass (full
151-option blocks collapsed accuracy from 52.4% to 8.6% at fixed weights).

The untrained-model tests run in CI; the trained-model test runs only when
the CLINC checkpoint from the validation pass exists locally.
"""
from __future__ import annotations

import dataclasses
import random
from pathlib import Path

import pytest
import torch

from vss.model.config import ModelConfig
from vss.model.tokenizer import VSSTokenizer
from vss.model.vss_model import VSSModel

STATE = {"message": "How do I locate my card?"}
OPTIONS = ["card_arrival", "card_delivery_issue", "exchange_rate",
           "card_payment_wrong_exchange_rate", "cancel_transfer",
           "top_up_failed", "verify_my_identity", "getting_spare_card",
           "-pin_blocked_", "Refund_not_showing_up"]


def _choice_question(options: list[str]) -> dict:
    return {"id": "intent", "type": "choice", "options": list(options)}


def _fit_tokenizer(model: VSSModel) -> None:
    tok = VSSTokenizer(
        vocab_size=model.config.vocab_size, hash_buckets=model.config.hash_buckets
    ).fit(["<STATE> message = how do i locate my card <QUESTION> "
           "choose the best intent from the options"])
    model.tokenizer = tok
    model.vss_encoder.tokenizer = tok


def _choice_logits(model: VSSModel, options: list[str]) -> torch.Tensor:
    out = model([dict(STATE)], [[_choice_question(options)]], device="cpu")
    rows = [r for r in out["rows"] if r["type"] == "choice"]
    assert len(rows) == 1
    logits = rows[0]["logits"]
    assert logits.shape == (len(options),)
    return logits.detach().cpu()


def _permuted_like(values: list[str], seed: int) -> tuple[list[str], list[int]]:
    idx = list(range(len(values)))
    random.Random(seed).shuffle(idx)
    return [values[i] for i in idx], idx


class TestPermutationInvarianceHeaderOnly:
    """header_only_choice=True: logits identical up to the applied permutation."""

    @pytest.fixture(scope="class")
    def ho_model(self, tiny_config: ModelConfig) -> VSSModel:
        cfg = dataclasses.replace(tiny_config, header_only_choice=True)
        torch.manual_seed(0)
        model = VSSModel(cfg)
        _fit_tokenizer(model)
        return model

    def test_logits_invariant_under_permutation(self, ho_model: VSSModel) -> None:
        base = _choice_logits(ho_model, OPTIONS)
        for seed in (1, 2, 3):
            permuted, idx = _permuted_like(OPTIONS, seed)
            got = _choice_logits(ho_model, permuted)
            assert torch.allclose(got, base[idx], atol=1e-5), (
                f"seed={seed}: logits changed under option permutation — "
                "serialization leaks option order into the question vector"
            )

    def test_repeated_calls_deterministic(self, ho_model: VSSModel) -> None:
        a = _choice_logits(ho_model, OPTIONS)
        b = _choice_logits(ho_model, OPTIONS)
        assert torch.equal(a, b)

    def test_argmax_follows_permutation(self, ho_model: VSSModel) -> None:
        base = _choice_logits(ho_model, OPTIONS)
        permuted, idx = _permuted_like(OPTIONS, 5)
        got = _choice_logits(ho_model, permuted)
        assert int(got.argmax()) == idx[int(base.argmax())]


class TestLegacySerializationDilution:
    """header_only_choice=False (pre-fix behavior): permutation changes logits.

    This documents the failure mode; if this test ever fails (i.e. logits
    become invariant), the dilution bug is fixed in the legacy path too.
    """

    @pytest.fixture(scope="class")
    def legacy_model(self, tiny_config: ModelConfig) -> VSSModel:
        cfg = dataclasses.replace(tiny_config, header_only_choice=False)
        torch.manual_seed(0)
        model = VSSModel(cfg)
        _fit_tokenizer(model)
        return model

    def test_permutation_changes_sequence_logits(self, legacy_model: VSSModel) -> None:
        base = _choice_logits(legacy_model, OPTIONS)
        permuted, idx = _permuted_like(OPTIONS, 1)
        got = _choice_logits(legacy_model, permuted)
        assert not torch.allclose(got, base[idx], atol=1e-4), (
            "legacy serialization unexpectedly permutation-invariant; "
            "update the dilution-bug documentation"
        )


@pytest.mark.slow
class TestPermutationInvarianceTrainedModel:
    """Same contract on the trained CLINC150 slot-ho checkpoint (real schema)."""

    CKPT = Path("runs/clinc150-slot-ho/final")

    @pytest.fixture(scope="class")
    def trained(self):
        pytest.importorskip("torch")
        from vss.api import VSS

        return VSS.from_pretrained(str(self.CKPT))

    @pytest.mark.skipif(not Path("runs/clinc150-slot-ho/final/config.yaml").exists(),
                        reason="trained CLINC checkpoint not present")
    def test_trained_logits_invariant(self, trained) -> None:
        model = trained.model
        model.eval()
        options = OPTIONS + ["transfer_not_received_by_recipient",
                             "automatic_top_up", "cash_withdrawal_not_recognised"]
        with torch.no_grad():
            base = _choice_logits(model, options)
            permuted, idx = _permuted_like(options, 11)
            got = _choice_logits(model, permuted)
        assert torch.allclose(got, base[idx], atol=1e-4)
