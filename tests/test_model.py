"""Model unit tests: shapes, masking, parallel heads, calibration heads."""
from __future__ import annotations

import pytest
import torch

from vss.model.questions import QuestionSpec
from vss.model.tokenizer import VSSTokenizer
from vss.model.vss_model import VSSModel


def fit_tokenizer(model: VSSModel, texts: list[str]) -> None:
    tok = VSSTokenizer(
        vocab_size=model.config.vocab_size, hash_buckets=model.config.hash_buckets
    ).fit(texts)
    model.tokenizer = tok
    model.vss_encoder.tokenizer = tok


class TestQuestionSpec:
    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError):
            QuestionSpec.from_dict({"id": "x", "type": "quantum"})

    def test_score_requires_range(self) -> None:
        with pytest.raises(ValueError):
            QuestionSpec.from_dict({"id": "u", "type": "score", "min": 5, "max": 5})


class TestForward:
    def test_single_pass_multi_question(
        self, tiny_model: VSSModel
    ) -> None:
        fit_tokenizer(tiny_model, ["<STATE> message = my card was charged twice <QUESTION>"])
        state = {"message": "my card was charged twice", "customer_age_days": 421}
        questions = [
            {"id": "department", "type": "choice",
             "options": ["billing", "technical", "sales", "other"]},
            {"id": "refund_requested", "type": "noul"},
            {"id": "urgency", "type": "score", "min": 0, "max": 10},
        ]
        out = tiny_model([state], [questions])
        rows = out["rows"]
        assert len(rows) == 3
        types = [r["type"] for r in rows]
        assert types == ["choice", "noul", "score"]
        assert rows[0]["logits"].shape == (4,)
        assert 0.0 <= float(rows[1]["prob"]) <= 1.0
        assert 0.0 <= float(rows[2]["value"]) <= 10.0
        assert 0.0 <= float(rows[0]["calibration"]) <= 1.0

    def test_choice_probs_sum_to_one(self, tiny_model: VSSModel) -> None:
        fit_tokenizer(tiny_model, ["<STATE> x <QUESTION>"])
        out = tiny_model([{"message": "hello"}], [[
            {"id": "d", "type": "choice", "options": ["a", "b", "c"]},
            {"id": "e", "type": "choice", "options": ["a", "b"]},
        ]])
        p3 = torch.softmax(out["rows"][0]["logits"], dim=-1)
        p2 = torch.softmax(out["rows"][1]["logits"], dim=-1)
        assert p3.shape == (3,) and p2.shape == (2,)
        assert torch.isclose(p3.sum(), torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(p2.sum(), torch.tensor(1.0), atol=1e-5)

    def test_batch_rows_aligned(self, tiny_model: VSSModel) -> None:
        fit_tokenizer(tiny_model, ["<STATE> a <QUESTION>"])
        out = tiny_model(
            [{"message": "a"}, {"message": "b"}],
            [
                [{"id": "x", "type": "noul"}],
                [{"id": "x", "type": "noul"}, {"id": "y", "type": "noul"}],
            ],
        )
        assert len(out["per_example_rows"]) == 2
        assert len(out["per_example_rows"][0]) == 1
        assert len(out["per_example_rows"][1]) == 2

    def test_padding_ignored(self, tiny_model: VSSModel) -> None:
        fit_tokenizer(tiny_model, ["<STATE> alpha beta <QUESTION>"])
        # same state, different sibling batch member -> padding must not leak
        out1 = tiny_model([{"message": "alpha beta"}], [[{"id": "n", "type": "noul"}]])
        out2 = tiny_model(
            [{"message": "alpha beta"}, {"message": "alpha beta and much more text here"}],
            [[{"id": "n", "type": "noul"}], [{"id": "n", "type": "noul"}]],
        )
        assert torch.isclose(
            out1["rows"][0]["prob"].detach(),
            out2["rows"][0]["prob"].detach(),
            atol=1e-4,
        )

    def test_sequence_limit_enforced(self, tiny_model: VSSModel) -> None:
        with pytest.raises(ValueError):
            tiny_model.encoder(torch.zeros(1, tiny_model.config.max_sequence_length + 1, dtype=torch.long))
