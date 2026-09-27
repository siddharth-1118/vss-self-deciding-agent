"""Engine/API tests: output contract, abstention, batching."""
from __future__ import annotations

import pytest
import torch

from vss.inference.engine import build_answer_rows, group_answers_by_example
from vss.model.questions import QuestionSpec
from vss.model.tokenizer import VSSTokenizer


def rows_for(specs: list[QuestionSpec], device: str = "cpu") -> list[dict]:
    torch.manual_seed(3)
    rows = []
    for s in specs:
        if s.type == "choice":
            rows.append({
                "type": "choice",
                "logits": torch.randn(len(s.options)),
                "calibration": torch.tensor(0.9),
            })
        elif s.type == "noul":
            rows.append({
                "type": "noul",
                "prob": torch.tensor(0.93),
                "calibration": torch.tensor(0.9),
            })
        else:
            rows.append({
                "type": "score",
                "value": torch.tensor(7.2),
                "confidence": torch.tensor(0.8),
                "probs": torch.softmax(torch.randn(16), -1),
                "centers": torch.linspace(s.min, s.max, 16),
                "calibration": torch.tensor(0.9),
            })
    return rows


class TestBuildAnswers:
    def test_choice_exact_options(self) -> None:
        specs = [QuestionSpec.from_dict(
            {"id": "d", "type": "choice", "options": ["billing", "technical", "sales"]})]
        entries = build_answer_rows(rows_for(specs), specs, 0.55, False, "blend")
        assert set(entries[0]["probabilities"]) == {"billing", "technical", "sales"}
        assert entries[0]["value"] in {"billing", "technical", "sales"}
        assert abs(sum(entries[0]["probabilities"].values()) - 1.0) < 1e-3

    def test_noul_contract(self) -> None:
        specs = [QuestionSpec.from_dict({"id": "r", "type": "noul"})]
        entries = build_answer_rows(rows_for(specs), specs, 0.55, True, "blend")
        assert entries[0]["value"] in (0, 1)
        assert entries[0]["probability"] == pytest.approx(0.93, abs=1e-3)
        assert entries[0]["confidence"] == pytest.approx(0.93, abs=1e-3)

    def test_score_within_bounds(self) -> None:
        specs = [QuestionSpec.from_dict({"id": "u", "type": "score", "min": 0, "max": 10})]
        entries = build_answer_rows(rows_for(specs), specs, 0.55, True, "blend")
        assert 0.0 <= entries[0]["value"] <= 10.0

    def test_abstention_low_confidence(self) -> None:
        specs = [QuestionSpec.from_dict({"id": "u", "type": "score", "min": 0, "max": 10})]
        rows = rows_for(specs)
        rows[0]["calibration"] = torch.tensor(0.2)
        entries = build_answer_rows(rows, specs, 0.55, True, "blend")
        assert entries[0]["value"] == "ABSTAIN"
        # abstention disabled -> numeric value returned
        entries2 = build_answer_rows(rows, specs, 0.55, False, "blend")
        assert entries2[0]["value"] != "ABSTAIN"

    def test_grouping_preserves_duplicates(self) -> None:
        specs = [QuestionSpec.from_dict({"id": "x", "type": "noul"}),
                 QuestionSpec.from_dict({"id": "x", "type": "noul"})]
        entries = build_answer_rows(rows_for(specs), specs, 0.99, False, "max_prob")
        grouped = group_answers_by_example(entries, [[{"id": "x"}], [{"id": "x"}]])
        assert len(grouped) == 2
        assert grouped[0]["answers"]["x"]["value"] == grouped[1]["answers"]["x"]["value"]


class TestVSSApi:
    def test_decide_end_to_end(self, tiny_model: VSSModel) -> None:
        from vss.api import VSS

        from vss.model.config import InferenceConfig

        tok = VSSTokenizer(vocab_size=tiny_model.config.vocab_size,
                           hash_buckets=tiny_model.config.hash_buckets)
        tok.fit(["<STATE> message = my card was charged twice <QUESTION>"])
        tiny_model.tokenizer = tok
        tiny_model.vss_encoder.tokenizer = tok
        api = VSS(tiny_model, InferenceConfig(enable_abstention=False))
        result = api.decide(
            {"message": "my card was charged twice", "customer_age_days": 421},
            [
                {"id": "department", "type": "choice",
                 "options": ["billing", "technical", "sales", "other"]},
                {"id": "refund", "type": "noul"},
                {"id": "urgency", "type": "score", "min": 0, "max": 10},
            ],
        )
        answers = result["answers"]
        assert set(answers) == {"department", "refund", "urgency"}
        assert answers["department"]["value"] in ["billing", "technical", "sales", "other"]
        assert answers["refund"]["value"] in (0, 1)
        assert 0 <= answers["urgency"]["value"] <= 10

    def test_decide_rejects_bad_request(self, tiny_model: VSSModel) -> None:
        from vss.api import VSS

        from vss.model.config import InferenceConfig

        api = VSS(tiny_model, InferenceConfig())
        with pytest.raises(Exception):
            api.decide({"message": "x"}, [{"id": "bad", "type": "nope"}])

    def test_batch_decide(self, tiny_model: VSSModel) -> None:
        from vss.api import VSS

        from vss.model.config import InferenceConfig

        tok = VSSTokenizer(vocab_size=tiny_model.config.vocab_size,
                           hash_buckets=tiny_model.config.hash_buckets)
        tok.fit(["<STATE> message = hello <QUESTION>"])
        tiny_model.tokenizer = tok
        tiny_model.vss_encoder.tokenizer = tok
        api = VSS(tiny_model, InferenceConfig(enable_abstention=False))
        results = api.decide_batch(
            [{"message": "hello"}, {"message": "hello there"}],
            [[{"id": "r", "type": "noul"}], [{"id": "r", "type": "noul"}]],
        )
        assert len(results) == 2
        assert results[0]["answers"]["r"]["value"] in (0, 1)

    def test_duplicate_ids_in_batch(self, tiny_model: VSSModel) -> None:
        from vss.api import VSS

        from vss.model.config import InferenceConfig

        tok = VSSTokenizer(vocab_size=tiny_model.config.vocab_size,
                           hash_buckets=tiny_model.config.hash_buckets)
        tok.fit(["<STATE> message = hi <QUESTION>"])
        tiny_model.tokenizer = tok
        tiny_model.vss_encoder.tokenizer = tok
        api = VSS(tiny_model, InferenceConfig(enable_abstention=False))
        results = api.decide_batch(
            [{"message": "hi"}, {"message": "hi"}],
            [[{"id": "r", "type": "noul"}], [{"id": "r", "type": "noul"}]],
        )
        assert results[0]["answers"]["r"]["value"] == results[1]["answers"]["r"]["value"]
