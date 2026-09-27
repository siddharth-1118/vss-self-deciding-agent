"""Unit tests: serialization, tokenizer, schema validation."""
from __future__ import annotations

import pytest

from vss.data.schema import (
    AnsweredQuestion,
    DecisionRequest,
    QuestionIn,
    TrainingExample,
    validate_jsonl_file,
)
from vss.model.serialize import serialize_example, serialize_question, serialize_state
from vss.model.tokenizer import VSSTokenizer, fnv1a


class TestSerialize:
    def test_state_is_deterministic(self) -> None:
        s1 = serialize_state({"b": 1, "a": {"x": [1, 2]}})
        s2 = serialize_state({"a": {"x": [1, 2]}, "b": 1})
        assert s1 == s2
        assert "<STATE>" in s1

    def test_nested_flattening(self) -> None:
        s = serialize_state({"customer": {"age": 32, "country": "IN"}, "message": "hi"})
        assert "customer.age = 32" in s
        assert "customer.country = IN" in s
        assert "message = hi" in s

    def test_arrays_indexed(self) -> None:
        s = serialize_state(["login failed", "account locked"])
        assert "0 = login failed" in s
        assert "1 = account locked" in s

    def test_question_block_fixed_order(self) -> None:
        q = serialize_question({"options": ["a", "b"], "type": "choice", "id": "dept"})
        lines = q.strip().splitlines()
        assert lines[1].startswith("id=")
        assert lines[2].startswith("type=")
        assert lines[3].startswith("options=")

    def test_score_requires_min_max(self) -> None:
        with pytest.raises(ValueError):
            serialize_question({"id": "u", "type": "score"})

    def test_full_example_once_state(self) -> None:
        text = serialize_example(
            {"message": "hi"},
            [
                {"id": "d", "type": "choice", "options": ["a", "b"]},
                {"id": "r", "type": "noul"},
            ],
        )
        assert text.count("<STATE>") == 1
        assert text.count("<QUESTION>") == 2


class TestTokenizer:
    def test_fnv1a_stable(self) -> None:
        assert fnv1a(b"hello") == fnv1a(b"hello")
        assert fnv1a(b"hello") != fnv1a(b"world")

    def test_fit_encode_deterministic(self) -> None:
        tok = VSSTokenizer(vocab_size=100, hash_buckets=64).fit(["alpha beta", "beta gamma"])
        a = tok.encode("alpha beta")
        b = tok.encode("alpha beta")
        assert a == b

    def test_oov_goes_to_hash_buckets(self) -> None:
        tok = VSSTokenizer(vocab_size=100, hash_buckets=64).fit(["known"])
        ids = tok.encode("totallyunknownword")
        base = len(tok.token_to_id)
        assert all(base <= i < base + 64 for i in ids)

    def test_special_tag_tokens(self) -> None:
        tok = VSSTokenizer(vocab_size=100, hash_buckets=64).fit(["<STATE> state"])
        ids = tok.encode("<STATE> hello </STATE>")
        assert tok.token_to_id["<STATE>"] in ids
        assert tok.token_to_id["</STATE>"] in ids
        # tags survive fit even when absent from the corpus
        tok2 = VSSTokenizer(vocab_size=100, hash_buckets=64).fit(["plain words"])
        assert "<QUESTION>" in tok2.token_to_id


class TestSchema:
    def test_choice_requires_options(self) -> None:
        with pytest.raises(Exception):
            QuestionIn(id="d", type="choice")

    def test_answer_validation_choice(self) -> None:
        q = AnsweredQuestion(id="d", type="choice", options=["a", "b"], answer="c")
        with pytest.raises(ValueError):
            q.validate_answer()

    def test_answer_validation_noul(self) -> None:
        q = AnsweredQuestion(id="r", type="noul", answer=2)
        with pytest.raises(ValueError):
            q.validate_answer()

    def test_answer_validation_score_range(self) -> None:
        q = AnsweredQuestion(id="u", type="score", min=0, max=10, answer=11)
        with pytest.raises(ValueError):
            q.validate_answer()

    def test_request_rejects_unknown_fields(self) -> None:
        with pytest.raises(Exception):
            DecisionRequest.model_validate(
                {"state": {}, "questions": [{"id": "x", "type": "noul", "wat": 1}]}
            )

    def test_training_example_rejects_bad_type(self) -> None:
        with pytest.raises(Exception):
            TrainingExample.model_validate(
                {"state": {}, "questions": [{"id": "x", "type": "weird", "answer": 1}]}
            )

    def test_validate_jsonl_file_reports_line_errors(self, tmp_path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text(
            '{"state": {"message": "hi"}, "questions": [{"id": "d", "type": "choice", '
            '"options": ["a"], "answer": "a"}]}\n'
            '{"state": {}, "questions": []}\n',
            encoding="utf-8",
        )
        valid, errors = validate_jsonl_file(str(p))
        assert valid == 1
        assert len(errors) == 1
        assert "line 2" in errors[0]
