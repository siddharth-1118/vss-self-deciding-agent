"""Release-blocking inference robustness (docs/release_readiness.md Gate D).

Found by probing the public API with malformed / degenerate inputs rather than
by reading the happy-path tests: an empty question list passed pydantic
validation and then crashed inside the encoder with an opaque torch
tensor-size `RuntimeError`. A client sending `{"questions": []}` got a 500-class
failure instead of a validation error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vss import VSS  # noqa: E402
from vss.inference.engine import decide_once  # noqa: E402
from vss.data.schema import QuestionIn  # noqa: E402


@pytest.fixture(scope="module")
def model():
    ckpt = REPO / "runs" / "prototype" / "final"
    if not ckpt.exists():
        pytest.skip("no prototype checkpoint; run the quick-start train command")
    return VSS.from_pretrained(str(ckpt))


CHOICE = [{"id": "d", "type": "choice", "options": ["a", "b"]}]


class TestEmptyQuestionList:
    def test_api_rejects_with_validation_error(self, model):
        with pytest.raises(ValidationError) as ei:
            model.decide(state={"message": "hi"}, questions=[])
        assert "questions" in str(ei.value)

    def test_engine_raises_valueerror_not_runtimeerror(self, model):
        """The engine must not leak a torch shape error to callers."""
        with pytest.raises(ValueError) as ei:
            decide_once(model.model, {"message": "hi"}, [], abstain_threshold=0.5,
                        enable_abstention=False, confidence_mode="max_prob")
        assert "at least one question" in str(ei.value)


class TestMalformedInputsAreClientErrors:
    @pytest.mark.parametrize("qs", [
        [{"id": "d", "type": "bogus", "options": ["a"]}],          # unknown type
        [{"id": "d", "type": "choice"}],                            # no options
        [{"id": "d", "type": "choice", "options": []}],             # empty options
        [{"id": "s", "type": "score", "min": 10, "max": 0}],        # inverted range
        [{"id": "n", "type": "noul", "answer": "yes"}],             # wrong answer type
    ], ids=["unknown-type", "no-options", "empty-options",
            "inverted-score-range", "noul-non-integer"])
    def test_invalid_question_raises_validation_error(self, model, qs):
        with pytest.raises(ValidationError):
            model.decide(state={"message": "hi"}, questions=qs)

    def test_extra_field_forbidden(self, model):
        with pytest.raises(ValidationError):
            model.decide(state={"message": "hi"},
                         questions=[{"id": "d", "type": "choice",
                                     "options": ["a", "b"], "bogus": 1}])


class TestDegenerateButValidInputs:
    """These are legal and must produce a well-formed answer, not crash."""

    def test_empty_state_object(self, model):
        out = model.decide(state={}, questions=CHOICE)
        assert set(out["answers"]) == {"d"}

    def test_single_option_choice(self, model):
        out = model.decide(state={"message": "hi"},
                           questions=[{"id": "d", "type": "choice", "options": ["only"]}])
        assert out["answers"]["d"]["value"] == "only"

    def test_empty_message_string(self, model):
        out = model.decide(state={"message": ""}, questions=CHOICE)
        assert set(out["answers"]) == {"d"}

    def test_long_input_is_truncated_not_crashed(self, model):
        out = model.decide(state={"message": "x" * 50_000}, questions=CHOICE)
        assert set(out["answers"]) == {"d"}


class TestOutputContract:
    def test_probabilities_normalised(self, model):
        out = model.decide(state={"message": "hi"}, questions=CHOICE)
        probs = out["answers"]["d"]["probabilities"]
        assert pytest.approx(sum(probs.values()), abs=1e-5) == 1.0
        assert all(0.0 <= p <= 1.0 for p in probs.values())

    def test_every_question_answered(self, model):
        qs = [
            {"id": "d", "type": "choice", "options": ["a", "b", "c"]},
            {"id": "n", "type": "noul"},
            {"id": "s", "type": "score", "min": 0, "max": 10},
        ]
        out = model.decide(state={"message": "order never arrived"}, questions=qs)
        assert set(out["answers"]) == {"d", "n", "s"}
        assert 0.0 <= out["answers"]["n"]["value"] <= 1.0
        assert 0.0 <= out["answers"]["s"]["value"] <= 10.0

    def test_adding_a_question_does_not_change_others(self, model):
        """Question isolation: an unrelated question must not move an answer."""
        a = model.decide(state={"message": "card charged twice"},
                         questions=[{"id": "d", "type": "choice",
                                     "options": ["billing", "sales"]}])
        b = model.decide(state={"message": "card charged twice"},
                         questions=[{"id": "d", "type": "choice",
                                     "options": ["billing", "sales"]},
                                    {"id": "u", "type": "score", "min": 0, "max": 10}])
        assert a["answers"]["d"]["probabilities"] == pytest.approx(
            b["answers"]["d"]["probabilities"], abs=1e-4)

class TestRestApi:
    """REST surface must answer the same contract as the library (Gate D)."""

    @pytest.fixture(scope="class")
    def client(self, request):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from vss.inference import server

        ckpt = REPO / "runs" / "prototype" / "final"
        if not ckpt.exists():
            pytest.skip("no prototype checkpoint")
        server.load_model(str(ckpt))
        return TestClient(server.app)

    def test_health_reports_a_real_boolean(self, client):
        """Regression: `model_loaded` was coerced to the string "True"."""
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_valid_request_is_200(self, client):
        r = client.post("/v1/decide", json={
            "state": {"message": "card charged twice"},
            "questions": [{"id": "d", "type": "choice",
                           "options": ["billing", "sales"]}]})
        assert r.status_code == 200
        assert set(r.json()["answers"]) == {"d"}

    def test_empty_questions_is_422_not_500(self, client):
        """The crash this suite was written for, at the HTTP boundary."""
        r = client.post("/v1/decide",
                        json={"state": {"message": "hi"}, "questions": []})
        assert r.status_code == 422

    def test_bad_question_type_is_422(self, client):
        r = client.post("/v1/decide", json={
            "state": {"message": "hi"},
            "questions": [{"id": "d", "type": "bogus", "options": ["a"]}]})
        assert r.status_code == 422


# --- score bounds must be rejected at validation, not crash in the encoder ---
#
# The encoder raises `ValueError: score question requires min and max`, but
# QuestionIn used to substitute 0.0 / 10.0 for a missing bound. A request naming
# only `min` therefore passed validation and died mid-inference, which the REST
# layer reports as a 500. Validation now matches the encoder's contract, so the
# caller gets a 422-class error naming the problem.

class TestScoreBoundsRequired:
    @pytest.mark.parametrize("q", [
        {"id": "s", "type": "score", "min": 0},
        {"id": "s", "type": "score", "max": 10},
        {"id": "s", "type": "score"},
        {"id": "s", "type": "score", "min": 10, "max": 0},
    ])
    def test_rejected(self, q):
        with pytest.raises(ValidationError):
            QuestionIn(**q)

    def test_error_names_the_requirement(self):
        with pytest.raises(ValidationError) as exc:
            QuestionIn(id="s", type="score", min=0)
        assert "min and max" in str(exc.value)

    def test_both_bounds_accepted(self):
        q = QuestionIn(id="s", type="score", min=0, max=10)
        assert q.resolved_min_max() == (0.0, 10.0)

    def test_choice_still_needs_options(self):
        with pytest.raises(ValidationError):
            QuestionIn(id="c", type="choice")
