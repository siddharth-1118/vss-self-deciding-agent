"""CLI contract tests: argument handling and malformed-input behaviour.

Found by running the documented quick-start command. `--state` accepted inline
JSON while `--questions` accepted only a path to a file, neither had help text
saying so, and passing inline JSON to `--questions` produced a raw

    OSError: [Errno 22] Invalid argument: '[{"id":"department",...}'

traceback from `Path.read_text` instead of an explanation. A user mistake is
not a crash: these tests pin that it produces one clear line on stderr and exit
code 2.

The model is never loaded here -- every assertion is about what happens before
that, and the success path stubs `VSS.from_pretrained` so the suite needs no
checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vss import cli  # noqa: E402

STATE = '{"message": "my invoice shows a duplicate payment"}'
QUESTIONS = ('[{"id": "department", "type": "choice", '
             '"options": ["billing", "sales", "shipping", "technical"]}]')


def _run(argv, capsys):
    """Invoke cli.main with argv; return (exit_code, stdout, stderr)."""
    old = sys.argv
    sys.argv = ["vss", *argv]
    try:
        code = cli.main()
    finally:
        sys.argv = old
    cap = capsys.readouterr()
    return code, cap.out, cap.err


class _StubModel:
    @classmethod
    def from_pretrained(cls, path):
        obj = cls()
        obj.seen = None
        return obj

    def decide(self, state, questions):
        self.seen = (state, questions)
        return {"answers": {"department": {"value": "billing", "confidence": 0.99}}}


@pytest.fixture
def stub_model(monkeypatch):
    import vss.api
    monkeypatch.setattr(vss.api, "VSS", _StubModel)
    return _StubModel


# --- accepted input forms -------------------------------------------------

def test_inline_json_for_both_arguments(stub_model, capsys):
    code, out, err = _run(
        ["decide", "--model", "fake", "--state", STATE, "--questions", QUESTIONS],
        capsys)
    assert code == 0, err
    assert json.loads(out)["answers"]["department"]["value"] == "billing"


def test_questions_from_a_file_path(stub_model, capsys, tmp_path):
    qf = tmp_path / "questions.json"
    qf.write_text(QUESTIONS, encoding="utf-8")
    code, out, err = _run(
        ["decide", "--model", "fake", "--state", STATE, "--questions", str(qf)],
        capsys)
    assert code == 0, err
    assert json.loads(out)["answers"]["department"]["value"] == "billing"


def test_both_arguments_from_files(stub_model, capsys, tmp_path):
    sf, qf = tmp_path / "state.json", tmp_path / "q.json"
    sf.write_text(STATE, encoding="utf-8")
    qf.write_text(QUESTIONS, encoding="utf-8")
    code, out, err = _run(
        ["decide", "--model", "fake", "--state-file", str(sf),
         "--questions", str(qf)], capsys)
    assert code == 0, err
    assert "answers" in json.loads(out)


def test_inline_and_file_forms_produce_identical_calls(stub_model, capsys,
                                                       tmp_path):
    """The two accepted spellings must not change what the model receives."""
    qf = tmp_path / "q.json"
    sf = tmp_path / "s.json"
    qf.write_text(QUESTIONS, encoding="utf-8")
    sf.write_text(STATE, encoding="utf-8")

    seen = []
    original = _StubModel.decide

    def spy(self, state, questions):
        seen.append((state, questions))
        return original(self, state, questions)

    _StubModel.decide = spy
    try:
        _run(["decide", "--model", "fake", "--state", STATE,
              "--questions", QUESTIONS], capsys)
        _run(["decide", "--model", "fake", "--state-file", str(sf),
              "--questions", str(qf)], capsys)
    finally:
        _StubModel.decide = original

    assert len(seen) == 2
    assert seen[0] == seen[1], (
        f"inline and file forms disagree:\n{seen[0]}\n{seen[1]}")


# --- malformed input ------------------------------------------------------

def _expect_clean_error(argv, capsys, fragment):
    code, out, err = _run(argv, capsys)
    assert code == 2, f"expected exit 2, got {code}: {err!r}"
    assert err.strip(), "an error must be printed to stderr"
    assert "Traceback" not in err, f"user error produced a traceback:\n{err}"
    assert fragment in err, f"expected {fragment!r} in:\n{err}"
    return err


def test_malformed_inline_json_is_a_clean_error(capsys):
    _expect_clean_error(
        ["decide", "--model", "fake", "--state", '{"message":',
         "--questions", QUESTIONS], capsys, "invalid inline JSON")


def test_nonexistent_path_is_a_clean_error(capsys):
    err = _expect_clean_error(
        ["decide", "--model", "fake", "--state", STATE, "--questions", "nope.json"],
        capsys, "neither inline JSON")
    assert "nope.json" in err


def test_questions_must_be_an_array(capsys):
    _expect_clean_error(
        ["decide", "--model", "fake", "--state", STATE, "--questions", '{"a": 1}'],
        capsys, "must be a JSON array")


def test_missing_state_is_a_clean_error(capsys):
    _expect_clean_error(
        ["decide", "--model", "fake", "--questions", QUESTIONS],
        capsys, "--state")


def test_invalid_json_file_reports_the_path(capsys, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    err = _expect_clean_error(
        ["decide", "--model", "fake", "--state", STATE, "--questions", str(bad)],
        capsys, "not valid JSON")
    assert "bad.json" in err


def test_help_text_documents_both_accepted_forms(capsys):
    """`vss decide --help` must say inline JSON is allowed.

    The traceback happened because the accepted form was undocumented, so the
    help string is part of the fix.
    """
    with pytest.raises(SystemExit):
        sys.argv = ["vss", "decide", "--help"]
        cli.main()
    out = capsys.readouterr().out.lower()
    assert "--state" in out and "--questions" in out
    assert "inline json" in out, (
        "`--help` does not mention inline JSON, the form that used to crash")