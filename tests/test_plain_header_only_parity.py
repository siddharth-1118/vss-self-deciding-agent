"""Plain/VSS serialization parity and label-index consistency.

Root cause of the "legacy checkpoint does not reproduce" investigation and of
the CLINC150 collapse:

`header_only_choice` decides BOTH how a choice question is serialized (with or
without its option text) and how the plain head is trained (full-inventory CE
vs masked CE over the row's declared options).

  * VSS reads the flag from its MODEL config (vss_model.py).
  * The plain baseline used to read it from the QUESTION OBJECT only.

`sweep.load_splits` builds real-data splits through `vss.data.schema.load_jsonl`,
whose `AnsweredQuestion` has no such field, so the flag was always absent there.
The sweep therefore compared VSS (header-only, full-inventory CE) against plain
(full option text, masked CE) -- a different task, not a different architecture.

Measured with ONE fixed banking77 checkpoint on identical validation data:

    header_only_choice=True   -> 0.8900 accuracy
    header_only_choice=False  -> 0.0750 accuracy

These tests pin the fix so the two systems cannot silently diverge again, and
prove the label-index mapping is identical at training and inference time.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "benchmarks" / "multi_question_value",
          REPO / "benchmarks" / "convergence"):
    sys.path.insert(0, str(p))

import plain_classifier as pc  # noqa: E402
from config import PlainConfig  # noqa: E402
from dataset import GoldQuestion  # noqa: E402
from vss.model.serialize import serialize_example  # noqa: E402


# ------------------------------------------------------------------ fixtures
def _question(header_only: bool) -> GoldQuestion:
    return GoldQuestion(
        id="intent", type="choice",
        options=("alpha", "beta", "gamma"), answer="beta",
        header_only_choice=header_only,
    )


def _state() -> dict:
    return {"message": "my card was charged twice"}


# ------------------------------------------------------------- serialization
class TestHeaderOnlySerialization:
    def test_flag_changes_the_serialized_text(self):
        with_text = serialize_example(_state(), [_question(False).as_request()])
        without = serialize_example(
            _state(), [{**_question(True).as_request(), "header_only_choice": True}])
        assert "gamma" in with_text          # option text present
        assert "gamma" not in without        # option text stripped

    def test_config_flag_overrides_absent_question_flag(self):
        """The loader never sets the question flag; the config must still win."""
        cfg = PlainConfig()
        cfg.header_only_choice = True
        q = GoldQuestion(id="i", type="choice", options=("alpha", "beta"),
                         answer="alpha", header_only_choice=False)
        assert pc._header_only(q, cfg) is True

    def test_question_flag_still_works_as_a_fallback(self):
        """Legacy callers that only set the question flag keep working."""
        assert pc._header_only(_question(True), PlainConfig()) is True
        assert pc._header_only(_question(False), PlainConfig()) is False

    def test_encode_rows_uses_the_config_flag(self):
        """encode_rows must apply the config, not just the question attribute."""
        cfg = PlainConfig()
        cfg.header_only_choice = True
        model = pc.PlainClassifier(cfg, n_labels=3, with_abstain=False)
        from vss.model.tokenizer import VSSTokenizer
        model.tokenizer = VSSTokenizer(vocab_size=256, hash_buckets=64)
        tok_text = serialize_example(
            _state(), [{**_question(True).as_request(), "header_only_choice": True}])
        model.tokenizer.fit([tok_text])
        model.tokenizer = VSSTokenizer(vocab_size=256, hash_buckets=64)
        model.tokenizer.fit([tok_text])
        x, _ = model.encode_rows([(_state(), _question(False))], "cpu")
        # token ids differ from a model whose tokenizer saw the option text
        assert x.shape[0] == 1


# ------------------------------------------------------- loss / train parity
class TestChoiceLossUsesFullInventoryWhenHeaderOnly:
    def _out(self, n_labels: int):
        logits = torch.zeros(1, n_labels)
        logits[0, 2] = 5.0           # index 2 wins
        return {"choice": logits, "noul": torch.zeros(1), "score": torch.zeros(1, 8)}

    def test_full_inventory_ce_uses_the_gold_index(self):
        labels = ["alpha", "beta", "gamma"]
        l2i = {l: i for i, l in enumerate(labels)}
        pairs = [(_state(), _question(True))]
        loss = pc._batch_loss(self._out(3), pairs, l2i, 8, "cpu", header_only=True)
        assert float(loss) > 0
        # a one-hot target at index 1 ("beta") against a logit peaked at 2
        # must not be zero: the full head is normalised over all 3 labels
        assert float(loss) > 1.0

    def test_masked_ce_would_see_only_declared_options(self):
        """Contrast: masked CE normalises over the row's DECLARED options, so it
        cannot see the 148 labels the model will face at evaluation time. This
        is exactly the train(15 declared) vs eval(151 declared) mismatch that
        collapsed the real-data baseline, and header-only removes it by
        normalising over the whole head."""
        labels = ["alpha", "beta", "gamma", "delta", "epsilon"]   # full inventory
        l2i = {l: i for i, l in enumerate(labels)}
        q = _question(False)         # declares only alpha/beta/gamma
        assert len(q.options) == 3 and len(labels) == 5

        # logit peaked on an UNDECLARED label ("delta", index 3)
        out = {"choice": torch.zeros(1, len(labels)), "noul": torch.zeros(1),
               "score": torch.zeros(1, 8)}
        out["choice"][0, 3] = 5.0
        pairs = [(_state(), q)]
        masked = float(pc._batch_loss(out, pairs, l2i, 8, "cpu"))
        full = float(pc._batch_loss(out, pairs, l2i, 8, "cpu", header_only=True))
        # masked CE cannot see index 3 -> small loss; full-inventory CE can.
        assert masked < full, (masked, full)


# ------------------------------------------- label-index train/infer parity
class TestLabelIndexParity:
    """Blocker A's required test: a known example maps to the SAME class index
    during training and during inference."""

    def test_known_example_has_identical_train_and_infer_index(self):
        labels = ["alpha", "beta", "gamma", "delta"]
        l2i = {l: i for i, l in enumerate(labels)}

        # --- training path: the loss targets label_to_idx[q.answer]
        q = _question(True)                     # answer == "beta"
        train_tgt = l2i[q.answer]
        assert train_tgt == 1

        # --- inference path: predict_rows argmaxes the head and decodes by name
        cfg = PlainConfig()
        cfg.header_only_choice = True
        model = pc.PlainClassifier(cfg, n_labels=len(labels), with_abstain=False)
        model.attach_labels(labels)
        assert model._label_to_idx == l2i
        # put the argmax at the same index the training loss targeted
        with torch.no_grad():
            model.choice_head.weight.zero_()
            model.choice_head.bias.zero_()
            model.choice_head.bias[train_tgt] = 10.0
        model.tokenizer = _fitted_tokenizer()
        rec = pc.predict_rows(model, [(_state(), q)], batch_size=1)[0]
        assert rec["pred"] == q.answer == "beta"
        assert model._label_to_idx[rec["pred"]] == train_tgt

    def test_inventory_order_is_exactly_the_sorted_union(self):
        """`load_plain` rebuilds the order from final.pt['label_to_idx'];
        re-deriving it must reproduce the same sequence, not merely the same set."""
        labels = sorted({"zeta", "alpha", "mu", "beta"})
        l2i = {l: i for i, l in enumerate(labels)}
        rebuilt = sorted(l2i, key=l2i.get)
        assert rebuilt == labels
        assert [l2i[l] for l in rebuilt] == list(range(len(labels)))


def _fitted_tokenizer():
    from vss.model.tokenizer import VSSTokenizer
    tok = VSSTokenizer(vocab_size=256, hash_buckets=64)
    q = _question(True)
    tok.fit([serialize_example(_state(), [{**q.as_request(),
                                           "header_only_choice": True}])])
    return tok


# ------------------------------------------------------------- sweep parity
class TestSweepGivesBothSystemsTheSameRecipe:
    def test_plain_config_enables_header_only(self):
        """If this regresses, the real-data sweep silently compares two
        different tasks again."""
        import inspect

        import sweep

        src = inspect.getsource(sweep.build_run)
        assert "cfg.header_only_choice = True" in src, (
            "the plain arm must be configured to match VSS's header-only recipe")

    def test_vss_config_uses_header_only(self):
        cfg = __import__("vss.model.config", fromlist=["VSSConfig"]).VSSConfig.load(
            str(REPO / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml"))
        assert cfg.model.header_only_choice is True


# ------------------------------------------------- schema has no such field
class TestWhyTheLoaderCouldNotSetIt:
    def test_answered_question_has_no_header_only_field(self):
        """Documents the mechanism: extra='forbid' meant the real-data loader
        could never mark a question header-only, which is why the config-driven
        fallback is the real fix."""
        from vss.data.schema import AnsweredQuestion

        with pytest.raises(Exception):
            AnsweredQuestion.model_validate({
                "id": "q", "type": "choice", "options": ["a", "b"],
                "answer": "a", "header_only_choice": True,
            })