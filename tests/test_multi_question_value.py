"""Tests for the multi-question value benchmark package.

Kept fast: tiny configs, a handful of synthetic states, no real checkpoint
loads (torch.load of 11M-param checkpoints is too slow for the suite). The
heavy machinery (real training, real latency, interference/permutation) is
exercised by the CLI scripts; these tests pin the contracts they rely on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "multi_question_value"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dataset as mqv_dataset  # noqa: E402
import metrics as mqv_metrics  # noqa: E402
from config import BenchConfig, PlainConfig  # noqa: E402


# ----------------------------------------------------------------- dataset

def test_synthetic_gold_is_deterministic_and_verified():
    exs_a = mqv_dataset.generate_synthetic(12, seed=7, split="goldtest", force=True)
    exs_b = mqv_dataset.generate_synthetic(12, seed=7, split="goldtest", force=True)
    assert len(exs_a) == len(exs_b) == 12
    for ea, eb in zip(exs_a, exs_b):
        assert ea.to_training_dict() == eb.to_training_dict()
        ea.verify_gold()  # raises if any answer violates its question schema


def test_synthetic_pool_prefixes_are_nested():
    full = mqv_dataset.POOL
    assert len(full) == 64
    assert len(mqv_dataset._base_pool()) == 16
    assert [s.qid for s in full[:16]] == [s.qid for s in mqv_dataset._base_pool()]
    # variants reuse the answer functions of the base templates
    for suf in mqv_dataset.VARIANT_SUFFIXES:
        for i, base in enumerate(mqv_dataset._base_pool()):
            assert full[16 + i].answer_fn is not None


def test_eval_subset_is_identical_across_modes_and_nested():
    exs = mqv_dataset.generate_synthetic(20, seed=7, split="goldtest", force=True)
    s1 = mqv_dataset.eval_subset(exs, 4, split_seed=42, cap=10)
    s2 = mqv_dataset.eval_subset(exs, 4, split_seed=42, cap=10)
    assert [(e.state, [q.id for q in e.questions]) for e in s1] == \
           [(e.state, [q.id for q in e.questions]) for e in s2]
    s_big = mqv_dataset.eval_subset(exs, 8, split_seed=42, cap=10)
    # nested prefixes: first 4 questions of the Q=8 set equal the Q=4 set
    for e4, e8 in zip(s1, s_big):
        assert e8.questions[:4] == e4.questions


def test_real_data_conversion_uses_stored_labels_and_gold():
    real = mqv_dataset.load_real("banking77", "validation")
    assert len(real) == 924
    ex = real[0]
    assert ex.questions[0].type == "choice"
    assert len(ex.questions[0].options) == 77
    assert ex.questions[0].answer in ex.questions[0].options
    # Q-scaling replication keeps gold and declared options
    sub = mqv_dataset.eval_subset(real[:5], 50, split_seed=42, cap=5)
    assert len(sub[0].questions) == 50
    assert all(q.answer == real[0].questions[0].answer for q in sub[0].questions)
    assert all(q.options == real[0].questions[0].options for q in sub[0].questions)


# ----------------------------------------------------------------- metrics

def test_per_question_metrics_are_per_question():
    recs = []
    for i in range(10):
        recs.append({"qid": f"c{i}", "type": "choice", "gold": "x",
                     "pred": "x" if i < 6 else "y", "correct": i < 6,
                     "conf": 0.5, "max_prob": 0.5, "probs": {"x": 0.5, "y": 0.5}})
    s = mqv_metrics.summarize(recs)
    assert s["n"] == 10
    assert abs(s["accuracy"] - 0.6) < 1e-9
    # request accuracy: all-correct requests only (second request has i=6..9 wrong)
    ra = mqv_metrics.request_accuracy([recs[:5], recs[5:]])
    assert abs(ra - 0.5) < 1e-9
    # F1_x = 0.75 (prec 1.0, rec 0.6), F1_y = 0 -> macro 0.375
    assert abs(s["macro_f1"] - 0.375) < 1e-9


def test_ece_perfect_confidence_is_zero():
    recs = [{"qid": str(i), "type": "noul", "gold": 1, "pred": 1,
             "correct": True, "conf": 1.0, "p_true": 1.0, "max_prob": 1.0}
            for i in range(50)]
    assert mqv_metrics.ece([r["conf"] for r in recs],
                           [r["correct"] for r in recs]) < 1e-9


def test_score_tolerance_and_brier_nll():
    assert mqv_metrics.correct_score(7.05, 7.0, 0.0, 10.0) is True
    assert mqv_metrics.correct_score(8.5, 7.0, 0.0, 10.0) is False  # tol = 1.0
    rec = {"qid": "s", "type": "score", "gold": 5.0, "pred": 5.0,
           "correct": True, "conf": 0.6, "value": 5.0,
           "probs": [0.1] * 7 + [0.3] + [0.1] * 56, "gold_bin": 7}
    s = mqv_metrics.summarize([rec])
    assert abs(s["score_mae"]) < 1e-9
    assert s["brier"] > 0  # distribution not a one-hot at gold


def test_auroc_recovers_perfect_separation():
    scores = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    assert abs(mqv_metrics.auroc(scores, labels) - 1.0) < 1e-9


def test_risk_coverage_monotone_non_increasing():
    recs = [{"qid": str(i), "type": "choice", "gold": "x", "pred": "x",
             "correct": True, "conf": 0.5 + 0.5 * i / 100, "max_prob": 1.0,
             "probs": {"x": 1.0}}
            for i in range(100)] + \
           [{"qid": f"b{i}", "type": "choice", "gold": "x", "pred": "y",
             "correct": False, "conf": 0.1, "max_prob": 0.2,
             "probs": {"x": 0.2, "y": 0.2}}
            for i in range(50)]
    rc = mqv_metrics.risk_coverage(recs, [0.5, 0.8, 1.0])
    assert rc["1.00"] <= rc["0.80"] <= rc["0.50"] + 1e-9


# --------------------------------------------------- plain classifier core

@pytest.fixture(scope="module")
def tiny_plain():
    import plain_classifier as pc
    import torch

    exs = mqv_dataset.generate_synthetic(4, seed=7, split="goldtest", force=True)
    labels = pc.build_label_inventory(exs, with_abstain=False)
    cfg = PlainConfig(epochs=1, vocab_size=1024, hash_buckets=256,
                      warmup_steps=2, batch_size=2)
    model = pc.PlainClassifier(cfg, n_labels=len(labels), with_abstain=False)
    model.attach_labels(labels)
    pc.init_scratch_(model, 0)
    from vss.model.serialize import serialize_example
    from vss.model.tokenizer import VSSTokenizer

    tok = VSSTokenizer(vocab_size=1024, hash_buckets=256).fit(
        [serialize_example(e.state, [q.as_request()])
         for e in exs for q in e.questions])
    model.tokenizer = tok
    return model, exs


def test_plain_forward_shapes(tiny_plain):
    import torch

    model, exs = tiny_plain
    pairs = [(exs[0].state, exs[0].questions[0])]
    x, _ = model.encode_rows(pairs, "cpu")
    out = model(x)
    assert out["choice"].shape == (1, model.n_labels)
    assert out["noul"].shape == (1,)
    assert out["score"].shape == (1, model.cfg.score_bins)


def test_plain_predictions_match_gold_schema(tiny_plain):
    model, exs = tiny_plain
    pairs = [(exs[0].state, q) for q in exs[0].questions[:4]]
    recs = __import__("plain_classifier").predict_rows(model, pairs)
    assert len(recs) == 4
    for r, q in zip(recs, pairs):
        if r["type"] == "choice":
            assert set(r["probs"]) == set(q[1].options) | ({"ABSTAIN"} if "ABSTAIN" in model._label_to_idx else set())
            assert abs(sum(r["probs"].values()) - 1.0) < 1e-4
        if r["type"] == "score":
            assert q[1].min <= r["value"] <= q[1].max


# ------------------------------------------------------- VSS runner pieces

def test_vss_record_extraction_from_tiny_model():
    import torch

    from vss.model.config import ModelConfig
    from vss.model.vss_model import VSSModel
    import vss_runner as vr

    cfg = ModelConfig(hidden_size=32, layers=2, heads=4, kv_heads=2,
                      intermediate_size=64, max_sequence_length=512,
                      vocab_size=1024, hash_buckets=256,
                      question_masked=True, header_only_choice=True,
                      use_refine_choice=False)
    model = VSSModel(cfg)
    tok_path = Path(__file__).resolve().parents[1] / "benchmarks" / \
        "multi_question_value" / "_tmp_vocab.json"
    from vss.model.tokenizer import VSSTokenizer

    exs = mqv_dataset.generate_synthetic(3, seed=7, split="goldtest", force=True)
    texts = []
    for ex in exs:
        for q in ex.questions:
            texts.append(__import__("vss.model.serialize", fromlist=["serialize_example"])
                         .serialize_example(ex.state, [q.as_request()]))
    tok = VSSTokenizer(vocab_size=1024, hash_buckets=256).fit(texts)
    tok.save(str(tok_path))
    model.tokenizer = tok
    model.vss_encoder.tokenizer = tok

    small = [mqv_dataset.MultiQuestionExample(state=e.state, questions=e.questions[:3])
             for e in exs[:2]]
    recs = vr.predict_examples(model, small, batch_size=2)
    assert len(recs) == 2 and len(recs[0]) == 3
    r = recs[0][0]
    assert {"qid", "type", "gold", "pred", "correct", "conf"} <= set(r)
    s = mqv_metrics.summarize([x for req in recs for x in req])
    assert 0.0 <= s["accuracy"] <= 1.0
    tok_path.unlink()
    # untrained model may abstain on everything; that is valid behavior


def test_lat_stats_percentiles():
    import benchmark as bm

    ts = [0.001 * i for i in range(1, 101)]
    st = bm.lat_stats(ts)
    assert st["p50_ms"] == pytest.approx(51.0, rel=0.01)  # sorted[50] of 1..100
    assert st["p95_ms"] >= st["p50_ms"]
    assert st["p99_ms"] >= st["p95_ms"]


def test_bench_config_defaults_match_prompt():
    cfg = BenchConfig()
    assert cfg.question_counts == [1, 2, 4, 8, 16, 32, 50]
    assert cfg.eval_seed == 42
    plain = PlainConfig()
    assert plain.seeds == [1, 2, 3]
    assert plain.epochs == 8 and plain.batch_size == 32
