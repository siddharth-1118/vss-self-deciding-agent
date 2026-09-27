"""TF-IDF + LogisticRegression baseline (fair-baseline protocol).

Same dataset files, same splits, same test subsample seed as the VSS
evaluation. Options are irrelevant to this baseline (it scores all classes),
which is exactly the comparison: does VSS's dynamic-schema head beat a
classical classifier on the same task?

Usage:
    python scripts/baseline_tfidf.py --data-dir data/clinc150 --name clinc150
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from vss.data.schema import load_jsonl  # noqa: E402
from vss.model.calibration import brier_score, expected_calibration_error, nll  # noqa: E402


def text_of(ex) -> str:
    s = ex.state
    if isinstance(s, dict):
        return " ".join(str(v) for v in s.values())
    return " ".join(str(x) for x in s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="benchmarks/baselines")
    args = ap.parse_args()

    d = Path(args.data_dir)
    train = load_jsonl(str(d / "train.jsonl"))
    test = load_jsonl(str(d / "test.jsonl"))
    if args.n_samples < len(test):
        test = random.Random(args.seed).sample(test, args.n_samples)

    X_train = [text_of(e) for e in train]
    y_train = [e.questions[0].answer for e in train]
    X_test = [text_of(e) for e in test]
    y_test = [e.questions[0].answer for e in test]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, C=4.0, n_jobs=-1)),
    ])
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = pipe.predict_proba(X_test)
    infer_s = time.perf_counter() - t0
    classes = list(pipe.named_steps["lr"].classes_)
    preds = [classes[i] for i in proba.argmax(axis=1)]

    # confidence = max class prob (standard for softmax-calibrated LR)
    confs = [float(p.max()) for p in proba]
    correct = [int(p == g) for p, g in zip(preds, y_test)]

    from vss.eval.metrics import accuracy, f1_scores

    gold_set = set(y_test)
    abstain_like = sum(1 for p in preds if p not in gold_set)  # sanity, should be 0
    report = {
        "baseline": "tfidf_lr",
        "data": str(d),
        "n_train": len(train),
        "n_test": len(test),
        "sampling": f"seeded subsample seed={args.seed}",
        "accuracy": accuracy(preds, y_test),
        "macro_f1": f1_scores(preds, y_test)["macro"],
        "ece": expected_calibration_error(confs, correct),
        "brier": brier_score(confs, correct),
        "nll": nll(confs, correct),
        "fit_seconds": round(fit_s, 2),
        "infer_seconds": round(infer_s, 2),
        "throughput_examples_per_s": round(len(test) / infer_s, 1),
        "invalid_label_predictions": abstain_like,
        "n_classes": len(classes),
    }
    out = Path(args.out_dir) / f"{args.name}-tfidf_lr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
