"""Evaluate a trained VSS checkpoint: accuracy, calibration, latency.

Usage:
    python scripts/evaluate.py --model runs/prototype/final \
        --data data/generated/eval.jsonl [--latency]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from vss.api import VSS  # noqa: E402
from vss.data.schema import load_jsonl  # noqa: E402
from vss.eval.metrics import (  # noqa: E402
    LatencyReport,
    accuracy,
    f1_scores,
    resource_usage,
    roc_pr,
    time_decide,
)
from vss.model.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    nll,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--sweep-thresholds", action="store_true")
    ap.add_argument("--abstain-threshold", type=float, default=None)
    args = ap.parse_args()

    model = VSS.from_pretrained(args.model)
    if args.abstain_threshold is not None:
        model.inference_cfg.abstain_threshold = args.abstain_threshold
    examples = load_jsonl(args.data)

    preds: list[str] = []
    golds: list[str] = []
    confs: list[float] = []
    correct: list[int] = []
    score_errors: list[float] = []
    abstained = 0
    score_abstained = 0
    total_choice = 0
    total_score = 0

    for ex in examples:
        result = model.decide(ex.state, [q.as_request() for q in ex.questions])
        for q in ex.questions:
            a = result["answers"][q.id]
            if q.type == "choice":
                total_choice += 1
                if a["value"] == "ABSTAIN":
                    abstained += 1
                preds.append(str(a["value"]))
                golds.append(str(q.answer))
                # effective confidence [vss]: for abstain entries, the
                # model's P(abstaining is right); otherwise its confidence
                confs.append(float(a.get("abstain_probability", a["confidence"])))
                correct.append(int(a["value"] == q.answer))
            elif q.type == "noul":
                pred = a["value"]
                confs.append(float(a["confidence"]))
                correct.append(int(pred == int(q.answer)))
            else:
                total_score += 1
                if a["value"] == "ABSTAIN":
                    score_abstained += 1
                    continue
                err = abs(float(a["value"]) - float(q.answer))
                denom = max(1e-6, float(q.max) - float(q.min))
                score_errors.append(err / denom)

    report = {
        "n_examples": len(examples),
        "choice": {
            "accuracy": accuracy(preds, golds),
            "f1": f1_scores(preds, golds),
            "abstain_rate": abstained / max(1, total_choice),
            "note": "accuracy includes correct ABSTAIN answers (gold=ABSTAIN rows exist)",
        },
        "calibration": {
            "ece": expected_calibration_error(confs, correct),
            "brier": brier_score(confs, correct),
            "nll": nll(confs, correct),
            **roc_pr(confs, correct),
        },
        "score": {
            "mean_relative_error": (
                sum(score_errors) / max(1, len(score_errors)) if score_errors else None
            ),
            "abstain_rate": score_abstained / max(1, total_score),
            "n_scored": len(score_errors),
        },
        "resources": resource_usage(),
    }

    def _sanitized(o: Any) -> Any:
        """NaN -> null so the report is strictly valid JSON."""
        import math as _m

        if isinstance(o, dict):
            return {k: _sanitized(v) for k, v in o.items()}
        if isinstance(o, float) and _m.isnan(o):
            return None
        return o

    print(json.dumps(_sanitized(report), indent=2))

    if args.sweep_thresholds:
        print("abstain threshold sweep (choice questions):")
        print("threshold | accuracy | abstain_rate")
        for thr in (0.0, 0.3, 0.4, 0.5, 0.55, 0.7):
            model.inference_cfg.abstain_threshold = thr
            p2: list[str] = []
            g2: list[str] = []
            ab = 0
            tot = 0
            for ex in examples:
                if not any(q.type == "choice" for q in ex.questions):
                    continue
                result = model.decide(ex.state, [q.as_request() for q in ex.questions])
                for q in ex.questions:
                    if q.type != "choice":
                        continue
                    a = result["answers"][q.id]
                    tot += 1
                    if a["value"] == "ABSTAIN":
                        ab += 1
                    p2.append(str(a["value"]))
                    g2.append(str(q.answer))
            print(f"{thr:>9.2f} | {accuracy(p2, g2):>8.3f} | {ab / max(1, tot):>11.3f}")

    if args.latency:
        lat = LatencyReport()
        state = {"message": "My card was charged twice and I need a refund."}
        for n_q in (1, 5, 10, 50):
            qs = (
                [
                    {"id": f"q{i}", "type": "choice",
                     "options": ["billing", "technical", "sales", "shipping", "other"]}
                    for i in range(n_q)
                ]
            )
            fn = lambda: model.decide(state, qs)  # noqa: E731
            lat.add(n_q, time_decide(fn, repeat=20, warmup=3))
        print("latency scaling (one forward pass per request):")
        print(lat.table())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
