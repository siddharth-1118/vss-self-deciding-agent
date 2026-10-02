"""Risk-coverage curves for the plain baseline and VSS at matched coverage.

The value-test report compares VSS's *selective* answered-accuracy against the
plain classifier's *always-on* accuracy, which is not a like-for-like metric.
This script puts both systems on the same footing: each question gets a
confidence score, questions are ranked by it, and accuracy is reported at fixed
coverage levels. VSS's shipped abstention gate is one point on this curve; the
plain classifier is scored with its own top-probability confidence, so it can
abstain too.

Writes JSON + a markdown table. No test split is touched here beyond the
benchmark's standard eval subset.

Usage:
    python benchmarks/convergence/risk_coverage.py \
        --vss-root runs/convergence/vss-synthetic-s1-lr0.0003-st400 \
        --plain-root runs/convergence/plain-synthetic-s1-lr0.0003-st3200-presmatch \
        --q 8 --out benchmarks/convergence/results/risk_coverage_synthetic.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "multi_question_value"))

import torch  # noqa: E402

torch.set_num_threads(6)

import dataset as mqv_dataset  # noqa: E402
import metrics  # noqa: E402
import plain_classifier as pc  # noqa: E402
import vss_runner  # noqa: E402
from config import PlainConfig  # noqa: E402

COVERAGES = [1.0, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]


def curve(records: list[dict]) -> dict:
    """Accuracy at fixed coverage, dropping lowest-confidence first."""
    ordered = sorted(records, key=lambda r: r["conf"], reverse=True)
    n = len(ordered)
    out = {}
    for cov in COVERAGES:
        k = max(1, int(round(cov * n)))
        sub = ordered[:k]
        out[f"{cov:.2f}"] = round(
            sum(1 for r in sub if r["correct"]) / len(sub), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vss-root", required=True)
    ap.add_argument("--plain-root", required=True)
    ap.add_argument("--dataset", default="synthetic")
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "results" / "risk_coverage.json"))
    a = ap.parse_args()

    if a.dataset == "synthetic":
        states = mqv_dataset.eval_subset(mqv_dataset.load_synthetic("test"), a.q,
                                         a.seed, a.n_states)
    else:
        states = mqv_dataset.eval_subset(mqv_dataset.load_real(a.dataset, "test"),
                                         a.q, a.seed, a.n_states)
    print(f"{len(states)} eval states at Q={a.q}", flush=True)

    plain = pc.load_plain(a.plain_root, PlainConfig())
    plain_recs = pc.predict_rows(
        plain,
        [(ex.state, q) for ex in states for q in ex.questions[: a.q]],
    )

    model = vss_runner.load_vss(a.vss_root)
    kw = vss_runner.default_inference_kwargs()
    per_request = vss_runner.predict_examples(model, states, **kw)
    vss_recs = [r for req in per_request for r in req]

    out = {
        "dataset": a.dataset,
        "q": a.q,
        "n_questions": len(plain_recs),
        "vss_root": a.vss_root,
        "plain_root": a.plain_root,
        "plain_curve": curve(plain_recs),
        "vss_curve": curve(vss_recs),
        "plain_always_on": round(sum(1 for r in plain_recs if r["correct"]) / len(plain_recs), 4),
        "vss_gate_coverage": round(
            sum(1 for r in vss_recs if not r.get("abstained")) / len(vss_recs), 4),
        "vss_gate_answered_accuracy": metrics.answered_accuracy(vss_recs),
        "plain_answerable_fraction": round(
            sum(1 for r in plain_recs if not r.get("abstained")) / len(plain_recs), 4),
        "plain_answered_accuracy": metrics.answered_accuracy(plain_recs),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\n{'coverage':>9} {'plain acc':>10} {'VSS acc':>10}")
    for cov in COVERAGES:
        k = f"{cov:.2f}"
        print(f"{k:>9} {out['plain_curve'][k]:>10.4f} {out['vss_curve'][k]:>10.4f}")
    print(f"\nVSS shipped gate: coverage {out['vss_gate_coverage']:.4f} "
          f"answered-accuracy {out['vss_gate_answered_accuracy']:.4f}")
    print(f"plain always-on : accuracy {out['plain_always_on']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())