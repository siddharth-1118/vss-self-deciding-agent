"""Human-readable dump of one or more benchmark results JSONs.

Convenience artifact for eyeballing results without opening the JSON; the
authoritative machine-readable output is the results JSON itself and the
generated tables in report.md.

Usage:
    python benchmarks/multi_question_value/digest.py \
        --results benchmarks/multi_question_value/results/synthetic_results.json \
        [more results files ...] \
        --out benchmarks/multi_question_value/results/DIGEST.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADER = (
    "  Q |  A/B acc  plainF1    ece  brier    nll |"
    "  C acc  C cov C ansacc   C F1    ece  brier    nll |"
    "               C-B ci |  A_req  C_req"
)


def f(v, nd=3):
    if v is None:
        return "None"
    try:
        if v != v:  # NaN
            return "nan"
    except TypeError:
        return str(v)
    return f"{v:.{nd}f}"


def digest(res: dict) -> str:
    out = ["=" * 70, f"DATASET {res['dataset']} | env: {json.dumps(res['env'])}"]
    out.append(HEADER)
    for c in sorted(res["cells"], key=lambda c: (c["seed"], c["q"])):
        A, C = c["A"], c["C"]
        b = c.get("boot_C_minus_B") or {}
        out.append(
            f"{c['q']:>3} | {f(A['accuracy'], 4):>9} {f(A['macro_f1']):>8} "
            f"{f(A['ece']):>6} {f(A['brier']):>6} {f(A['nll']):>6} |"
            f"{f(C['accuracy'], 4):>7} {f(C['coverage']):>6} "
            f"{f(C['answered_accuracy']):>8} {f(C['macro_f1']):>6} "
            f"{f(C['ece']):>6} {f(C['brier']):>6} {f(C['nll']):>6} |"
            f"  [{f(b.get('ci95_low'), 3)},{f(b.get('ci95_high'), 3)}] |"
            f"  {f(c.get('A_request_accuracy'))}  {f(c.get('C_request_accuracy'))}"
        )
    last = res["cells"][-1]
    out.append("")
    out.append(f"per-type accuracy (last cell): A {last['A'].get('per_type_accuracy')} "
               f"| C {last['C'].get('per_type_accuracy')}")
    out.append(f"A_equals_B all cells: {all(c.get('A_equals_B') for c in res['cells'])}")
    out.append("")
    out.append("latency (p50/p95/p99 ms) and throughput:")
    for c in res["cells"]:
        lat = c.get("latency", {})
        if c["q"] in (1, 50):
            out.append(f"  seed {c['seed']} Q={c['q']}: " + " | ".join(
                f"{m} p50={lat.get(m, {}).get('p50_ms')} "
                f"p95={lat.get(m, {}).get('p95_ms')} "
                f"q/s={lat.get(m, {}).get('questions_per_sec_p50')}"
                for m in ("A", "B", "C")))
    out.append("")
    out.append("interference (paired (state, question) sets, VSS solo vs joint):")
    for k in sorted(k for k in res if k.startswith("interference_")):
        blk = res[k]
        flips = sum(d.get("paired_decision_flips", 0) for d in blk["per_qid"].values())
        pairs = sum(d.get("solo_n", 0) for d in blk["per_qid"].values())
        out.append(f"  {k}: Q={blk['q']} n_states={blk['n_states']} "
                   f"mean_delta={blk['mean_delta_pts']} pts "
                   f"paired_decision_flips={flips}/{pairs}")
    out.append("")
    out.append("permutation (question-order agreement):")
    for k in sorted(k for k in res if k.startswith("permutation_")):
        blk = res[k]
        out.append(f"  {k}: agreement={blk['agreement']} "
                   f"n_pairs={blk['n_pairs']} n_states={blk['n_states']} "
                   f"note: {blk.get('note', '-')}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chunks = []
    for p in a.results:
        with open(p, encoding="utf-8") as fh:
            chunks.append(digest(json.load(fh)))
    Path(a.out).write_text("\n".join(chunks), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
