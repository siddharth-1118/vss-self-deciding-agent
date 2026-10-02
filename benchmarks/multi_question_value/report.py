"""Aggregate benchmark results into report-ready markdown.

Reads one or more results JSONs (benchmark.py output) plus training logs and
emits the final comparison tables used by docs/multi_question_value_report.md.

Usage:
    python benchmarks/multi_question_value/report.py \
        --results benchmarks/multi_question_value/results/synthetic_results.json \
        [more results files ...] \
        --out benchmarks/multi_question_value/report.md
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _fmt(v: float, nd: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def _ms(vals: list[float]) -> str:
    """mean ± std formatting."""
    return f"{_fmt(_mean(vals), 4)} ± {_fmt(_std(vals), 4)}"


def load_results(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def seed_cells(res: dict) -> dict[int, dict[int, dict]]:
    """res -> {seed: {q: cell}}"""
    by_seed: dict[int, dict[int, dict]] = defaultdict(dict)
    for c in res["cells"]:
        by_seed[c["seed"]][c["q"]] = c
    return by_seed


def accuracy_table(res: dict) -> str:
    by_seed = seed_cells(res)
    qs = sorted({c["q"] for c in res["cells"]})
    lines = ["| Q | A accuracy | B accuracy | C accuracy | request acc (A) | request acc (C) |",
             "|--:|-----------:|-----------:|-----------:|----------------:|----------------:|"]
    for q in qs:
        row = [str(q)]
        for m in ("A", "B", "C"):
            vals = [by_seed[s][q][m]["accuracy"] for s in by_seed
                    if q in by_seed[s] and m in by_seed[s][q]]
            row.append(_ms(vals))
        for key in ("A_request_accuracy", "C_request_accuracy"):
            vals = [by_seed[s][q][key] for s in by_seed
                    if q in by_seed[s] and key in by_seed[s][q]]
            row.append(_ms(vals) if key in by_seed[min(by_seed)][q] else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def calibration_table(res: dict, qs: list[int]) -> str:
    by_seed = seed_cells(res)
    lines = ["| Q | metric | A | B | C |", "|--:|--------|--:|--:|--:|"]
    for q in qs:
        for metric in ("ece", "brier", "nll"):
            row = [str(q), metric]
            for m in ("A", "B", "C"):
                vals = [by_seed[s][q][m][metric] for s in by_seed
                        if q in by_seed[s] and m in by_seed[s][q]]
                row.append(_ms(vals))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def latency_table(res: dict) -> str:
    by_seed = seed_cells(res)
    qs = sorted({c["q"] for c in res["cells"]
                 if "latency" in c})
    lines = ["| Q | stat | A sequential | B batched | C VSS |",
             "|--:|------|-------------:|----------:|------:|"]
    for q in qs:
        for stat, lab in (("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"), ("p99_ms", "p99 ms")):
            row = [str(q), lab]
            for m in ("A", "B", "C"):
                vals = [by_seed[s][q]["latency"][m][stat] for s in by_seed
                        if q in by_seed[s] and "latency" in by_seed[s][q]
                        and m in by_seed[s][q]["latency"]]
                row.append(_fmt(_mean(vals), 1) if vals else "-")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def throughput_table(res: dict) -> str:
    by_seed = seed_cells(res)
    qs = sorted({c["q"] for c in res["cells"] if "latency" in c})
    lines = ["| Q | A req/s | B req/s | C req/s | A q/s | B q/s | C q/s |",
             "|--:|--------:|--------:|--------:|------:|------:|------:|"]
    for q in qs:
        row = [str(q)]
        for m in ("A", "B", "C"):
            vals = [by_seed[s][q]["latency"][m].get("requests_per_sec_p50")
                    for s in by_seed if q in by_seed[s]
                    and "latency" in by_seed[s][q] and m in by_seed[s][q]["latency"]]
            vals = [v for v in vals if v is not None]
            row.append(_fmt(_mean(vals), 1) if vals else "-")
        for m in ("A", "B", "C"):
            vals = [by_seed[s][q]["latency"][m].get("questions_per_sec_p50")
                    for s in by_seed if q in by_seed[s]
                    and "latency" in by_seed[s][q] and m in by_seed[s][q]["latency"]]
            vals = [v for v in vals if v is not None]
            row.append(_fmt(_mean(vals), 1) if vals else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def speedup_table(res: dict) -> str:
    """Question scaling: latency ratios B/A, C/A, C/B per Q (p50, p95)."""
    by_seed = seed_cells(res)
    qs = sorted({c["q"] for c in res["cells"] if "latency" in c})
    lines = ["| Q | B/A p50 | C/A p50 | C/B p50 | C/A p95 |",
             "|--:|--------:|--------:|--------:|--------:|"]
    for q in qs:
        def ratio(stat: str, num: str, den: str) -> float:
            vals = []
            for s in by_seed:
                if q in by_seed[s] and "latency" in by_seed[s][q]:
                    lat = by_seed[s][q]["latency"]
                    if num in lat and den in lat and lat[den]["p50_ms"] > 0:
                        vals.append(lat[num][stat] / lat[den][stat])
            return _mean(vals)
        lines.append("| {} | {} | {} | {} | {} |".format(
            q,
            _fmt(ratio("p50_ms", "B", "A"), 2),
            _fmt(ratio("p50_ms", "C", "A"), 2),
            _fmt(ratio("p50_ms", "C", "B"), 2),
            _fmt(ratio("p95_ms", "C", "A"), 2),
        ))
    return "\n".join(lines)


def boot_block(res: dict, qs: list[int]) -> str:
    by_seed = seed_cells(res)
    lines = ["| Q | mean acc diff C-B | 95% CI | n |", "|--:|------------------:|--------|--:|"]
    for q in qs:
        lows, highs, means = [], [], []
        for s in by_seed:
            b = by_seed[s].get(q, {}).get("boot_C_minus_B")
            if b and b.get("mean_diff") is not None:
                means.append(b["mean_diff"])
                lows.append(b["ci95_low"])
                highs.append(b["ci95_high"])
        if not means:
            continue
        lines.append(f"| {q} | {_fmt(_mean(means), 4)} | "
                     f"[{_fmt(_mean(lows), 4)}, {_fmt(_mean(highs), 4)}] | "
                     f"{by_seed[min(by_seed)][q]['boot_C_minus_B'].get('n', '-')} |")
    return "\n".join(lines)


def _seed_of(key: str) -> str:
    """'interference_Q8_s1' -> '1'; 'permutation_Q8_s13' -> '13'."""
    tail = key.rsplit("_s", 1)[-1]
    return tail if tail.isdigit() else "?"


def extras(res: dict) -> str:
    chunks = []
    for key in sorted(res):
        if key.startswith("interference_Q"):
            blk = res[key]
            flips = sum(d.get("paired_decision_flips", 0)
                        for d in blk["per_qid"].values())
            pairs = sum(d.get("solo_n", 0) for d in blk["per_qid"].values())
            rows = ["### Interference (solo vs joint, VSS)",
                    f"Q={blk['q']}, seed={_seed_of(key)}, "
                    f"n_states={blk['n_states']}, "
                    f"mean delta = {blk['mean_delta_pts']} pts, "
                    f"paired decision flips = {flips}/{pairs}", "",
                    "| question | solo | joint | delta pts | paired flips |",
                    "|----------|-----:|------:|----------:|--------------:|"]
            for qid, d in sorted(blk["per_qid"].items()):
                rows.append(f"| {qid} | {d['solo']:.4f} | {d['joint']:.4f} | "
                            f"{d['delta_pts']:+.2f} | "
                            f"{d.get('paired_decision_flips', '-')} |")
            chunks.append("\n".join(rows))
        elif key.startswith("permutation_Q"):
            blk = res[key]
            chunks.append(
                f"### Question order (permutation)\nQ={blk['q']}, "
                f"seed={_seed_of(key)}, "
                f"agreement = {blk['agreement']:.4f} over {blk['n_pairs']} pairs")
    return "\n\n".join(chunks)


def accuracy_latency(res: dict) -> str:
    """Joint accuracy x p95 latency view (no single arbitrary score)."""
    by_seed = seed_cells(res)
    qs = sorted({c["q"] for c in res["cells"]})
    lines = ["| Q | A acc | A p95 ms | B acc | B p95 ms | C acc | C p95 ms | C p95 per q (ms) |",
             "|--:|------:|---------:|------:|---------:|------:|---------:|-----------------:|"]
    for q in qs:
        row = [str(q)]
        for m in ("A", "B", "C"):
            accs = [by_seed[s][q][m]["accuracy"] for s in by_seed
                    if q in by_seed[s] and m in by_seed[s][q]]
            lats = [by_seed[s][q]["latency"][m]["p95_ms"] for s in by_seed
                    if q in by_seed[s] and "latency" in by_seed[s][q]
                    and m in by_seed[s][q]["latency"]]
            row.append(_fmt(_mean(accs), 4))
            row.append(_fmt(_mean(lats), 1) if lats else "-")
        latsC = [by_seed[s][q]["latency"]["C"]["p95_ms"] / q for s in by_seed
                 if q in by_seed[s] and "latency" in by_seed[s][q]
                 and "C" in by_seed[s][q]["latency"]]
        row.append(_fmt(_mean(latsC), 2) if latsC else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_report(results: list[dict], title: str) -> str:
    parts = [f"# Multi-question value results: {title}", ""]
    for res in results:
        ds = res["dataset"]
        env = res.get("env", {})
        parts.append(f"## Dataset: {ds}")
        parts.append(f"seeds={res['seeds']}, Q={res['questions']}, "
                     f"n_eval_states={res.get('n_eval_states')}, "
                     f"torch={env.get('torch')}, threads={env.get('torch_threads')}, "
                     f"precision={env.get('precision')}")
        parts.append("")
        parts.append("### Per-question accuracy (mean ± std over seeds)")
        parts.append(accuracy_table(res))
        parts.append("")
        qs = sorted({c["q"] for c in res["cells"]})
        parts.append("### Calibration (per-question)")
        parts.append(calibration_table(res, qs))
        parts.append("")
        if any("latency" in c for c in res["cells"]):
            parts.append("### Request latency (percentiles)")
            parts.append(latency_table(res))
            parts.append("")
            parts.append("### Throughput (p50)")
            parts.append(throughput_table(res))
            parts.append("")
            parts.append("### Question scaling (latency ratios)")
            parts.append(speedup_table(res))
            parts.append("")
            parts.append("### Accuracy x latency (joint view)")
            parts.append(accuracy_latency(res))
            parts.append("")
        if any("boot_C_minus_B" in c for c in res["cells"]):
            parts.append("### Paired bootstrap: C vs B accuracy difference")
            parts.append(boot_block(res, qs))
            parts.append("")
        ex = extras(res)
        if ex:
            parts.append(ex)
            parts.append("")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--title", default="multi-question value benchmark")
    ap.add_argument("--out", default="benchmarks/multi_question_value/report.md")
    args = ap.parse_args()
    results = load_results(args.results)
    report = build_report(results, args.title)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report, encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
