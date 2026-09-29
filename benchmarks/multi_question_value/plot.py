"""SVG plot generation for the multi-question value benchmark.

No matplotlib dependency (not installed in this environment): charts are
generated as plain SVG files with fixed viewBox, axes, ticks, polyline
series and a legend. Deterministic output.

Usage:
    python benchmarks/multi_question_value/plot.py \
        --results benchmarks/multi_question_value/results/synthetic_results.json \
        --outdir benchmarks/multi_question_value/plots
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

W, H = 640, 400
MARGIN = 48
COLORS = {"A": "#d62728", "B": "#ff7f0e", "C": "#1f77b4"}
LABELS = {"A": "A sequential classifier", "B": "B batched classifier",
          "C": "C VSS single-pass"}


def _scale(vals: list[float], lo: float, hi: float, px0: float, px1: float):
    span = (hi - lo) or 1.0
    return [px0 + (v - lo) / span * (px1 - px0) for v in vals]


def line_chart_svg(series: dict[str, list[tuple[float, float]]],
                   x_label: str, y_label: str, title: str,
                   y_log: bool = False) -> str:
    """series: name -> [(x, y), ...]; returns SVG string."""
    xs = [p[0] for pts in series.values() for p in pts]
    ys = [p[1] for pts in series.values() for p in pts]
    if y_log:
        ys = [max(y, 1e-4) for y in ys]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    pad_y = (y_hi - y_lo) * 0.08 or 0.1
    y_lo, y_hi = y_lo - pad_y, y_hi + pad_y
    x0, x1 = MARGIN, W - 16
    y0, y1 = H - 40, MARGIN  # pixel y bounds (inverted axis)

    def sx(v): return _scale([v], x_lo, x_hi, x0, x1)[0]

    def sy(v):
        if y_log:
            import math
            lo, hi = math.log10(max(y_lo, 1e-4)), math.log10(max(y_hi, 1e-4))
            return _scale([math.log10(max(v, 1e-4))], lo, hi, y0, y1)[0]
        return _scale([v], y_lo, y_hi, y0, y1)[0]

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'font-family="monospace" font-size="11">',
             f'<rect width="{W}" height="{H}" fill="white"/>',
             f'<text x="{W / 2}" y="18" text-anchor="middle">{title}</text>']
    # axes + ticks (5 y ticks, all x points)
    for i in range(6):
        v = y_lo + (y_hi - y_lo) * i / 5
        py = sy(v)
        parts.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x1}" y2="{py:.1f}" '
                     f'stroke="#eee"/>')
        lab = f"{v:.3g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"
        parts.append(f'<text x="{x0 - 6}" y="{py + 3:.1f}" text-anchor="end" '
                     f'fill="#555">{lab}</text>')
    xt = sorted({p[0] for pts in series.values() for p in pts})
    for xv in xt:
        px = sx(xv)
        parts.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" '
                     f'y2="{y0 + 4}" stroke="#888"/>')
        parts.append(f'<text x="{px:.1f}" y="{y0 + 16}" text-anchor="middle" '
                     f'fill="#555">{xv:g}</text>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#333"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>')
    parts.append(f'<text x="{W / 2}" y="{H - 4}" text-anchor="middle" '
                 f'fill="#333">{x_label}</text>')
    parts.append(f'<text x="12" y="{H / 2}" text-anchor="middle" fill="#333" '
                 f'transform="rotate(-90 12 {H / 2})">{y_label}</text>')
    # series
    lx = x1 - 190
    ly = y1 + 6
    for i, (name, pts) in enumerate(sorted(series.items())):
        pts = sorted(pts)
        poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        color = COLORS.get(name.split()[0], "#2ca02c")
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" '
                     f'stroke-width="2"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3" '
                         f'fill="{color}"/>')
        parts.append(f'<rect x="{lx}" y="{ly + i * 16}" width="14" height="3" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{ly + i * 16 + 5}">'
                     f'{LABELS.get(name.split()[0], name)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _mean_over_seeds(cells: list[dict], mode: str, q: int, key: str) -> float:
    vals = [c[mode][key] for c in cells
            if c["q"] == q and mode in c and key in c[mode]]
    return sum(vals) / len(vals) if vals else float("nan")


def _lat(cells: list[dict], mode: str, q: int, stat: str) -> float:
    vals = [c["latency"][mode][stat] for c in cells
            if c["q"] == q and "latency" in c and mode in c["latency"]]
    return sum(vals) / len(vals) if vals else float("nan")


def make_plots(results_path: str, outdir: str) -> list[str]:
    with open(results_path, encoding="utf-8") as f:
        res = json.load(f)
    cells = res["cells"]
    qs = sorted({c["q"] for c in cells})
    ds = res.get("dataset", "data")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    modes = [m for m in ("A", "B", "C")
             if any(m in c for c in cells)]

    # accuracy vs Q
    series = {m: [(q, _mean_over_seeds(cells, m, q, "accuracy")) for q in qs]
              for m in modes}
    p = out / f"{ds}_accuracy_vs_Q.svg"
    p.write_text(line_chart_svg(series, "questions per request",
                                "per-question accuracy", f"{ds}: accuracy vs Q"),
                 encoding="utf-8")
    made.append(str(p))

    # ECE vs Q
    series = {m: [(q, _mean_over_seeds(cells, m, q, "ece")) for q in qs]
              for m in modes}
    p = out / f"{ds}_ece_vs_Q.svg"
    p.write_text(line_chart_svg(series, "questions per request", "ECE",
                                f"{ds}: calibration (ECE) vs Q"), encoding="utf-8")
    made.append(str(p))

    if all("latency" in c for c in cells):
        # latency p50 vs Q (log scale) — THE question-scaling graph
        for stat, fname, ttl in (
            ("p50_ms", "p50", "median request latency"),
            ("p95_ms", "p95", "p95 request latency"),
        ):
            series = {m: [(q, _lat(cells, m, q, stat)) for q in qs]
                      for m in modes if any(m in c["latency"] for c in cells)}
            p = out / f"{ds}_latency_{fname}_vs_Q.svg"
            p.write_text(line_chart_svg(series, "questions per request",
                                        "ms (log scale)",
                                        f"{ds}: {ttl} vs Q", y_log=True),
                         encoding="utf-8")
            made.append(str(p))

        # throughput: questions/sec vs Q
        series = {}
        for m in modes:
            pts = []
            for q in qs:
                vals = [c["latency"][m].get("questions_per_sec_p50")
                        for c in cells if c["q"] == q and m in c["latency"]]
                vals = [v for v in vals if v is not None]
                if vals:
                    pts.append((q, sum(vals) / len(vals)))
            if pts:
                series[m] = pts
        if series:
            p = out / f"{ds}_throughput_vs_Q.svg"
            p.write_text(line_chart_svg(series, "questions per request",
                                        "questions / second (p50)",
                                        f"{ds}: throughput vs Q"), encoding="utf-8")
            made.append(str(p))
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default="benchmarks/multi_question_value/plots")
    args = ap.parse_args()
    for path in make_plots(args.results, args.outdir):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
