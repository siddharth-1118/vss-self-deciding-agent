"""Analysis for the convergence study: read sweep run logs, emit markdown.

Regenerates every number in docs/convergence_report.md from the run JSONs --
no hand-edited figures.

Usage:
    python benchmarks/convergence/analyze.py --out benchmarks/convergence/tables.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def load() -> list[dict]:
    out = []
    for p in sorted(RUNS.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("done"):
            out.append(d)
    return out


def _f(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:.{nd}f}"


def curves_table(recs: list[dict]) -> str:
    lines = ["| run | epoch | train loss | eval loss | choice acc | noul acc | "
             "score MAE | ECE | lr | grad norm | param update |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda r: (r["system"], r["dataset"], r["lr"])):
        for h in r["history"]:
            lines.append(
                f"| {r['system']}/{r['dataset']}/s{r['seed']}/lr{r['lr']:g} "
                f"| {h['epoch']} | {_f(h.get('train_loss'))} "
                f"| {_f(h.get('eval_loss', h.get('val_loss')))} "
                f"| {_f(h.get('choice_accuracy'), 4)} "
                f"| {_f(h.get('noul_accuracy'), 4)} "
                f"| {_f(h.get('score_mae'), 3)} | {_f(h.get('ece'))} "
                f"| {h.get('lr', 0):.2e} | {_f(h.get('grad_norm_mean'), 3)} "
                f"| {h.get('param_update_rel_mean', 0):.2e} |"
            )
    return "\n".join(lines)


def summary_table(recs: list[dict]) -> str:
    lines = ["| run | params | steps | epochs run | early stop | best val loss "
             "| final val loss | last-3-epoch Δ | best epoch | wall s |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda r: (r["dataset"], r["system"], r["lr"], r["seed"])):
        h = r["history"]
        losses = [e.get("eval_loss", e.get("val_loss")) for e in h]
        losses = [x for x in losses if x is not None]
        best = min(losses) if losses else float("nan")
        bi = losses.index(best) if losses else -1
        tail = losses[-3:] if len(losses) >= 3 else losses
        delta = (tail[-1] - min(tail)) if len(tail) >= 2 else float("nan")
        name = (f"{r['system']}/{r['dataset']}/s{r['seed']}/lr{r['lr']:g}")
        lines.append(
            f"| {name} | {r['params']:,} | {r['steps']} | {r['epochs_run']} "
            f"| {r['stopped_early']} | {_f(best)} | {_f(losses[-1] if losses else None)} "
            f"| {delta:+.4f} | {bi} | {r.get('wall_seconds', 0):.0f} |"
        )
    return "\n".join(lines)


def lr_selection(recs: list[dict]) -> str:
    """Pick the LR per system on VALIDATION loss only. Never on test."""
    lines = ["| system | dataset | lr | best val loss | final val loss | epochs | "
             "choice acc (last) |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda r: (r["dataset"], r["system"], r["lr"])):
        h = r["history"]
        losses = [e.get("eval_loss", e.get("val_loss")) for e in h]
        lines.append(
            f"| {r['system']} | {r['dataset']} | {r['lr']:g} "
            f"| {_f(min(losses))} | {_f(losses[-1])} | {r['epochs_run']} "
            f"| {_f(h[-1].get('choice_accuracy'))} |"
        )
    return "\n".join(lines)


def step_matched(recs: list[dict]) -> str:
    """Systems compared at the same optimizer-step budget.

    NOTE: the validation *loss* column is NOT comparable across systems. VSS's
    objective includes a calibration-head BCE term and a soft-target ordinal
    term that the plain baseline's `_batch_loss` does not have, so the same
    numeric value means different things. The per-task metrics (choice
    accuracy, noul accuracy, score MAE) ARE comparable: both trainers score
    choice rows by argmax over the row's declared options plus ABSTAIN.
    """
    lines = ["*(validation loss is not cross-system comparable -- see docstring; "
             "choice accuracy is)*", "",
             "| dataset | steps | passes | system | seeds | best val loss | "
             "best choice acc | best noul acc | best score MAE | params | wall s/run |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]

    def stats(vals):
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        if not vals:
            return "-"
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        return f"{m:.4f} ± {sd:.4f}"

    def best_of(r, key):
        vals = [e.get(key) for e in r["history"] if e.get(key) is not None]
        if not vals:
            return float("nan")
        return min(vals) if key == "score_mae" else max(vals)

    keys = sorted({(r["dataset"], r["steps"], r["system"]) for r in recs})
    for ds, steps, system in keys:
        sel = [r for r in recs
               if r["dataset"] == ds and r["system"] == system
               and abs(r["steps"] - steps) <= max(2, 0.05 * steps)]
        if not sel:
            continue
        best = [min(e.get("eval_loss", e.get("val_loss")) for e in r["history"])
                for r in sel]
        acc = [best_of(r, "choice_accuracy") for r in sel]
        noul = [best_of(r, "noul_accuracy") for r in sel]
        mae = [best_of(r, "score_mae") for r in sel]
        seeds = ",".join(str(r["seed"]) for r in sel)
        params = f"{sel[0]['params']:,}"
        wall = sum(r.get("wall_seconds", 0) for r in sel) / max(1, len(sel))
        passes = max((r["epochs_run"] for r in sel), default=0)
        lines.append(f"| {ds} | {steps} | {passes} | {system} | {len(sel)} ({seeds}) | "
                     f"{stats(best)} | {stats(acc)} | {stats(noul)} | {stats(mae)} "
                     f"| {params} | {wall:.0f} |")
    return "\n".join(lines)


def update_stats(recs: list[dict]) -> str:
    lines = ["| run | first-epoch grad norm | last-epoch grad norm | "
             "first-epoch param update | last-epoch param update | "
             "final lr |", "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda r: (r["dataset"], r["system"], r["lr"], r["seed"])):
        h = r["history"]
        if not h:
            continue
        lines.append(
            f"| {r['system']}/{r['dataset']}/s{r['seed']}/lr{r['lr']:g} "
            f"| {_f(h[0].get('grad_norm_mean'), 3)} "
            f"| {_f(h[-1].get('grad_norm_mean'), 3)} "
            f"| {h[0].get('param_update_rel_mean', 0):.2e} "
            f"| {h[-1].get('param_update_rel_mean', 0):.2e} "
            f"| {h[-1].get('lr', 0):.2e} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "tables.md"))
    a = ap.parse_args()
    recs = load()
    if not recs:
        print("no completed runs yet")
        return 0
    parts = [
        f"# Convergence sweep tables ({len(recs)} completed runs)\n",
        "Auto-generated by `benchmarks/convergence/analyze.py` from "
        "`benchmarks/convergence/runs/*.json`. Do not hand-edit.\n",
        "## Per-run summary\n", summary_table(recs), "\n",
        "## Learning-rate selection (validation only)\n", lr_selection(recs), "\n",
        "## Step-matched comparison\n", step_matched(recs), "\n",
        "## Optimizer health: gradient norms and parameter updates\n",
        update_stats(recs), "\n",
        "## Full per-epoch curves\n", curves_table(recs), "\n",
    ]
    Path(a.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {a.out} ({len(recs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
