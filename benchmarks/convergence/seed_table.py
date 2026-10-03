"""Aggregate the convergence-matched multi-seed runs into per-configuration statistics.

The single-seed result for VSS on Banking77 was 0.895 against plain's 0.870 --
a nominal 2.5-point lead. Running the same configuration at three seeds shows
that lead was seed luck: VSS spans 0.765-0.895 (sd 0.065) while plain spans
0.870-0.885 (sd 0.008), and plain wins on the mean.

This script exists so that table is reproducible from the run files rather than
transcribed by hand. It reports mean, sample standard deviation, min/max spread
and per-seed values, and it refuses to rank two configurations whose seed sets
differ, so a 3-seed number is never compared against a 1-seed number.

Usage:
    python benchmarks/convergence/seed_table.py
    python benchmarks/convergence/seed_table.py --steps 2000
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "convergence"))

from lr_table import selected_epoch  # noqa: E402


def collect(steps: int, rule: str = "loss"):
    """Gather (seed, accuracy, wall, early) per (dataset, system).

    `rule` selects how the reported accuracy is picked from each run's history:
    "loss" reproduces what the trainer did at the time (minimum validation loss),
    "accuracy" is the corrected rule (maximum validation choice accuracy, loss as
    tie-break). Both are derived from the same recorded history, so the
    corrected comparison needs no new training.
    """
    runs: dict[tuple[str, str], list[tuple[int, float, float, bool]]] = {}
    pattern = str(ROOT / "benchmarks" / "convergence" / "runs" /
                  f"*-st{steps}-final.json")
    for path in sorted(glob.glob(pattern)):
        run = json.load(open(path, encoding="utf-8"))
        history = [h for h in run.get("history", [])
                   if h.get("choice_accuracy") is not None]
        if not history:
            continue
        if rule == "accuracy":
            # Max accuracy, ties broken by the shared rule's secondary metric
            # (loss), earliest epoch on a full tie.
            def key(h):
                loss = h.get("val_loss", h.get("eval_loss"))
                return (-float(h["choice_accuracy"]),
                        float(loss) if loss is not None else float("inf"), h["epoch"])
            sel = min(history, key=key)
        else:
            sel = selected_epoch(run)
            if sel is None:
                continue
        stem = Path(path).stem.replace(f"-st{steps}-final", "")
        system, dataset, seed = stem.split("-")[0], stem.split("-")[1], \
            int(stem.split("-")[2].lstrip("s"))
        key2 = (dataset, system)
        runs.setdefault(key2, []).append(
            (seed, float(sel["choice_accuracy"]),
             float(run.get("wall_seconds", 0)), bool(run["stopped_early"])))
    for key in runs:
        runs[key].sort()
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    args = ap.parse_args()

    report_rule(args.steps, "loss")
    report_rule(args.steps, "accuracy")

    runs = collect(args.steps)
    if not runs:
        print(f"no convergence runs at {args.steps} steps; run --plan real_final first")
        return 1

    print(f"# Multi-seed convergence results ({args.steps}-step runs)\n")
    print("Accuracy at the validation-selected checkpoint. Sample standard\n"
          "deviation (ddof=1); undefined for a single seed.\n")
    print(f"{'dataset':10s} {'system':7s} {'n':>2s} {'mean':>7s} {'sd':>7s} "
          f"{'min':>6s} {'max':>6s} {'wall s':>7s}  per-seed")
    for key in sorted(runs):
        dataset, system = key
        rows = runs[key]
        accs = [a for _, a, _, _ in rows]
        walls = [w for _, _, w, _ in rows]
        per_seed = "  ".join(f"s{seed}={acc:.3f}" for seed, acc, _, _ in rows)
        sd = statistics.stdev(accs) if len(accs) > 1 else None
        print(f"{dataset:10s} {system:7s} {len(accs):>2d} "
              f"{statistics.mean(accs):>7.4f} "
              f"{(f'{sd:.4f}' if sd is not None else '-'):>7s} "
              f"{min(accs):>6.3f} {max(accs):>6.3f} "
              f"{statistics.mean(walls):>7.0f}  {per_seed}")

    # Rank only within matched seed coverage.
    print("\n## Comparison (only where every pair has the same seed count)")
    for dataset in sorted({d for d, _ in runs}):
        cells = {s: runs[(dataset, s)] for s in ("vss", "plain") if (dataset, s) in runs}
        if len(cells) != 2:
            continue
        counts = {s: len(v) for s, v in cells.items()}
        means = {s: statistics.mean(a for _, a, _, _ in v)
                 for s, v in cells.items()}
        if len(set(counts.values())) != 1:
            print(f"  {dataset:10s} SKIPPED — unequal seed counts {counts}; "
                  "a mean must not be compared against a different n")
            continue
        n = counts["vss"]
        verdict = ("tie" if abs(means["vss"] - means["plain"]) < 0.02
                   else ("VSS" if means["vss"] > means["plain"] else "plain"))
        print(f"  {dataset:10s} n={n}  VSS {means['vss']:.4f}  "
              f"plain {means['plain']:.4f}  delta {means['vss'] - means['plain']:+.4f}"
              f"  -> {verdict}")
        for s in ("vss", "plain"):
            accs = [a for _, a, _, _ in cells[s]]
            if len(accs) > 1:
                print(f"               {s:5s} sd {statistics.stdev(accs):.4f} "
                      f"spread {max(accs) - min(accs):.3f}")
    return 0


def report_rule(steps: int, rule: str) -> None:
    runs = collect(steps, rule=rule)
    if not runs:
        return
    label = ("minimum validation loss (what the trainer did)"
             if rule == "loss"
             else "maximum validation accuracy (corrected rule)")
    print(f"\n# Selection rule: {label}\n")
    for key in sorted(runs):
        dataset, system = key
        accs = [a for _, a, _, _ in runs[key]]
        sd = statistics.stdev(accs) if len(accs) > 1 else None
        per_seed = "  ".join(f"s{seed}={acc:.3f}" for seed, acc, _, _ in runs[key])
        print(f"{dataset:10s} {system:7s} n={len(accs)} "
              f"mean={statistics.mean(accs):.4f} "
              f"sd={(f'{sd:.4f}' if sd is not None else '-'):>7s}  {per_seed}")


if __name__ == "__main__":
    raise SystemExit(main())