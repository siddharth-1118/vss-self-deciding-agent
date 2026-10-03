"""Render the per-dataset learning-rate screen from the run JSON files.

Reports the metric **at the checkpoint the run actually selected** (lowest
validation loss), not the best epoch by accuracy. Picking the best epoch by
accuracy would be a second, inconsistent selection rule applied to the
validation set, and would make the screen look better than the model that gets
shipped.

Usage:
    python benchmarks/convergence/lr_table.py [--lrs 3e-5 1e-4 3e-4 1e-3 3e-3]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "benchmarks" / "convergence" / "runs"

DEFAULT_LRS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
DATASETS = ["banking77", "clinc150"]
SYSTEMS = ["plain", "vss"]


def find_run(system: str, dataset: str, lr: float) -> str | None:
    for path in glob.glob(str(RUNS / f"{system}-{dataset}-s13-lr*-st600-screen.json")):
        try:
            if abs(json.load(open(path))["lr"] - lr) < 1e-12:
                return path
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return None


def val_loss_of(h: dict) -> float | None:
    """Validation loss for either trainer.

    plain writes `val_loss`; the VSS trainer writes `eval_loss`. Both denote the
    same quantity *within* a system. They are **not** comparable across systems:
    VSS's eval loss includes a calibration BCE and a soft-ordinal term that the
    plain loss never had. Selection is always within one system, so this is safe
    for picking an LR -- but the two columns must never be compared.
    """
    for key in ("val_loss", "eval_loss"):
        if h.get(key) is not None:
            return float(h[key])
    return None


def selected_epoch(run: dict) -> dict | None:
    """The epoch the run selected: lowest validation loss.

    Ties resolve to the earlier epoch, matching a strict-improvement rule.
    """
    history = [h for h in run.get("history", []) if val_loss_of(h) is not None]
    if not history:
        return None
    return min(history, key=lambda h: val_loss_of(h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lrs", nargs="*", type=float, default=DEFAULT_LRS)
    args = ap.parse_args()

    print(f"{'system':7s} {'dataset':10s} {'lr':>7s} {'val_loss':>9s} "
          f"{'choice_acc':>10s} {'n':>5s} {'ep':>3s} {'secs':>6s}")
    best: dict[tuple[str, str], tuple[float, float]] = {}
    for dataset in DATASETS:
        for system in SYSTEMS:
            for lr in args.lrs:
                path = find_run(system, dataset, lr)
                if path is None:
                    print(f"{system:7s} {dataset:10s} {lr:>7g} {'pending':>9s}")
                    continue
                run = json.load(open(path))
                sel = selected_epoch(run)
                if sel is None:
                    print(f"{system:7s} {dataset:10s} {lr:>7g} {'no hist':>9s}")
                    continue
                loss = val_loss_of(sel)
                acc = sel.get("choice_accuracy")
                n = sel.get("choice_n")
                print(f"{system:7s} {dataset:10s} {lr:>7g} {loss:>9.4f} "
                      f"{acc:>10.4f} {int(n):>5d} {sel['epoch']:>3d} "
                      f"{run.get('wall_seconds', 0):>6.0f}")
                key = (dataset, system)
                if key not in best or loss < best[key][0]:
                    best[key] = (loss, lr)

    print("\nselected LR per (dataset, system) -- lowest validation loss within "
          "that system (losses are NOT comparable across systems):")
    for dataset in DATASETS:
        for system in SYSTEMS:
            entry = best.get((dataset, system))
            if entry is None:
                print(f"  {dataset:10s} {system:7s} pending")
            else:
                print(f"  {dataset:10s} {system:7s} lr={entry[1]:g} "
                      f"(val_loss {entry[0]:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())