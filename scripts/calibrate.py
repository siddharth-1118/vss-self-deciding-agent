"""Fit temperature scaling on a held-out JSONL file and report calibration.

Usage:
    python scripts/calibrate.py --model runs/prototype/final \
        --data data/generated/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from vss.data.schema import load_jsonl  # noqa: E402
from vss.model.calibration import (  # noqa: E402
    collect_choice_confidence,
    fit_temperature,
    binary_metrics_report,
)
from vss.model.vss_model import VSSModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    model = VSSModel.load_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples = load_jsonl(args.data)

    confs, correct, logits_by_row, targets = collect_choice_confidence(model, examples, device)
    if not confs:
        print(json.dumps({"error": "no choice questions found in eval data"}))
        return 1
    valid_idx = [i for i in sorted(logits_by_row) if i < len(targets) and targets[i] >= 0]
    before = binary_metrics_report(
        [confs[i] for i in valid_idx], [correct[i] for i in valid_idx]
    )

    temp = fit_temperature(logits_by_row, targets)
    # apply temperature to the SAME valid rows used for the before-report
    confs_t: list[float] = []
    correct_t: list[int] = []
    for i in valid_idx:
        probs = torch.softmax(logits_by_row[i] / temp, dim=-1)
        conf, idx = probs.max(dim=-1)
        confs_t.append(float(conf))
        correct_t.append(1 if int(idx) == targets[i] else 0)
    after = binary_metrics_report(confs_t, correct_t)

    print(json.dumps({
        "temperature": temp,
        "before": {"ece": before.ece, "brier": before.brier, "nll": before.nll, "accuracy": before.accuracy},
        "after": {"ece": after.ece, "brier": after.brier, "nll": after.nll, "accuracy": after.accuracy},
        "note": "temperature is reported; fold it into serving via inference config or head scale",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
