"""Evaluate a VSS checkpoint on a real-dataset JSONL split (choice only).

Batched, seeded subsampling for validation sets, full option schema,
honest per-set reporting into benchmarks/<name>/report.json.

Usage:
    python scripts/eval_realdata.py --model runs/clinc150/final \
        --data data/clinc150/test.jsonl --name clinc150-test \
        --n-samples 1000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from vss.api import VSS  # noqa: E402
from vss.data.schema import load_jsonl  # noqa: E402
from vss.eval.metrics import accuracy, f1_scores  # noqa: E402
from vss.model.calibration import brier_score, expected_calibration_error, nll  # noqa: E402


def _sanitize(o: Any) -> Any:
    import math

    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default="benchmarks")
    args = ap.parse_args()

    model = VSS.from_pretrained(args.model)
    model.inference_cfg.enable_abstention = True
    examples = load_jsonl(args.data)

    sampled_note = "full split"
    if args.n_samples and args.n_samples < len(examples):
        rng = random.Random(args.seed)
        examples = rng.sample(examples, args.n_samples)
        sampled_note = f"seeded subsample seed={args.seed}"

    preds: list[str] = []
    golds: list[str] = []
    confs: list[float] = []
    correct: list[int] = []
    t0 = time.perf_counter()
    n_forward = 0
    bs = args.batch_size
    for i in range(0, len(examples), bs):
        chunk = examples[i : i + bs]
        states = [ex.state for ex in chunk]
        qs = [[q.as_request() for q in ex.questions] for ex in chunk]
        results = model.decide_batch(states, qs, batch_size=bs)
        n_forward += 1
        for ex, res in zip(chunk, results):
            q = ex.questions[0]
            a = res["answers"][q.id]
            preds.append(str(a["value"]))
            golds.append(str(q.answer))
            if a["value"] == "ABSTAIN":
                confs.append(float(a.get("abstain_probability", a["confidence"])))
            else:
                confs.append(float(a["confidence"]))
            correct.append(int(a["value"] == q.answer))
    wall = time.perf_counter() - t0

    n_abstain = sum(1 for p in preds if p == "ABSTAIN")
    f1 = f1_scores(preds, golds)
    report = {
        "model": args.model,
        "data": args.data,
        "sampling": sampled_note,
        "n_examples": len(examples),
        "accuracy": accuracy(preds, golds),
        "macro_f1": f1["macro"],
        "abstain_rate": n_abstain / max(1, len(preds)),
        "ece": expected_calibration_error(confs, correct),
        "brier": brier_score(confs, correct),
        "nll": nll(confs, correct),
        "throughput_examples_per_s": len(examples) / wall,
        "batched_forward_passes": n_forward,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_threads": torch.get_num_threads(),
    }
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(_sanitize(report), indent=2))
    print(json.dumps(_sanitize(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
