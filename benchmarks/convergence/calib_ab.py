"""A/B: does the EMA calibration target de-saturate the head? [D28]

D28 predicted the fix for the saturated calibration head (D5: mean 0.9916,
std 0.0134, 99.7% above 0.9). This runs the SAME dataset, seed, LR and step
budget with only `calibration_target_mode` changed, then reports the head's
output distribution and its usefulness as a P(correct) signal (AUROC).

Read-only with respect to the model: no claim is made about accuracy here.
The question is narrower -- does the head stop being a near-constant?
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "convergence"))

from vss.model.config import VSSConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402
from vss.training.trainer import Trainer  # noqa: E402

from sweep import load_splits  # noqa: E402

SEED = 13
LR = 3e-4          # the LR VSS's own screen selected for CLINC150 (D25b)
MAX_STEPS = 150    # short by design: this probes the head, not convergence
CFG_PATH = "configs/vss-prototype-clinc-slot-ho-qmask.yaml"
OUT = REPO / "benchmarks" / "convergence" / "results" / "calibration_ab.json"


def run_mode(mode: str) -> dict:
    cfg = VSSConfig.load(str(REPO / CFG_PATH))
    t = cfg.training
    t.seed = SEED
    t.lr = LR
    t.epochs = 50
    t.max_steps = MAX_STEPS
    t.early_stop_patience = 50
    t.min_epochs = 50
    t.log_every = 10_000
    t.calibration_target_mode = mode
    t.checkpoint_dir = str(REPO / "runs" / f"calib_ab_{mode}")

    train_ex, val_ex, _ = load_splits("clinc150")
    model = VSSModel(cfg.model)
    trainer = Trainer(model, cfg, train_ex, val_ex[:200])
    trainer.fit()

    # Measure the head's output distribution + usefulness on held-out data.
    vals, correct = [], []
    model.eval()
    with torch.no_grad():
        for ex in val_ex[:400]:
            states = [ex.state]
            qs = [[q.as_request() for q in ex.questions]]
            out = model(states, qs, device=trainer.device)
            from vss.training.trainer import build_targets

            tgts = build_targets(ex)
            for row, tgt in zip(out["per_example_rows"][0], tgts):
                if row["type"] != "choice":
                    continue
                p = float(row["calibration"].reshape(-1)[0])
                vals.append(p)
                logits = row["logits"].detach()
                gold = logits.shape[-1] - 1 if tgt.get("abstain") else tgt["answer_index"]
                correct.append(1 if int(torch.argmax(logits)) == gold else 0)

    auroc = _auroc(vals, correct)
    return {
        "mode": mode,
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 4) if vals else None,
        "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else None,
        "frac_above_0.9": round(sum(v > 0.9 for v in vals) / len(vals), 4) if vals else None,
        "frac_above_0.99": round(sum(v > 0.99 for v in vals) / len(vals), 4) if vals else None,
        "accuracy": round(statistics.fmean(correct), 4) if correct else None,
        "auroc_correctness": round(auroc, 4),
    }


def _auroc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


import torch  # noqa: E402

if __name__ == "__main__":
    import os
    only = os.environ.get("CALIB_MODE")
    modes = (only,) if only else ("self", "ema")
    results = []
    for mode in modes:
        print(f"=== running {mode} ===", flush=True)
        results.append(run_mode(mode))
        print(json.dumps(results[-1], indent=2), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "dataset": "clinc150",
        "seed": SEED,
        "lr": LR,
        "max_steps": MAX_STEPS,
        "note": "single seed; probes head saturation only, not accuracy ranking",
        "results": results,
    }, indent=2))
    print(f"\nwrote {OUT}")