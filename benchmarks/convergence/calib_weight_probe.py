"""Calibration-weight instability probe [early diagnostic, 400 steps].

Tests whether the calibration loss term's weight contributes to VSS's measured
seed instability on CLINC150 (ledger D28: VSS sd 0.052 vs plain 0.006).

Matrix: calibration weight in {1.0, 0.0} x seeds {7, 13, 21}. Everything else
is held fixed: same split, config, LR, batch size, step budget, evaluation.

Deliberately NOT a convergence study. 400 steps is roughly 2 epochs on
CLINC150; the only claims licensed here are about early trajectory and
seed-to-seed spread, never about final accuracy.

Resume: each cell writes its own JSON. A cell whose JSON exists and carries
`"status": "complete"` is skipped. A cell that dies mid-run leaves no JSON and
is re-run from scratch -- incomplete runs are never reported as complete.

Usage:
    python benchmarks/convergence/calib_weight_probe.py                # all six
    python benchmarks/convergence/calib_weight_probe.py --only w1.0-s7  # one cell
    python benchmarks/convergence/calib_weight_probe.py --analyze      # summarize
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "convergence"))

from vss.model.config import VSSConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402
from vss.training.trainer import Trainer, build_targets  # noqa: E402
from vss.model.calibration import expected_calibration_error, brier_score  # noqa: E402

from sweep import load_splits  # noqa: E402

CFG_PATH = REPO / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml"
OUT_DIR = REPO / "benchmarks" / "convergence" / "results" / "calib_weight_probe"
MAX_STEPS = 400
VAL_SLICE = 200          # matches the 2000-step protocol's validation slice
WEIGHTS = [1.0, 0.0]
SEEDS = [7, 13, 21]


def cell_id(w: float, seed: int) -> str:
    return f"w{w}-s{seed}"


def run_cell(w: float, seed: int) -> dict:
    cid = cell_id(w, seed)
    out_path = OUT_DIR / f"{cid}.json"
    # Unique run dir per cell; never touches existing checkpoints.
    run_dir = OUT_DIR / f"runs_{cid}"

    cfg = VSSConfig.load(str(CFG_PATH))
    t = cfg.training
    t.seed = seed
    t.lr = 3e-4               # the LR VSS's own CLINC150 screen selected (D25b)
    t.epochs = 50
    t.max_steps = MAX_STEPS
    t.min_epochs = 50
    t.early_stop_patience = 50
    t.log_every = 10 ** 9
    t.checkpoint_dir = str(run_dir)
    # THE ONLY VARIABLE. loss_weights is a plain dict read by combined_loss
    # as weights.get("calibration", 1.0), so this sets the calibration term's
    # contribution and nothing else.
    t.loss_weights = {"choice": 1.0, "noul": 1.0, "score": 1.0, "calibration": w}
    t.calibration_target_mode = "self"   # default path; the EMA track is separate

    train_ex, val_ex, _ = load_splits("clinc150")
    torch.manual_seed(seed)
    model = VSSModel(cfg.model)
    trainer = Trainer(model, cfg, train_ex, val_ex[:VAL_SLICE])

    started = time.time()
    global_step, losses, stats = trainer.train_epoch(
        train_ex, 0, 0, skip_batches=0
    )
    elapsed = time.time() - started

    metrics = evaluate(trainer, val_ex)
    record = {
        "status": "complete",
        "cell": cid,
        "calibration_weight": w,
        "seed": seed,
        "max_steps": MAX_STEPS,
        "lr": t.lr,
        "batch_size": t.batch_size,
        "steps_completed": global_step,
        "val_slice": VAL_SLICE,
        "elapsed_sec": round(elapsed, 1),
        "train_loss_mean": round(statistics.fmean(losses), 4) if losses else None,
        "train_loss_last": round(losses[-1], 4) if losses else None,
        # train_epoch already returns the epoch means, so use them directly
        # rather than re-averaging (grad_norms is per-step; this is a mean).
        "grad_norm_mean": round(float(stats["grad_norm_mean"]), 4),
        "grad_norm_max": round(float(stats["grad_norm_max"]), 4),
        "param_update_rel_mean": round(float(stats["param_update_rel_mean"]), 6),
        "n_updates": int(stats["n_updates"]),
        "train_components": {
            k.replace("train_", ""): round(float(v), 4)
            for k, v in stats.items() if k.startswith("train_")
        },
        **metrics,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    return record


@torch.no_grad()
def evaluate(trainer: Trainer, val_ex) -> dict:
    """Validation accuracy / macro-F1 / ECE / Brier on the choice rows.

    Macro-F1 is computed over the argmax choice only (the abstain logit is
    excluded from the label space), so it stays comparable across cells.
    """
    model = trainer.model
    model.eval()
    confs, correct, preds, golds = [], [], [], []
    for ex in val_ex[:VAL_SLICE]:
        out = model([ex.state], [[q.as_request() for q in ex.questions]],
                    device=trainer.device)
        tgts = build_targets(ex)
        for row, tgt in zip(out["per_example_rows"][0], tgts):
            if row["type"] != "choice":
                continue
            logits = row["logits"].detach()
            gold = logits.shape[-1] - 1 if tgt.get("abstain") else int(tgt["answer_index"])
            p = torch.softmax(logits, dim=-1)
            pred = int(torch.argmax(logits))
            confs.append(float(p[pred]))
            correct.append(1 if pred == gold else 0)
            preds.append(pred)
            golds.append(gold)

    return {
        "val_n": len(preds),
        "val_accuracy": round(sum(correct) / len(correct), 4) if correct else None,
        "val_macro_f1": round(macro_f1(preds, golds), 4) if preds else None,
        "val_ece": round(expected_calibration_error(confs, correct), 4) if confs else None,
        "val_brier": round(brier_score(confs, correct), 4) if confs else None,
        "val_mean_confidence": round(statistics.fmean(confs), 4) if confs else None,
    }


def macro_f1(preds: list[int], golds: list[int]) -> float:
    """Unweighted mean F1 over classes present in gold."""
    labels = sorted(set(golds))
    if not labels:
        return 0.0
    scores = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return statistics.fmean(scores)


def analyze() -> dict:
    rows = []
    for w in WEIGHTS:
        for s in SEEDS:
            p = OUT_DIR / f"{cell_id(w, s)}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            if d.get("status") == "complete":
                rows.append(d)

    summary = {"cells_found": len(rows), "by_weight": {}}
    for w in WEIGHTS:
        sub = [r for r in rows if r["calibration_weight"] == w]
        if not sub:
            continue
        accs = [r["val_accuracy"] for r in sub if r["val_accuracy"] is not None]
        f1s = [r["val_macro_f1"] for r in sub if r["val_macro_f1"] is not None]
        eces = [r["val_ece"] for r in sub if r["val_ece"] is not None]
        briers = [r["val_brier"] for r in sub if r["val_brier"] is not None]
        tl = [r["train_loss_mean"] for r in sub if r["train_loss_mean"] is not None]
        gn = [r["grad_norm_mean"] for r in sub if r["grad_norm_mean"] is not None]
        summary["by_weight"][str(w)] = {
            "n_seeds": len(sub),
            "seeds": sorted(r["seed"] for r in sub),
            "acc_mean": round(statistics.fmean(accs), 4) if accs else None,
            "acc_sd": round(statistics.pstdev(accs), 4) if len(accs) > 1 else 0.0,
            "acc_per_seed": {str(r["seed"]): r["val_accuracy"] for r in sub},
            "f1_mean": round(statistics.fmean(f1s), 4) if f1s else None,
            "ece_mean": round(statistics.fmean(eces), 4) if eces else None,
            "brier_mean": round(statistics.fmean(briers), 4) if briers else None,
            "train_loss_mean": round(statistics.fmean(tl), 4) if tl else None,
            "grad_norm_mean": round(statistics.fmean(gn), 4) if gn else None,
        }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.analyze:
        print(json.dumps(analyze(), indent=2))
        sys.exit(0)

    targets = [(w, s) for w in WEIGHTS for s in SEEDS]
    if args.only:
        w, s = args.only.split("-")
        targets = [(float(w[1:]), int(s[1:]))]

    for w, s in targets:
        cid = cell_id(w, s)
        p = OUT_DIR / f"{cid}.json"
        if p.exists() and json.loads(p.read_text()).get("status") == "complete":
            print(f"SKIP {cid} (already complete)", flush=True)
            continue
        print(f"RUN {cid} ...", flush=True)
        try:
            rec = run_cell(w, s)
            print(f"DONE {cid} acc={rec['val_accuracy']} f1={rec['val_macro_f1']} "
                  f"ece={rec['val_ece']} gnorm={rec['grad_norm_mean']} "
                  f"{rec['elapsed_sec']}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            # Leave no JSON behind: an incomplete cell must re-run, never count.
            print(f"FAIL {cid}: {type(exc).__name__}: {exc}", flush=True)
            raise
    print("\n" + json.dumps(analyze(), indent=2))