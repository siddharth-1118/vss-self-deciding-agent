"""CLINC150 training diagnostic — why does validation accuracy degrade?

The three-seed runs show a consistent shape on CLINC150: choice accuracy peaks
around epoch 1-3 and then falls while the plain baseline keeps improving, and
validation *loss* is non-monotonic in a way accuracy is not (finding 7).

Neither existing run artifact could explain it, because the trainer recorded
only the summed loss. `combined_loss` returns a per-component split
(choice / noul / score / ordinal / calibration) and the trainer discarded
everything except the total; `evaluate_detailed` folded the same terms into one
scalar per row.

This script trains the same configuration as the sweep at the screened learning
rate, with the component breakdown now recorded, and prints the epoch table. It
is a diagnostic, not a benchmark: one seed, and it writes to its own directory so
it cannot disturb the recorded runs.

Usage:
    python benchmarks/convergence/clinc_diag.py --steps 2000 --seed 13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--lr", type=float, default=None,
                    help="defaults to the screened CLINC150/VSS LR")
    ap.add_argument("--val-slice", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "runs" / "clinc_diag"))
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    sys.path.insert(0, str(ROOT / "benchmarks" / "convergence"))
    import sweep  # load_splits / PATIENCE / MIN_EPOCHS, single source of truth
    from vss.model.config import VSSConfig
    from vss.model.vss_model import VSSModel
    from vss.training.trainer import Trainer

    train_ex, val_ex, cal_ex = sweep.load_splits("clinc150")
    chosen = sweep.selected_lr_per_dataset().get(("clinc150", "vss"))
    lr = args.lr if args.lr is not None else chosen
    if lr is None:
        raise SystemExit("no screened CLINC150/VSS LR found; pass --lr")

    cfg = VSSConfig.load(str(ROOT / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml"))
    t = cfg.training
    t.seed = args.seed
    t.lr = lr
    t.epochs = sweep.EPOCH_CAP_REAL
    t.max_steps = args.steps
    t.early_stop_patience = sweep.PATIENCE
    t.early_stop_min_delta = sweep.MIN_DELTA
    t.min_epochs = sweep.MIN_EPOCHS
    t.log_every = 10_000
    t.checkpoint_dir = args.out

    model = VSSModel(cfg.model)
    print(f"train={len(train_ex)} val={len(val_ex)} lr={lr:g} steps={args.steps} "
          f"seed={args.seed} params={model.num_parameters():,}", flush=True)
    trainer = Trainer(model, cfg, train_ex, val_ex[: args.val_slice])
    res = trainer.fit()

    hist = [h for h in res["history"] if h.get("choice_accuracy") is not None]
    comps = sorted({k for h in hist for k in h if k.startswith("comp_")})
    tr_comps = sorted({k for h in hist for k in h if k.startswith("train_")})
    print("\nepoch   lr      acc     " + "  ".join(c.replace("comp_", "")[:9].rjust(9)
                                                  for c in comps))
    for h in hist:
        print(f"{h['epoch']:<6d} {h['lr']:.2e}  {float(h['choice_accuracy']):.3f}   "
              + "  ".join(f"{h.get(c, float('nan')):9.4f}" for c in comps))
    print("\ntrain-side components")
    print("epoch  " + "  ".join(c.replace("train_", "")[:9].rjust(9) for c in tr_comps))
    for h in hist:
        print(f"{h['epoch']:<6d}" + "  ".join(f"{h.get(c, float('nan')):9.4f}" for c in tr_comps))

    best = max(hist, key=lambda h: float(h["choice_accuracy"]))
    print(f"\nbest accuracy {float(best['choice_accuracy']):.3f} at epoch {best['epoch']}; "
          f"steps={res['steps']} epochs={res['epochs_run']} early={res['stopped_early']}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(
        json.dumps({"lr": lr, "seed": args.seed, "steps": args.steps,
                    "history": hist}, indent=1), encoding="utf-8")
    print(f"wrote {out / 'history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())