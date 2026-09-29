"""Train the plain classifier (Systems A/B) on one dataset split.

Usage:
    python benchmarks/multi_question_value/train_plain.py \
        --dataset synthetic --seed 1 [--epochs 8] [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset  # noqa: E402
import plain_classifier as pc  # noqa: E402
from config import PlainConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["synthetic", "clinc150", "banking77"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<dir>/last.pt if present")
    ap.add_argument("--ckpt-root", default="runs")
    args = ap.parse_args()

    cfg = PlainConfig()
    if args.epochs:
        cfg.epochs = args.epochs
    run_dir = Path(args.ckpt_root) / f"mqv-plain-{args.dataset}-s{args.seed}"

    if args.dataset == "synthetic":
        train = dataset.load_synthetic("train")
        valid = dataset.load_synthetic("validation")
    else:
        train = dataset.load_real(args.dataset, "train")
        valid = dataset.load_real(args.dataset, "validation")

    labels = pc.build_label_inventory(train, with_abstain=True)
    # eval-side options may exceed the training inventory (e.g. CLINC 'oos'):
    labels = sorted(set(labels) | pc.build_label_inventory(valid, with_abstain=False))
    print(f"train {len(train)} valid {len(valid)} labels {len(labels)}", flush=True)

    model = pc.PlainClassifier(cfg, n_labels=len(labels), with_abstain=True)
    model.attach_labels(labels)
    pc.init_scratch_(model, args.seed)

    if args.resume and (run_dir / "last.pt").exists():
        # train_plain resumes automatically when last.pt exists
        pass

    result = pc.train_plain(model, train, valid, cfg, args.seed, str(run_dir))
    log_dir = HERE / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"plain-{args.dataset}-s{args.seed}.json"
    log.write_text(json.dumps({
        "system": "plain", "dataset": args.dataset, "seed": args.seed,
        "epochs": cfg.epochs, "batch_size": cfg.batch_size, "lr": cfg.lr,
        "weight_decay": cfg.weight_decay, "warmup_steps": cfg.warmup_steps,
        "optimizer": "AdamW", "scheduler": "warmup+cosine",
        "clip_grad_norm": cfg.clip_grad_norm,
        "n_train_examples": len(train), "n_train_rows": sum(len(e.questions) for e in train),
        "n_valid_examples": len(valid), "n_labels": len(labels),
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_loss": result["best_val"], "steps": result["steps"],
        "history": result["history"],
    }, indent=1), encoding="utf-8")
    print("wrote", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
