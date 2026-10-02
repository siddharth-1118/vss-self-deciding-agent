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

import torch  # noqa: E402

# measured fastest on this box (6 physical cores): see train_logs env notes
torch.set_num_threads(6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["synthetic", "clinc150", "banking77"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<dir>/last.pt if present")
    ap.add_argument("--ckpt-root", default="runs")
    ap.add_argument("--save-every", type=int, default=250)
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
    # per-epoch validation slice: full splits are the benchmark's job; the
    # training loop only needs a stable checkpoint-selection signal and must
    # finish an epoch inside the 600 s shell cap (checkpointing resumes the
    # rest, but per-epoch eval repeats every resume)
    valid = valid[:200]

    labels = pc.build_label_inventory(train, with_abstain=True)
    # eval-side options may exceed the training inventory (e.g. CLINC 'oos'):
    labels = sorted(set(labels) | set(pc.build_label_inventory(valid, with_abstain=False)))
    print(f"train {len(train)} valid {len(valid)} labels {len(labels)}", flush=True)

    model = pc.PlainClassifier(cfg, n_labels=len(labels), with_abstain=True)
    model.attach_labels(labels)
    pc.init_scratch_(model, args.seed)

    if args.resume and (run_dir / "last.pt").exists():
        # train_plain resumes automatically when last.pt exists
        pass

    result = pc.train_plain(model, train, valid, cfg, args.seed, str(run_dir),
                            save_every=args.save_every)
    log_dir = HERE / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"plain-{args.dataset}-s{args.seed}.json"
    # merge with any existing log so a finalization-only resume (start_epoch
    # past the last epoch -> empty history) doesn't clobber earlier epochs
    merged: list = []
    if log.exists():
        try:
            merged = json.loads(log.read_text(encoding="utf-8")).get("history", [])
        except Exception:
            merged = []
    seen_epochs = {h.get("epoch") for h in merged}
    merged += [h for h in result["history"] if h.get("epoch") not in seen_epochs]
    merged.sort(key=lambda h: h.get("epoch", -1))
    result["history"] = merged
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
        "history": merged,
    }, indent=1), encoding="utf-8")
    print("wrote", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
