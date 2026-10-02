"""Train VSS (System C) on the synthetic multi-question task.

Reuses the repository trainer (src/vss/training/trainer.py) with a cloned
qmask recipe: question_masked: true, stable RoPE positions, identical
optimizer budget as the plain classifier. Supports --resume with the
trainer's built-in partial-epoch checkpoints (survives the 600 s shell cap).

Usage:
    python benchmarks/multi_question_value/train_vss.py \
        --dataset synthetic --seed 1 [--epochs 8] [--resume]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import dataset  # noqa: E402
import torch  # noqa: E402

# measured fastest on this box (6 physical cores)
torch.set_num_threads(6)

from vss.data.schema import dump_jsonl  # noqa: E402
from vss.model.config import VSSConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402
from vss.training.trainer import Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "clinc150", "banking77"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ckpt-root", default="runs")
    args = ap.parse_args()

    base = VSSConfig.load(str(REPO / "configs" / "vss-prototype-clinc-slot-ho-qmask.yaml"))
    base.training.seed = args.seed
    if args.epochs:
        base.training.epochs = args.epochs
    run_dir = Path(args.ckpt_root) / f"mqv-vss-{args.dataset}-s{args.seed}"
    base.training.checkpoint_dir = str(run_dir)

    if args.dataset == "synthetic":
        train = dataset.load_synthetic("train")
        valid = dataset.load_synthetic("validation")
        # materialize as schema-validated JSONL so the trainer sees the same
        # validated path as every other run
        tmp = HERE / "_tmp_splits"
        tmp.mkdir(exist_ok=True)
        train_path = tmp / f"synthetic_train_s{args.seed}.jsonl"
        valid_path = tmp / f"synthetic_valid_s{args.seed}.jsonl"
        dump_jsonl(
            [__import__("vss.data.schema", fromlist=["TrainingExample"])
             .TrainingExample.model_validate(ex.to_training_dict()) for ex in train],
            str(train_path),
        )
        dump_jsonl(
            [__import__("vss.data.schema", fromlist=["TrainingExample"])
             .TrainingExample.model_validate(ex.to_training_dict()) for ex in valid],
            str(valid_path),
        )
    else:
        train_path = REPO / "data" / args.dataset / "train.jsonl"
        valid_path = REPO / "data" / args.dataset / "validation.jsonl"

    from vss.data.schema import load_jsonl

    train_ex = load_jsonl(str(train_path))
    valid_ex = load_jsonl(str(valid_path))
    # per-epoch validation slice (checkpoint-selection signal only; keeps the
    # epoch + eval inside the 600 s shell cap; benchmark.py evaluates properly)
    valid_ex = valid_ex[:100]
    print(f"train {len(train_ex)} valid {len(valid_ex)}", flush=True)

    model = VSSModel(base.model)
    print(f"parameters: {model.num_parameters():,}", flush=True)
    trainer = Trainer(model, base, train_ex, valid_ex)
    resume = str(run_dir / "last.pt") if args.resume and (run_dir / "last.pt").exists() else None
    result = trainer.fit(resume_from=resume)

    log_dir = HERE / "train_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"vss-{args.dataset}-s{args.seed}.json"
    log.write_text(json.dumps({
        "system": "vss", "dataset": args.dataset, "seed": args.seed,
        "epochs": base.training.epochs, "batch_size": base.training.batch_size,
        "lr": base.training.lr, "weight_decay": base.training.weight_decay,
        "warmup_steps": base.training.warmup_steps, "optimizer": "AdamW",
        "scheduler": base.training.scheduler,
        "clip_grad_norm": base.training.clip_grad_norm,
        "question_masked": base.model.question_masked,
        "header_only_choice": base.model.header_only_choice,
        "n_train_examples": len(train_ex),
        "n_valid_examples": len(valid_ex),
        "params": model.num_parameters(),
        "best_val_loss": result["best_loss"], "steps": result["steps"],
        "history": result["history"],
    }, indent=1), encoding="utf-8")
    print("wrote", log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
