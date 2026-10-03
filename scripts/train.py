"""Train VSS from a YAML config on JSONL data.

Every invocation writes to its own run directory (never to a shared one), and
records a manifest so a result can always be traced back to a config hash, a git
revision, the data, and an explicit outcome.

Usage:
    python scripts/train.py --config configs/vss-prototype.yaml \
        --train data/generated/train.jsonl --eval data/generated/eval.jsonl \
        --out runs/my-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "convergence"))

import manifest as run_manifest  # noqa: E402
from vss.data.schema import load_jsonl  # noqa: E402
from vss.model.config import VSSConfig  # noqa: E402
from vss.model.vss_model import VSSModel  # noqa: E402
from vss.training.trainer import Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval")
    ap.add_argument("--resume")
    ap.add_argument("--out", default=None,
                    help="run directory (default: the config's checkpoint_dir). "
                         "Each run must own its own directory.")
    ap.add_argument("--force", action="store_true",
                    help="take over a run directory even if a live process owns it")
    args = ap.parse_args()

    cfg = VSSConfig.load(args.config)
    out_dir = Path(args.out) if args.out else Path(cfg.training.checkpoint_dir)

    # Refuse to train into a directory another live process is writing.
    try:
        run_manifest.claim_run_dir(out_dir, force=args.force)
    except run_manifest.RunCollision as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    train = load_jsonl(args.train)
    eval_examples = load_jsonl(args.eval) if args.eval else []
    print(f"train examples: {len(train)}, eval: {len(eval_examples)}")
    print(f"run directory:  {out_dir}")

    model = VSSModel(cfg.model)
    cfg.training.checkpoint_dir = str(out_dir)
    print(f"parameters: {model.num_parameters():,}")

    trainer = Trainer(model, cfg, train, eval_examples)
    man = run_manifest.start_manifest(
        out_dir, run_id=out_dir.name, system="vss", dataset=Path(args.train).parent.name,
        splits={"train": len(train), "eval": len(eval_examples), "test_used": False},
        model_cfg=cfg.model, training_cfg=cfg.training,
        params=model.num_parameters(), repo=ROOT,
        extra={"entrypoint": "scripts/train.py", "config_path": args.config,
               "train_path": args.train, "eval_path": args.eval})
    try:
        result = trainer.fit(resume_from=args.resume)
    except Exception as exc:
        run_manifest.finish_manifest(out_dir, man, status="failed",
                                     error=f"{type(exc).__name__}: {exc}")
        run_manifest.release_run_dir(out_dir)
        raise

    print(json.dumps(result["history"], indent=2))
    print(f"final checkpoint: {out_dir / 'final'}")
    run_manifest.finish_manifest(
        out_dir, man, status="done",
        result={"history": result["history"], "best_loss": result["best_loss"],
                "steps": result["steps"], "epochs_run": result["epochs_run"],
                "stopped_early": result["stopped_early"]},
        best_checkpoint=str(out_dir / "best.pt"))
    run_manifest.release_run_dir(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())