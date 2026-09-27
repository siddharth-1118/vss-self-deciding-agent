"""Train VSS from a YAML config on JSONL data.

Usage:
    python scripts/train.py --config configs/vss-prototype.yaml \
        --train data/generated/train.jsonl --eval data/generated/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    args = ap.parse_args()

    cfg = VSSConfig.load(args.config)
    train = load_jsonl(args.train)
    eval_examples = load_jsonl(args.eval) if args.eval else []
    print(f"train examples: {len(train)}, eval: {len(eval_examples)}")

    model = VSSModel(cfg.model)
    print(f"parameters: {model.num_parameters():,}")
    trainer = Trainer(model, cfg, train, eval_examples)
    result = trainer.fit(resume_from=args.resume)
    print(json.dumps(result["history"], indent=2))
    print(f"final checkpoint: {cfg.training.checkpoint_dir}/final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
