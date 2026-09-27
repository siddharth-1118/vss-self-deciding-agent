"""Generate synthetic training/eval data for VSS.

Usage:
    python scripts/prepare_data.py --out data/generated --seed 13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vss.data.schema import dump_jsonl, validate_jsonl_file  # noqa: E402
from vss.data.synthetic import SyntheticGenerator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--train-per-stage", type=int, default=400)
    ap.add_argument("--eval-per-stage", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_gen = SyntheticGenerator(seed=args.seed)
    eval_gen = SyntheticGenerator(seed=args.seed + 1)  # disjoint seed -> disjoint draws

    counts_train = {s: args.train_per_stage for s in train_gen.STAGES}
    counts_eval = {s: args.eval_per_stage for s in eval_gen.STAGES}

    train = train_gen.generate(counts_train)
    eval = eval_gen.generate(counts_eval)

    train_path = out / "train.jsonl"
    eval_path = out / "eval.jsonl"
    dump_jsonl(train, str(train_path))
    dump_jsonl(eval, str(eval_path))

    # validate what we just wrote — generator output must pass the same
    # validator as user data [public]
    for path in (train_path, eval_path):
        valid, errors = validate_jsonl_file(str(path))
        status = "OK" if not errors else f"{len(errors)} ERRORS"
        print(f"{path}: {valid} valid ({status})")
        if errors:
            for e in errors[:5]:
                print("  -", e)
            return 1
    print(json.dumps({"train": len(train), "eval": len(eval)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
