"""CLINC150 (clinc/clinc_oos, `imbalanced` split) -> VSS JSONL.

Source: official train/validation/test parquet shards via the HF datasets
server parquet API. Labels are intent NAME strings; raw label ids are never
serialized into the state (the label text appears only as the declared
option / gold answer).

Protocol [vss]:
  - training examples declare a SUBSET of options (true intent + k random
    distractors) to keep sequences tractable with 151 intents and to force
    dynamic-schema learning;
  - evaluation examples declare the FULL official option set, so real-data
    evaluation doubles as a dynamic-schema generalization test;
  - splits are the official ones; VSS adds nothing to train that appears
    in test.

Usage:
    python -m vss.data.clinc150 --raw-dir data/raw --out data/clinc150
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import TrainingExample, dump_jsonl

SOURCE = "clinc/clinc_oos (imbalanced)"
NUM_DISTRACTORS_DEFAULT = 14
OOS_LABEL = "oos"


def _row_to_example(
    text: str,
    gold_intent: str,
    options: list[str],
    *,
    question_id: str = "intent",
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {"message": str(text)}
    if extra_state:
        state.update(extra_state)
    return {
        "state": state,
        "questions": [
            {
                "id": question_id,
                "type": "choice",
                "options": options,
                "answer": gold_intent,
            }
        ],
    }


def convert_split(
    df: pd.DataFrame,
    label_names: list[str],
    *,
    training: bool,
    n_distractors: int = NUM_DISTRACTORS_DEFAULT,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Convert one CLINC150 split to VSS examples.

    training=True: subset option schema per example (true + k distractors).
    training=False: full official option set.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for text, lid in zip(df["text"], df["intent"]):
        gold = label_names[int(lid)]
        if training:
            pool = [l for l in label_names if l != gold]
            distractors = rng.sample(pool, min(n_distractors, len(pool)))
            options = sorted([gold] + distractors)
        else:
            options = sorted(label_names)
        out.append(_row_to_example(text, gold, options))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/clinc150")
    ap.add_argument("--n-distractors", type=int, default=NUM_DISTRACTORS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    names = json.loads((raw / "clinc_label_names.json").read_text())
    assert len(names) == 151, f"expected 151 intent names, got {len(names)}"

    train_df = pd.read_parquet(raw / "clinc_train.parquet")
    val_df = pd.read_parquet(raw / "clinc_validation.parquet")
    test_df = pd.read_parquet(raw / "clinc_test.parquet")

    train = convert_split(train_df, names, training=True, n_distractors=args.n_distractors, seed=args.seed)
    if args.max_train:
        train = train[: args.max_train]
    val = convert_split(val_df, names, training=False)
    test = convert_split(test_df, names, training=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_jsonl([TrainingExample.model_validate(e) for e in train], str(out / "train.jsonl"))
    dump_jsonl([TrainingExample.model_validate(e) for e in val], str(out / "validation.jsonl"))
    dump_jsonl([TrainingExample.model_validate(e) for e in test], str(out / "test.jsonl"))
    meta = {
        "source": SOURCE,
        "splits": {"train": len(train), "validation": len(val), "test": len(test)},
        "n_intents": len(names),
        "training_options_per_example": args.n_distractors + 1,
        "eval_options": len(names),
        "note": "training uses subset schemas; evaluation uses the full official option set",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
