"""Banking77 (legacy-datasets/banking77) -> VSS JSONL.

Source: official train/test parquet shards via the HF datasets server
parquet API. The official validation split does not exist, so a stratified
1000-example validation set is carved out of train with a fixed seed.

Protocol mirrors clinc150.py: subset option schemas for training, full
77-option schema for evaluation (dynamic-schema generalization test).

Usage:
    python -m vss.data.banking77 --raw-dir data/raw --out data/banking77
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from .schema import TrainingExample, dump_jsonl

SOURCE = "legacy-datasets/banking77"
NUM_DISTRACTORS_DEFAULT = 14


def convert_split(
    df: pd.DataFrame,
    label_names: list[str],
    *,
    training: bool,
    n_distractors: int = NUM_DISTRACTORS_DEFAULT,
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for text, lid in zip(df["text"], df["label"]):
        gold = label_names[int(lid)]
        if training:
            pool = [l for l in label_names if l != gold]
            distractors = rng.sample(pool, min(n_distractors, len(pool)))
            options = sorted([gold] + distractors)
        else:
            options = sorted(label_names)
        out.append(
            {
                "state": {"message": str(text)},
                "questions": [
                    {"id": "intent", "type": "choice", "options": options, "answer": gold}
                ],
            }
        )
    return out


def stratified_validation(
    train_df: pd.DataFrame, n: int = 1000, seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified carve-out of a validation set from train."""
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = {}
    for idx, lid in enumerate(train_df["label"]):
        by_label.setdefault(int(lid), []).append(idx)
    per = max(1, n // len(by_label))
    val_idx: list[int] = []
    for lid, idxs in sorted(by_label.items()):
        rng.shuffle(idxs)
        val_idx.extend(idxs[:per])
    val_idx = sorted(val_idx)
    mask = train_df.index.isin(val_idx)
    return train_df[~mask], train_df[mask]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/banking77")
    ap.add_argument("--n-distractors", type=int, default=NUM_DISTRACTORS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    names = json.loads((raw / "banking_label_names.json").read_text())
    assert len(names) == 77, f"expected 77 label names, got {len(names)}"

    train_df = pd.read_parquet(raw / "banking77_train.parquet")
    test_df = pd.read_parquet(raw / "banking77_test.parquet")
    train_df, val_df = stratified_validation(train_df, n=1000, seed=7)

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
        "n_labels": len(names),
        "training_options_per_example": args.n_distractors + 1,
        "eval_options": len(names),
        "note": "official splits; validation carved from train (stratified, seed=7)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
