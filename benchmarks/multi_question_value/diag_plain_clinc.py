"""Diagnose the CLINC150 plain-baseline collapse.

The 1200-step sweep run collapsed to 0.000 choice accuracy while a pre-audit
checkpoint of the same architecture, trained on the same data for a longer
schedule, reached validation loss 0.395. This script localises the difference:

  1. evaluate the HISTORICAL checkpoint with the CURRENT evaluation code path
     (same label inventory, same 200-example validation slice);
  2. evaluate a freshly initialised model for reference.

If (1) scores well, the collapse is in training for this schedule. If (1) also
scores ~0, the problem is in the evaluation/label-mapping setup instead.

Usage:
    python benchmarks/multi_question_value/diag_plain_clinc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "multi_question_value"))
sys.path.insert(0, str(REPO / "benchmarks" / "convergence"))

import torch  # noqa: E402

import plain_classifier as pc  # noqa: E402
from config import PlainConfig  # noqa: E402
from sweep import VAL_SLICE, load_splits  # noqa: E402


def batches(rows, bs):
    """Same contiguous chunking as train_plain's inner helper (validation never
    shuffles)."""
    for s in range(0, len(rows), bs):
        yield rows[s : s + bs]

HIST_CANDIDATES = [
    REPO / "runs" / "mqv-plain-clinc150-s13" / "best.pt",
    REPO / "runs" / "mqv-plain-clinc150-s13_maskedce_v1" / "best.pt",
    REPO / "runs" / "mqv-plain-banking77-s13" / "best.pt",
]
DATASETS = ["clinc150", "clinc150", "banking77"]


def fit_tokenizer(model, train_ex, cfg, vocab_path=None):
    """Fit on training texts -- or, when evaluating a saved checkpoint, load the
    tokenizer that was SAVED alongside it.

    This matters: `load_plain()` loads `vocab.json` from the checkpoint dir. If
    a freshly fitted tokenizer is used instead, token ids are permuted relative
    to the trained embedding matrix and the model produces confidently wrong
    predictions while still loading with zero shape mismatches.
    """
    from vss.model.serialize import serialize_example
    from vss.model.tokenizer import VSSTokenizer

    if vocab_path is not None and Path(vocab_path).exists():
        model.tokenizer = VSSTokenizer.load(str(vocab_path))
        return model
    if model.tokenizer is None:
        tok = VSSTokenizer(vocab_size=cfg.vocab_size, hash_buckets=cfg.hash_buckets)
        texts = []
        for ex in train_ex:
            for q in ex.questions:
                req = q.as_request()
                if getattr(q, "header_only_choice", False):
                    req = {**req, "header_only_choice": True}
                texts.append(serialize_example(ex.state, [req]))
        tok.fit(texts[:20000])
        model.tokenizer = tok
    return model


def evaluate(model, val_ex, labels) -> dict[str, float]:
    """Mirror train_plain's validation path exactly (same masking, same metric)."""
    cfg = PlainConfig()
    device = "cpu"
    bins = cfg.score_bins
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    valid_rows = [(ex.state, q) for ex in val_ex for q in ex.questions]
    model.eval()
    tot, n = 0.0, 0
    n_choice = n_correct = 0
    has_abstain = label_to_idx.get(pc.ABSTAIN) is not None
    with torch.no_grad():
        for chunk in batches(valid_rows, cfg.batch_size):
            x, _ = model.encode_rows(chunk, device)
            out = model(x)
            loss = pc._batch_loss(out, chunk, label_to_idx, bins, device)
            tot += float(loss) * len(chunk)
            n += len(chunk)
            for i, (_state, q) in enumerate(chunk):
                if q.type == "choice":
                    opts = list(q.options or ())
                    allowed = [label_to_idx[o] for o in opts]
                    names = list(opts)
                    if has_abstain:
                        allowed.append(label_to_idx[pc.ABSTAIN])
                        names.append(pc.ABSTAIN)
                    probs = torch.softmax(out["choice"][i, allowed], dim=-1)
                    pred = names[int(torch.argmax(probs))]
                    n_choice += 1
                    n_correct += int(pred == q.answer)
    return {"val_loss": tot / max(1, n),
            "choice_accuracy": n_correct / max(1, n_choice)}


def main() -> int:
    cfg = PlainConfig()
    cache: dict[str, tuple] = {}
    for ckpt_path, ds in zip(HIST_CANDIDATES, DATASETS):
        if not ckpt_path.exists():
            print(f"[skip] {ckpt_path} missing")
            continue
        if ds not in cache:
            train_ex, val_ex, _ = load_splits(ds)
            val = val_ex[:VAL_SLICE]
            labels = pc.build_label_inventory(train_ex, with_abstain=True)
            labels = sorted(set(labels) | set(
                pc.build_label_inventory(val, with_abstain=False)))
            cache[ds] = (train_ex, val, labels)
        train_ex, val, labels = cache[ds]
        name = f"{ds} / {ckpt_path.parent.name}"

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        m = pc.PlainClassifier(cfg, n_labels=len(labels), with_abstain=True)
        fit_tokenizer(m, train_ex, cfg,
                      vocab_path=ckpt_path.parent / "vocab.json")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        m.attach_labels(labels)
        res = evaluate(m, val, labels)
        print(f"{name}: n_labels={len(labels)} epoch={state.get('epoch')} "
              f"missing={len(missing)} unexpected={len(unexpected)} "
              f"val_loss={res['val_loss']:.4f} acc={res['choice_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())