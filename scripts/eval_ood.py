"""OOD / abstention evaluation on real data.

CLINC150's official test split contains 4500 in-scope and 1000 out-of-scope
(oos) utterances. This script measures [protocol vss]:

  - in-scope accuracy under the FULL official option schema (no abstention);
  - whether model confidence separates in-scope from OOS (AUROC);
  - selective risk / coverage: accuracy if we answer the top-c fraction of
    examples ranked by confidence, for c in [0, 1];
  - abstention quality at a threshold selected ON THE VALIDATION SPLIT
    (never on test).

Usage:
    python scripts/eval_ood.py --model runs/clinc150-slot/best \
        --test data/clinc150/test.jsonl --valid data/clinc150/validation.jsonl \
        --labels data/raw/clinc_label_names.json --name clinc150_slot
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def load_model(path: str):
    sys.path.insert(0, "src")
    from vss.model.vss_model import VSSModel
    from vss.model.config import ModelConfig
    from vss.model.tokenizer import VSSTokenizer

    ck = torch.load(path + ".pt" if not path.endswith(".pt") else path,
                    map_location="cpu", weights_only=False)
    raw = ck["config"]
    raw = raw.get("model", raw) if isinstance(raw, dict) else raw
    model = VSSModel(ModelConfig(**raw))
    model.load_state_dict(ck["model"])
    vocab = Path("runs") / "tmp_vocab.json"  # fallback; prefer sibling dir
    tok_path = Path(path).parent / "final" / "vocab.json"
    model.tokenizer = VSSTokenizer.load(str(tok_path))
    model.vss_encoder.tokenizer = model.tokenizer
    model.eval()
    return model


def collect(model, examples, options):
    """One confidence per example under the full schema (no abstention)."""
    from vss.inference.batching import run_batch

    confs, preds, golds = [], [], []
    B = 16
    for s in range(0, len(examples), B):
        batch = examples[s : s + B]
        states = [ex["state"] for ex in batch]
        questions = [[{"id": "intent", "type": "choice", "options": options}] for _ in batch]
        answers = run_batch(model, states, questions, abstain_threshold=0.55,
                            enable_abstention=False, confidence_mode="blend")
        for a, ex in zip(answers, batch):
            row = a["answers"]["intent"]
            confs.append(float(row["confidence"]))
            preds.append(str(row["value"]))
            golds.append(str(ex["questions"][0]["answer"]))
    return np.array(confs), preds, golds


def auroc(scores_pos, scores_neg) -> float:
    """Rank-based AUROC of P(pos scores > neg scores)."""
    all_s = np.concatenate([scores_pos, scores_neg])
    ranks = all_s.argsort().argsort().astype(np.float64) + 1.0
    r_pos = ranks[: len(scores_pos)].sum()
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def risk_coverage(confs, correct) -> list[tuple[float, float]]:
    """(coverage, selective_accuracy) for coverage in (0, 1]."""
    order = np.argsort(-confs)
    c = np.asarray(correct, dtype=bool)[order]
    n = len(c)
    cum_correct = np.cumsum(c)
    ks = np.arange(1, n + 1)
    cov = ks / n
    sel_acc = cum_correct / ks
    return list(zip(cov.tolist(), sel_acc.tolist()))


def ece_at(conf_bin_pairs, n_bins: int = 15) -> float:
    confs = np.array([c for c, _ in conf_bin_pairs])
    corr = np.array([x for _, x in conf_bin_pairs])
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (confs > bins[i]) & (confs <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(corr[m].mean() - confs[m].mean())
    return float(e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="benchmarks/ood")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    options = sorted(json.loads(Path(args.labels).read_text()))
    oos = "oos"
    test = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    valid = [json.loads(l) for l in open(args.valid, encoding="utf-8")]

    model = load_model(args.model)
    print(f"model: {args.model} | options: {len(options)} | test n={len(test)}", flush=True)

    vc, vp, vg = collect(model, valid, options)
    v_is_oos = np.array([g == oos for g in vg])
    v_correct = np.array([p == g for p, g in zip(vp, vg)])

    # threshold selected on VALIDATION: maximize selective accuracy on
    # in-scope validation examples subject to abstaining on >=90% of OOS
    cand = np.quantile(vc, np.linspace(0.05, 0.99, 40))
    best_thr, best_cov = None, -1.0
    for thr in cand:
        keep = vc >= thr
        oos_caught = float((v_is_oos & ~keep).mean()) if v_is_oos.any() else 0.0
        inscope = ~v_is_oos
        cov = float((keep & inscope).sum() / max(1, inscope.sum()))
        if oos_caught >= 0.90 and cov > best_cov:
            best_thr, best_cov = float(thr), cov
    if best_thr is None:  # fall back: max accuracy point on validation
        accs = [((vc >= t)[~v_is_oos] & v_correct[~v_is_oos]).mean() for t in cand]
        best_thr = float(cand[int(np.argmax(accs))])
    print(f"validation-selected abstain threshold: {best_thr:.4f} (in-scope coverage {best_cov:.3f})")

    tc, tp, tg = collect(model, test, options)
    t_is_oos = np.array([g == oos for g in tg])
    t_correct = np.array([p == g for p, g in zip(tp, tg)])

    ins = ~t_is_oos
    inscope_acc = float(t_correct[ins].mean())
    # OOS separation: in-scope examples should score HIGHER -> pos = in-scope
    sep = auroc(tc[ins], tc[t_is_oos])

    rc = risk_coverage(tc[ins], t_correct[ins])
    rc_points = [{"coverage": round(c, 4), "selective_accuracy": round(a, 4)}
                 for c, a in rc[:: max(1, len(rc) // 20)]]

    thr = best_thr
    abstain = tc < thr
    answered = ~abstain
    sel_acc = float(t_correct[answered & ins].mean()) if (answered & ins).any() else None
    oos_recall = float((~answered[t_is_oos]).mean())
    coverage = float(answered[ins].mean())

    # calibration on answered in-scope examples
    cal = [(float(c), int(x)) for c, x in zip(tc[answered & ins], t_correct[answered & ins])]

    out = {
        "model": args.model,
        "protocol": "full official option schema; threshold from validation split; seed 42",
        "n_test": len(test),
        "n_inscope": int(ins.sum()),
        "n_oos": int(t_is_oos.sum()),
        "inscope_accuracy_full_schema": round(inscope_acc, 4),
        "oos_detection_auroc": round(sep, 4),
        "validation_threshold": round(thr, 4),
        "coverage_inscope": round(coverage, 4),
        "selective_accuracy_inscope": round(sel_acc, 4) if sel_acc is not None else None,
        "oos_abstention_recall": round(oos_recall, 4),
        "false_abstention_rate_inscope": round(1 - coverage, 4),
        "ece_answered_inscope": round(ece_at(cal), 4),
        "risk_coverage": rc_points,
    }
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{args.name}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
