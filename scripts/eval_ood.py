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
    tok_path = Path(path).parent / "final" / "vocab.json"
    if not tok_path.exists():
        raise SystemExit(f"vocab.json not found at {tok_path}; --model must point "
                         "at a run directory's best.pt (vocab lives in <run>/final/)")
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
    """Rank-based AUROC of P(pos scores > neg scores).

    Uses AVERAGE ranks for tied scores. Plain `argsort().argsort()` breaks ties
    by position, which is arbitrary: four identical scores gave 0.00 instead of
    the correct 0.50, i.e. a perfectly uninformative score would be reported as
    perfectly anti-predictive. Discrete confidence estimates tie constantly, so
    this materially changes the number.
    """
    all_s = np.concatenate([np.asarray(scores_pos, dtype=np.float64),
                            np.asarray(scores_neg, dtype=np.float64)])
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    order = np.argsort(all_s, kind="mergesort")
    sorted_s = all_s[order]
    # average rank (1-based) per element
    ranks_sorted = np.empty(len(all_s), dtype=np.float64)
    i = 0
    while i < len(all_s):
        j = i
        while j + 1 < len(all_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks_sorted[i:j + 1] = avg
        i = j + 1
    ranks = np.empty(len(all_s), dtype=np.float64)
    ranks[order] = ranks_sorted
    r_pos = ranks[:n_pos].sum()
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


def select_threshold(conf: np.ndarray, is_oos: np.ndarray, target: float,
                     correct: np.ndarray | None = None, n_cand: int = 40):
    """Pick an abstain threshold using ONLY the validation split.

    Among candidate thresholds that abstain on at least `target` of the OOS
    examples, take the one with the highest in-scope coverage (answer as much
    legitimate traffic as possible). The test split is never passed in.

    Returns (threshold, in_scope_coverage, oos_recall, target_feasible,
    tradeoff_rows). If no candidate reaches `target`, the most conservative
    supported point is returned with feasible=False -- the shortfall is
    reported, never papered over with an easier rule.
    """
    cand = np.quantile(conf, np.linspace(0.05, 0.99, n_cand))
    inscope = ~is_oos
    best_thr, best_cov, best_recall = None, -1.0, 0.0
    trade = []
    for thr in cand:
        keep = conf >= thr
        # Average over the OOS subset ONLY. Averaging over the whole split
        # divides by n instead of n_oos and understates recall by the class
        # ratio: it made 90% OOS recall look unreachable (0.03 vs 1.00) at a
        # checkpoint where it is comfortably attained.
        oos_recall = float((~keep)[is_oos].mean()) if is_oos.any() else 0.0
        cov = float(keep[inscope].mean()) if inscope.any() else 0.0
        kept = keep & inscope
        kept_acc = (float(np.asarray(correct, dtype=bool)[kept].mean())
                    if correct is not None and kept.any() else 0.0)
        trade.append({"threshold": round(float(thr), 4),
                      "oos_recall": round(oos_recall, 4),
                      "inscope_coverage": round(cov, 4),
                      "inscope_accuracy": round(kept_acc, 4)})
        if oos_recall >= target and cov > best_cov:
            best_thr, best_cov, best_recall = float(thr), cov, oos_recall
    feasible = best_thr is not None
    if feasible:
        return best_thr, best_cov, best_recall, True, trade
    pick = max(trade, key=lambda r: (r["oos_recall"], r["inscope_coverage"]))
    return pick["threshold"], pick["inscope_coverage"], pick["oos_recall"], False, trade


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default="benchmarks/ood")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-oos-recall", type=float, default=0.90,
                    help="validation-selected operating point must abstain on at "
                         "least this fraction of OOS examples. If unreachable the "
                         "run reports the best achievable instead of pretending.")
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

    validation_threshold, best_cov, best_recall, target_feasible, trade = \
        select_threshold(vc, v_is_oos, args.target_oos_recall, correct=v_correct)
    print(f"validation-selected abstain threshold: {validation_threshold:.4f} "
          f"(in-scope coverage {best_cov:.3f}, OOS recall {best_recall:.3f}, "
          f"target {args.target_oos_recall:.2f} feasible={target_feasible})")

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

    thr = validation_threshold
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
        "target_oos_recall": args.target_oos_recall,
        "target_feasible": target_feasible,
        "validation_tradeoff": trade[:: max(1, len(trade) // 12)],
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
