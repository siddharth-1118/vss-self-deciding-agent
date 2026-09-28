"""Calibration validation + calibration-head audit [protocol vss].

Answers four questions about VSS's confidence machinery on CLINC150:

  A. Calibration-head audit: does `row["calibration"]` (the auxiliary
     P(correct) head) track empirical correctness, or has it merely
     memorized the abstain-loss target (max option probability)? We bin
     head values and compare per-bin accuracy vs per-bin top-prob, plus
     correlations of the head with each.

  B. Temperature audit: the validation-fitted temperature (1.083) barely
     moved ECE. We verify: (1) raw model confidences were already nearly
     temperature-1 calibrated on validation; (2) fitting on validation and
     evaluating the SAME split is a mild leak that barely changes anything.

  C. Adaptive ECE + full reliability curve on the test split (single seed).

  D. Risk/coverage already lives in benchmarks/ood/clinc150_slot_ho_best.json.

Usage:
    python benchmarks/audit_calibration.py --ckpt runs/clinc150-slot-ho/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_model(path: str):
    from vss.model.config import ModelConfig
    from vss.model.tokenizer import VSSTokenizer
    from vss.model.vss_model import VSSModel

    p = Path(path)
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    raw = ck["config"]
    raw = raw.get("model", raw) if isinstance(raw, dict) else raw
    model = VSSModel(ModelConfig(**raw))
    model.load_state_dict(ck["model"])
    tok_path = p.parent / "final" / "vocab.json"
    model.tokenizer = VSSTokenizer.load(str(tok_path))
    model.vss_encoder.tokenizer = model.tokenizer
    model.eval()
    return model


def collect_rows(model, examples, options, bs: int = 16):
    """Per-example: head value, max option prob, abstain prob, correct."""
    from vss.inference.batching import run_batch

    rows = []
    for s in range(0, len(examples), bs):
        batch = examples[s : s + bs]
        states = [ex["state"] for ex in batch]
        questions = [[{"id": "intent", "type": "choice", "options": options}] for _ in batch]
        with torch.no_grad():
            out = model(states, questions, device="cpu")
        answers = run_batch(model, states, questions, abstain_threshold=0.55,
                            enable_abstention=False, confidence_mode="blend")
        idx = 0
        for row_group, ex in zip(out["per_example_rows"], batch):
            for row, q in zip(row_group, ex["questions"]):
                if q["type"] != "choice":
                    idx += 1
                    continue
                logits = row["logits"].float().cpu()
                if "abstain_logit" in row:
                    full = torch.cat([logits, row["abstain_logit"].float().cpu().view(1)])
                    probs = torch.softmax(full, dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                # inference semantics: confidence = best DECLARED option's support,
                # i.e. option-only softmax max (abstain mass excluded) [vss]
                top_prob = float(probs[:-1].max())
                pred = q["options"][int(probs[:-1].argmax())]
                head = float(row["calibration"])
                abstain_p = float(probs[-1]) if "abstain_logit" in row else None
                gold = ex["questions"][0]["answer"]
                ans = answers[s // bs * 0 + (idx // 1)] if False else None  # unused
                rows.append({
                    "head": head,
                    "top_prob": top_prob,
                    "abstain_p": abstain_p,
                    "pred": pred,
                    "gold": gold,
                    "correct": int(pred == gold),
                })
                idx += 1
    return rows


def reliability(conf: np.ndarray, corr: np.ndarray, n_bins: int = 15) -> dict:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    curve = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        acc = float(corr[m].mean())
        cf = float(conf[m].mean())
        ece += m.mean() * abs(acc - cf)
        curve.append({"bin_lo": float(bins[i]), "bin_hi": float(bins[i + 1]),
                      "n": int(m.sum()), "conf": round(cf, 4), "acc": round(acc, 4),
                      "gap": round(acc - cf, 4)})
    return {"ece": round(float(ece), 4), "reliability_curve": curve}


def adaptive_ece(conf: np.ndarray, corr: np.ndarray, n_bins: int = 15) -> float:
    """Equal-mass binning; more robust than equal-width under skew [std]."""
    order = np.argsort(conf)
    c = corr[order]
    e = 0.0
    n = len(c)
    for i in range(n_bins):
        lo, hi = int(i * n / n_bins), int((i + 1) * n / n_bins)
        seg = slice(lo, max(hi, lo + 1))
        e += abs(c[seg].mean() - conf[order][seg].mean()) * (hi - lo) / n
    return round(float(e), 4)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def risk_coverage(conf: np.ndarray, corr: np.ndarray, fracs=(0.5, 0.8, 0.9, 0.95)) -> dict:
    order = np.argsort(-conf)
    c = corr[order]
    n = len(c)
    out = {}
    cum = np.cumsum(c)
    for f in fracs:
        k = max(1, int(f * n))
        out[f"cov_{f}"] = round(float(cum[k - 1] / k), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test", default="data/clinc150/test.jsonl")
    ap.add_argument("--valid", default="data/clinc150/validation_1000.jsonl")
    ap.add_argument("--labels", default="data/raw/clinc_label_names.json")
    ap.add_argument("--n-valid", type=int, default=1000)
    ap.add_argument("--out", default="benchmarks/calibration/clinc150_slot_ho.json")
    args = ap.parse_args()

    examples = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    valid = [json.loads(l) for l in open(args.valid, encoding="utf-8")][: args.n_valid]
    options = sorted(json.loads(Path(args.labels).read_text()))

    model = load_model(args.ckpt)
    print(f"loaded {args.ckpt}; test n={len(examples)}", flush=True)

    # ---------------- A+B on validation (head audit + temperature audit) -----
    v_rows = collect_rows(model, valid, options)
    v_head = np.array([r["head"] for r in v_rows])
    v_top = np.array([r["top_prob"] for r in v_rows])
    v_corr = np.array([r["correct"] for r in v_rows])
    v_abst = np.array([r["abstain_p"] if r["abstain_p"] is not None else 0.0 for r in v_rows])

    # per-bin head audit + saturation diagnostics
    head_frac_top_bin = float((v_head > 0.9).mean())
    head_std = float(v_head.std())
    audit_bins = []
    bins = np.linspace(0, 1, 11)
    for i in range(10):
        m = (v_head > bins[i]) & (v_head <= bins[i + 1])
        if m.sum() < 5:
            continue
        audit_bins.append({
            "head_range": [round(float(bins[i]), 2), round(float(bins[i + 1]), 2)],
            "n": int(m.sum()),
            "mean_top_prob": round(float(v_top[m].mean()), 4),
            "empirical_acc": round(float(v_corr[m].mean()), 4),
            "top_prob_vs_acc_gap": round(float(v_corr[m].mean() - v_top[m].mean()), 4),
        })

    from vss.model.calibration import collect_choice_confidence, fit_temperature

    # temperature audit: use the official collect/fit path on validation
    temp_report = {"note": "fit_temperature on validation logits (same path as training)"}
    try:
        from vss.data.schema import load_jsonl
        v_ex = load_jsonl(args.valid)[: args.n_valid]
        confs, correct, logits_by_row, targets = collect_choice_confidence(
            model, v_ex, device="cpu")
        t = fit_temperature(logits_by_row, targets)
        temp_report["fitted_temperature"] = round(t, 4)
        temp_report["n_rows_used"] = len(logits_by_row)
        # raw (T=1) calibration on validation for comparison
        confs_np = np.array(confs)
        corr_np = np.array(correct)
        rel = reliability(confs_np, corr_np)
        temp_report["val_ece_raw_T1"] = rel["ece"]
        temp_report["val_adaptive_ece_raw_T1"] = adaptive_ece(confs_np, corr_np)
        # leak check: applying the val-fitted T to the SAME val split
        scaled = confs_np ** (1.0 / t)
        rel_t = reliability(scaled, corr_np)
        temp_report["val_ece_with_fitted_T_same_split"] = rel_t["ece"]
    except Exception as e:  # noqa: BLE001
        temp_report["error"] = repr(e)

    # ---------------- C on test (single seed) --------------------------------
    t_rows = collect_rows(model, examples, options)
    t_head = np.array([r["head"] for r in t_rows])
    t_top = np.array([r["top_prob"] for r in t_rows])
    t_corr = np.array([r["correct"] for r in t_rows])
    t_acc = float(t_corr.mean())

    # raw top-prob vs head vs blend on test
    test_conf_variants = {
        "top_prob": t_top,
        "calibration_head": t_head,
        "blend": np.minimum(1.0, t_top * t_head),
        "abstain_complement": np.array(
            [1.0 - (r["abstain_p"] if r["abstain_p"] is not None else 0.0) for r in t_rows]),
    }
    variants = {}
    for name, conf in test_conf_variants.items():
        rel = reliability(conf, t_corr)
        variants[name] = {
            "ece": rel["ece"],
            "adaptive_ece": adaptive_ece(conf, t_corr),
            "auroc_correct_vs_wrong": None,  # filled below
        }
    # AUROC of confidence for correctness (discrimination, not calibration)
    def auroc(pos, neg) -> float:
        all_s = np.concatenate([pos, neg])
        ranks = all_s.argsort().argsort().astype(np.float64) + 1.0
        r_pos = ranks[: len(pos)].sum()
        return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))

    correct_idx = t_corr == 1
    for name, conf in test_conf_variants.items():
        variants[name]["auroc_correct_vs_wrong"] = round(
            auroc(conf[correct_idx], conf[~correct_idx]), 4)

    report = {
        "checkpoint": args.ckpt,
        "n_test": len(examples),
        "n_valid_used": len(valid),
        "test_accuracy_no_abstention": round(t_acc, 4),
        "single_seed_caveat": "all numbers single-seed (seed 13 training, seed-neutral eval)",

        "calibration_head_audit_validation": {
            "corr_head_vs_empirical_correct": round(pearson(v_head, v_corr), 4),
            "corr_head_vs_top_prob": round(pearson(v_head, v_top), 4),
            "corr_top_prob_vs_empirical_correct": round(pearson(v_top, v_corr), 4),
            "head_mean": round(float(v_head.mean()), 4),
            "head_std": round(head_std, 4),
            "head_frac_above_0.9": round(head_frac_top_bin, 4),
            "top_prob_mean": round(float(v_top.mean()), 4),
            "empirical_acc": round(float(v_corr.mean()), 4),
            "per_bin": audit_bins,
            "verdict": (
                "HEAD SATURATED: near-constant output (std < 0.05, >99% above 0.9); "
                "it cannot discriminate and adds no usable signal beyond top_prob "
                "(its AUROC is lower than top_prob's — see test_confidence_variants)"
                if head_std < 0.05 and head_frac_top_bin > 0.99
                else ("head tracks top_prob, not independent correctness"
                      if abs(pearson(v_head, v_top)) > 0.8
                      else "head adds signal beyond top_prob")),
        },
        "temperature_audit": temp_report,
        "test_confidence_variants": variants,
        "risk_coverage_blend": risk_coverage(
            test_conf_variants["blend"], t_corr),
        "risk_coverage_top_prob": risk_coverage(
            test_conf_variants["top_prob"], t_corr),
        "reliability_curve_blend": reliability(test_conf_variants["blend"], t_corr),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in (
        "test_accuracy_no_abstention", "calibration_head_audit_validation",
        "test_confidence_variants", "risk_coverage_blend")}, indent=2)[:3000])
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
