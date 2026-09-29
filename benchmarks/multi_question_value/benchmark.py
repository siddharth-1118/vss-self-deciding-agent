"""Multi-question value benchmark: the critical experiment.

Compares, on identical states and identical per-question gold:
  A  plain classifier, one forward pass per question (sequential)
  B  plain classifier, all N questions batched into one padded forward
  C  VSS qmask, one request with all N questions (single forward)

Usage:
  python benchmarks/multi_question_value/benchmark.py --dataset synthetic \
      --questions 1 2 4 8 16 32 50 --seeds 1 2 3 \
      --plain-root runs/mqv-plain-synthetic-s{seed} \
      --vss-root runs/mqv-vss-synthetic-s{seed} \
      --out benchmarks/multi_question_value/results/synthetic.json

Methodology notes
-----------------
- Per-question gold scoring: every question is scored individually;
  request accuracy (all-questions-correct) is reported alongside.
- Identical eval states and question sets across modes at every Q
  (dataset.eval_subset; subsample seed independent of training seeds).
- Latency: warmup + measured iterations per BenchConfig; a "request" for
  mode A is N sequential single-pair forwards, for B one padded batch
  forward over N pairs, for C one VSS forward with N questions.
- Cold start (model load) is measured separately from warm inference.
- No training happens here; models must already be trained.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dataset  # noqa: E402
import metrics  # noqa: E402
import plain_classifier as pc  # noqa: E402
import vss_runner as vr  # noqa: E402
from config import QUESTION_COUNTS, SEEDS, BenchConfig, PlainConfig  # noqa: E402

RESULTS = HERE / "results"


# ----------------------------------------------------------------- utilities

def env_block() -> dict:
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        ram_gb = None
    return {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "ram_gb": ram_gb,
        "cuda": torch.cuda.is_available(),
        "precision": "fp32",
    }


def lat_stats(times_s: list[float]) -> dict:
    ts = sorted(times_s)
    n = len(ts)
    return {
        "mean_ms": round(sum(ts) / n * 1000.0, 2),
        "p50_ms": round(ts[n // 2] * 1000.0, 2),
        "p95_ms": round(ts[min(n - 1, int(n * 0.95))] * 1000.0, 2),
        "p99_ms": round(ts[min(n - 1, int(n * 0.99))] * 1000.0, 2),
    }


def _sig(r: dict):
    """Distribution signature used to verify mode A == mode B exactly."""
    if r["type"] == "choice":
        return ("c", r["qid"], r["pred"], round(r["max_prob"], 9))
    if r["type"] == "noul":
        return ("n", r["qid"], r["pred"], round(r["p_true"], 9))
    return ("s", r["qid"], round(r["value"], 9))


def load_eval_states(dataset_name: str, cfg: BenchConfig):
    if dataset_name in ("clinc150", "banking77"):
        examples = dataset.load_real(dataset_name, "test")
    else:
        examples = dataset.load_synthetic("test")
    rng = random.Random("eval:" + dataset_name + ":" + str(cfg.eval_seed))
    if len(examples) > cfg.n_eval_states:
        idx = sorted(rng.sample(range(len(examples)), cfg.n_eval_states))
        examples = [examples[i] for i in idx]
    return examples


# ---------------------------------------------------------------- latency

def latency_request_plain(model, state: dict, questions: list, mode: str,
                          warmup: int, measured: int) -> dict:
    """Warm request latency for one state with len(questions) questions.

    mode "A": sequential single-pair forwards.
    mode "B": one batched forward over all pairs of the request.
    """
    pairs = [(state, q) for q in questions]

    def once():
        if mode == "A":
            for p in pairs:
                pc.predict_rows(model, [p], batch_size=1)
        else:
            pc.predict_rows(model, pairs, batch_size=len(pairs))

    for _ in range(warmup):
        once()
    times = []
    for _ in range(measured):
        t0 = time.perf_counter()
        once()
        times.append(time.perf_counter() - t0)
    return lat_stats(times)


def latency_request_vss(model, state: dict, questions: list,
                        warmup: int, measured: int) -> dict:
    qs = [q.as_request() for q in questions]
    for _ in range(warmup):
        vr.run_batch(model, [state], [qs], **vr.default_inference_kwargs())
    times = []
    for _ in range(measured):
        t0 = time.perf_counter()
        vr.run_batch(model, [state], [qs], **vr.default_inference_kwargs())
        times.append(time.perf_counter() - t0)
    return lat_stats(times)


# -------------------------------------------------------- paired bootstrap

def paired_bootstrap_ci(recs_x: list[dict], recs_y: list[dict],
                        n_boot: int = 1000, seed: int = 0) -> dict:
    """95% CI for mean(correct_x - correct_y) over paired per-question records."""
    assert len(recs_x) == len(recs_y)
    dx = [1.0 if a["correct"] else 0.0 for a in recs_x]
    dy = [1.0 if b["correct"] else 0.0 for b in recs_y]
    diff = [a - b for a, b in zip(dx, dy)]
    n = len(diff)
    if n == 0:
        return {"ci95_low": None, "ci95_high": None, "mean_diff": None}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diff[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return {
        "mean_diff": round(sum(diff) / n, 4),
        "ci95_low": round(means[int(0.025 * n_boot)], 4),
        "ci95_high": round(means[min(n_boot - 1, int(0.975 * n_boot))], 4),
        "n": n,
    }


# ------------------------------------------------------------- eval cells

def evaluate_cell(examples, q: int, plain_model, vss_model,
                  split_seed: int = 0) -> dict:
    # states are pre-subsampled by load_eval_states; cap = len keeps them
    # identical across Q and across modes (eval_subset only prefixes questions)
    subset = dataset.eval_subset(examples, q, split_seed, len(examples))
    pairs = [(ex.state, qst) for ex in subset for qst in ex.questions]

    recs_A = pc.predict_rows(plain_model, pairs, batch_size=1)
    recs_B = pc.predict_rows(plain_model, pairs, batch_size=32)
    a_eq_b = all(_sig(a) == _sig(b) for a, b in zip(recs_A, recs_B))
    reqs_AB = [recs_A[i * q:(i + 1) * q] for i in range(len(subset))]

    cell: dict = {
        "q": q,
        "n_states": len(subset),
        "n_questions": len(pairs),
        "A": metrics.summarize(recs_A),
        "B": metrics.summarize(recs_B),
        "A_request_accuracy": metrics.request_accuracy(reqs_AB),
        "A_equals_B": a_eq_b,
    }
    if vss_model is not None:
        recs_C_req = vr.predict_examples(vss_model, subset, batch_size=8)
        flat_C = [r for req in recs_C_req for r in req]
        cell["C"] = metrics.summarize(flat_C)
        cell["C_request_accuracy"] = metrics.request_accuracy(recs_C_req)
        n_abs = sum(1 for req in recs_C_req for r in req if r.get("abstained"))
        cell["C_abstention_rate"] = round(n_abs / max(1, len(flat_C)), 4)
        cell["boot_C_minus_B"] = paired_bootstrap_ci(flat_C, recs_B, seed=q)
        cell["boot_B_minus_A"] = None  # A and B are the same model by design
    return cell


def interference_block(vss_model, examples, q_max: int,
                       n_states: int = 200, seed: int = 7) -> dict:
    """Solo-vs-joint per-question accuracy delta for VSS (mode C).

    Solo: each question template asked alone (Q=1) on its own state sample.
    Joint: the same template inside the full Q=q_max request. Delta near 0
    means no cross-question interference; negative delta means interference.
    """
    rng = random.Random("interf:" + str(seed))
    pool_examples = [ex for ex in examples if len(ex.questions) >= q_max]
    if len(pool_examples) > n_states:
        idx = sorted(rng.sample(range(len(pool_examples)), n_states))
        pool_examples = [pool_examples[i] for i in idx]
    qids = [ex.questions[i].id for i in range(q_max)
            for ex in pool_examples[:1]]
    qids = list(dict.fromkeys(qids))

    solo_acc: dict[str, float] = {}
    for qid in qids:
        solo_exs = []
        for ex in pool_examples:
            q = next(qq for qq in ex.questions if qq.id == qid)
            solo_exs.append(dataset.MultiQuestionExample(
                state=ex.state, questions=[q]))
        recs = vr.predict_examples(vss_model, solo_exs, batch_size=8)
        flat = [r for req in recs for r in req]
        solo_acc[qid] = round(metrics.accuracy(flat), 4)

    joint_exs = [dataset.MultiQuestionExample(state=ex.state,
                                              questions=ex.questions[:q_max])
                 for ex in pool_examples]
    recs_joint = vr.predict_examples(vss_model, joint_exs, batch_size=8)
    by_qid: dict[str, list[dict]] = {}
    for req in recs_joint:
        for r in req:
            by_qid.setdefault(r["qid"], []).append(r)
    deltas = {}
    for qid, rows in by_qid.items():
        if qid in solo_acc:
            joint = metrics.accuracy(rows)
            deltas[qid] = {
                "solo": solo_acc[qid],
                "joint": round(joint, 4),
                "delta_pts": round((joint - solo_acc[qid]) * 100, 2),
            }
    worst = sorted(deltas.items(), key=lambda kv: kv[1]["delta_pts"])
    return {
        "q": q_max,
        "n_states": len(pool_examples),
        "per_qid": deltas,
        "mean_delta_pts": round(
            sum(v["delta_pts"] for v in deltas.values()) / max(1, len(deltas)), 3),
        "worst5": [dict(qid=k, **v) for k, v in worst[:5]],
    }


def permutation_block(vss_model, examples, q: int, n_states: int = 200,
                      seed: int = 11) -> dict:
    """Question-order robustness: canonical vs seeded-permuted order.

    Agreement = fraction of (state, qid) pairs with identical predictions
    under both orders (choice/noul label; score rounded to 2 decimals).
    """
    rng = random.Random("perm:" + str(seed))
    pool = [ex for ex in examples if len(ex.questions) >= q]
    if len(pool) > n_states:
        idx = sorted(rng.sample(range(len(pool)), n_states))
        pool = [pool[i] for i in idx]
    canon = [dataset.MultiQuestionExample(state=ex.state,
                                          questions=ex.questions[:q])
             for ex in pool]
    perm_exs = []
    perms = []
    for ex in canon:
        order = list(range(q))
        rng.shuffle(order)
        perms.append(order)
        perm_exs.append(dataset.MultiQuestionExample(
            state=ex.state, questions=[ex.questions[i] for i in order]))
    recs_can = vr.predict_examples(vss_model, canon, batch_size=8)
    recs_perm = vr.predict_examples(vss_model, perm_exs, batch_size=8)
    same = 0
    total = 0
    for req_c, req_p, order in zip(recs_can, recs_perm, perms):
        by_id_c = {r["qid"]: r for r in req_c}
        for r_p in req_p:
            r_c = by_id_c[r_p["qid"]]
            total += 1
            if r_c["type"] == "score":
                eq = abs(r_c["value"] - r_p["value"]) < 5e-3
            else:
                eq = r_c["pred"] == r_p["pred"]
            same += 1 if eq else 0
    return {
        "q": q,
        "n_states": len(pool),
        "agreement": round(same / max(1, total), 4),
        "n_pairs": total,
    }


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["synthetic", "clinc150", "banking77"])
    ap.add_argument("--questions", type=int, nargs="+", default=list(QUESTION_COUNTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--plain-root", required=True,
                    help="checkpoint dir template, e.g. runs/mqv-plain-synthetic-s{seed}")
    ap.add_argument("--vss-root", default=None,
                    help="VSS run dir template; {seed} substituted when present")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--interference-at", type=int, nargs="*", default=None,
                    help="Q values for solo-vs-joint interference (VSS only)")
    ap.add_argument("--permutation-at", type=int, nargs="*", default=None)
    ap.add_argument("--interference-states", type=int, default=200)
    args = ap.parse_args()

    cfg = BenchConfig()
    plain_cfg = PlainConfig()
    out_path = Path(args.out) if args.out else RESULTS / (args.dataset + "_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict = {
        "dataset": args.dataset,
        "questions": args.questions,
        "seeds": args.seeds,
        "env": env_block(),
        "n_eval_states": cfg.n_eval_states,
        "latency": {"warmup_iters": cfg.warmup_iters,
                    "measured_iters": cfg.measured_iters},
        "cells": [],
    }

    examples = load_eval_states(args.dataset, cfg)
    print("eval states:", len(examples), flush=True)

    for seed in args.seeds:
        plain_dir = (args.plain_root.replace("{seed}", str(seed))
                     if "{seed}" in args.plain_root else args.plain_root)
        t0 = time.perf_counter()
        plain_model = pc.load_plain(plain_dir, plain_cfg)
        cold_plain = time.perf_counter() - t0
        vss_model = None
        cold_vss = None
        if args.vss_root:
            vss_dir = (args.vss_root.replace("{seed}", str(seed))
                       if "{seed}" in args.vss_root else args.vss_root)
            t0 = time.perf_counter()
            vss_model = vr.load_vss(vss_dir)
            cold_vss = time.perf_counter() - t0
        print(f"seed {seed}: plain={plain_dir} vss={args.vss_root} "
              f"(cold {cold_plain:.1f}s / {cold_vss and round(cold_vss, 1)}s)",
              flush=True)

        # one latency reference state per seed (fixed, split-identical)
        lat_state = examples[0]

        for q in args.questions:
            cell = evaluate_cell(examples, q, plain_model, vss_model,
                                 split_seed=cfg.eval_seed)
            cell["seed"] = seed
            cell["cold_start_s"] = {"plain": round(cold_plain, 2),
                                    "vss": round(cold_vss, 2) if cold_vss else None}
            if not args.skip_latency:
                qs = examples[0].questions[:q]
                cell["latency"] = {
                    "A": latency_request_plain(plain_model, lat_state.state, qs,
                                               "A", cfg.warmup_iters,
                                               cfg.measured_iters),
                    "B": latency_request_plain(plain_model, lat_state.state, qs,
                                               "B", cfg.warmup_iters,
                                               cfg.measured_iters),
                }
                if vss_model is not None:
                    cell["latency"]["C"] = latency_request_vss(
                        vss_model, lat_state.state, qs,
                        cfg.warmup_iters, cfg.measured_iters)
                for m in cell["latency"]:
                    p50_s = cell["latency"][m]["p50_ms"] / 1000.0
                    cell["latency"][m]["requests_per_sec_p50"] = round(1.0 / p50_s, 2)
                    cell["latency"][m]["questions_per_sec_p50"] = round(q / p50_s, 2)
            results["cells"].append(cell)
            line = (f"seed {seed} Q={q}: A {cell['A']['accuracy']:.4f} "
                    f"B {cell['B']['accuracy']:.4f}")
            if "C" in cell:
                line += f" C {cell['C']['accuracy']:.4f}"
            if "latency" in cell:
                line += (f" | p50ms A {cell['latency']['A']['p50_ms']}"
                         f" B {cell['latency']['B']['p50_ms']}")
                if "C" in cell["latency"]:
                    line += f" C {cell['latency']['C']['p50_ms']}"
            print(line, flush=True)

        if args.interference_at and vss_model is not None:
            for qi in args.interference_at:
                key = "interference_Q" + str(qi)
                results[key] = interference_block(
                    vss_model, examples, qi, n_states=args.interference_states)
                print(key, "mean delta pts:",
                      results[key]["mean_delta_pts"], flush=True)
        if args.permutation_at and vss_model is not None:
            for qi in args.permutation_at:
                key = "permutation_Q" + str(qi)
                results[key] = permutation_block(
                    vss_model, examples, qi, n_states=args.interference_states)
                print(key, "agreement:", results[key]["agreement"], flush=True)

        del plain_model, vss_model

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
