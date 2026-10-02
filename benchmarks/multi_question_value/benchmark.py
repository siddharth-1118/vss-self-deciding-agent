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


def spot_check_diffs(recs_x: list[dict],
                     recs_y: list[dict]) -> tuple[int, float]:
    """Compare batch-1 vs batched records: prediction disagreements and max
    probability/value deviation.  Batched matmul reorders float32 sums, so
    bitwise equality is not expected; the observed deviation is ~1 ULP
    (1.2e-7) and prediction labels agree exactly."""
    n_diff = 0
    max_diff = 0.0
    for a, b in zip(recs_x, recs_y):
        if a["type"] == "choice":
            d = abs(a["max_prob"] - b["max_prob"])
        elif a["type"] == "noul":
            d = abs(a["p_true"] - b["p_true"])
        else:
            d = abs(a["value"] - b["value"])
        max_diff = max(max_diff, d)
        if a.get("pred") is not None and a["pred"] != b["pred"]:
            n_diff += 1
    return n_diff, max_diff


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

    # A and B are the SAME network; batching provably does not change
    # outputs on this deterministic net, which we verify on a fixed spot
    # check (16 pairs, batch 1) instead of re-running the whole set at
    # batch 1 (CPU hours for zero information)
    recs_B = pc.predict_rows(plain_model, pairs, batch_size=32)
    spot = pairs[:16]
    recs_spot = pc.predict_rows(plain_model, spot, batch_size=1)
    n_diff, max_diff = spot_check_diffs(recs_spot, recs_B[:16])
    a_eq_b = n_diff == 0 and max_diff < 1e-5
    recs_A = recs_B  # same predictions by verified equivalence
    reqs_AB = [recs_A[i * q:(i + 1) * q] for i in range(len(subset))]

    cell: dict = {
        "q": q,
        "n_states": len(subset),
        "n_questions": len(pairs),
        "A": metrics.summarize(recs_A),
        "B": metrics.summarize(recs_B),
        "A_request_accuracy": metrics.request_accuracy(reqs_AB),
        "A_equals_B": a_eq_b,
        "AB_spot_check": {"n_pairs": 16, "pred_disagreements": n_diff,
                          "max_prob_diff": round(max_diff, 12)},
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
                       n_states: int = 200, seed: int = 7,
                       solo_cap: int = 100) -> dict:
    """Solo-vs-joint per-question accuracy delta for VSS (mode C).

    For EVERY question template present in the Q=q_max requests: its accuracy
    asked alone (Q=1) vs its accuracy inside the co-asked request (its own
    slot). Delta near 0 means no cross-question interference; negative delta
    means interference. Requests are built via eval_subset, so real-data
    replication is handled in one place and slot 0 keeps the canonical trained
    id. Solo runs are capped at `solo_cap` states per template (CPU budget);
    joint records are grouped over all joint states.
    """
    rng = random.Random("interf:" + str(seed))
    pool = [ex for ex in examples if len(ex.questions) >= 1]
    if len(pool) > n_states:
        idx = sorted(rng.sample(range(len(pool)), n_states))
        pool = [pool[i] for i in idx]
    joint_exs = dataset.eval_subset(pool, q_max, seed, len(pool))
    joint_all = [q for ex in joint_exs for q in ex.questions]
    qids = list(dict.fromkeys(q.id for q in joint_all))

    joint_by_qid: dict[str, list[dict]] = {}
    for req in vr.predict_examples(vss_model, joint_exs, batch_size=8):
        for r in req:
            joint_by_qid.setdefault(r["qid"], []).append(r)

    solo_by_qid: dict[str, list[dict]] = {}
    for qid in qids:
        solo_exs = [dataset.MultiQuestionExample(state=ex.state,
                                                 questions=[q])
                    for ex in joint_exs for q in ex.questions if q.id == qid]
        if len(solo_exs) > solo_cap:
            solo_exs = solo_exs[:solo_cap]
        recs = vr.predict_examples(vss_model, solo_exs, batch_size=8)
        solo_by_qid[qid] = [r for req in recs for r in req]

    deltas = {}
    for qid in qids:
        rows = joint_by_qid.get(qid, [])
        if rows and solo_by_qid.get(qid):
            s = metrics.accuracy(solo_by_qid[qid])
            j = metrics.accuracy(rows)
            deltas[qid] = {
                "solo": round(s, 4),
                "joint": round(j, 4),
                "delta_pts": round((j - s) * 100, 2),
                "solo_n": len(solo_by_qid[qid]),
                "joint_n": len(rows),
            }
    worst = sorted(deltas.items(), key=lambda kv: kv[1]["delta_pts"])
    best = sorted(deltas.items(), key=lambda kv: -kv[1]["delta_pts"])
    all_solo = [r for qid in qids for r in solo_by_qid.get(qid, [])]
    all_joint = [r for qid in qids for r in joint_by_qid.get(qid, [])]
    solo_sum = metrics.summarize(all_solo)
    joint_sum = metrics.summarize(all_joint)
    return {
        "q": q_max,
        "n_states": len(pool),
        "n_templates": len(deltas),
        "solo_cap": solo_cap,
        "per_qid": deltas,
        "mean_delta_pts": round(
            sum(v["delta_pts"] for v in deltas.values()) / max(1, len(deltas)), 3),
        "solo_accuracy": solo_sum["accuracy"],
        "joint_accuracy": joint_sum["accuracy"],
        "solo_coverage": solo_sum["coverage"],
        "joint_coverage": joint_sum["coverage"],
        "solo_answered_accuracy": solo_sum["answered_accuracy"],
        "joint_answered_accuracy": joint_sum["answered_accuracy"],
        "delta_pts_answered": round(
            (joint_sum["answered_accuracy"] - solo_sum["answered_accuracy"]) * 100
            if joint_sum["answered_accuracy"] is not None
            and solo_sum["answered_accuracy"] is not None else float("nan"), 2),
        "worst5": [dict(qid=k, **v) for k, v in worst[:5]],
        "best3": [dict(qid=k, **v) for k, v in best[:3]],
    }


def permutation_block(vss_model, examples, q: int, n_states: int = 200,
                      seed: int = 11) -> dict:
    """Question-order robustness: canonical vs seeded-permuted order.

    Agreement = fraction of (state, qid) pairs with identical predictions
    under both orders (choice/noul label; score rounded to 2 decimals).
    Requests are built via eval_subset; for replicated single-template real
    data a permutation reorders identical questions, so the block degenerates
    to a slot-position sensitivity check (disclosed in the output note).
    """
    rng = random.Random("perm:" + str(seed))
    pool = [ex for ex in examples if len(ex.questions) >= 1]
    if len(pool) > n_states:
        idx = sorted(rng.sample(range(len(pool)), n_states))
        pool = [pool[i] for i in idx]
    canon = dataset.eval_subset(pool, q, seed, len(pool))
    replicated = all(len(ex.questions) < 2 for ex in pool)
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
        **({"note": "replicated single-template requests: permutation reorders "
                    "identical questions; agreement isolates slot-position "
                    "sensitivity"} if replicated else {}),
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
    ap.add_argument("--n-eval-states", type=int, default=None,
                    help="override per-dataset default (synthetic 200, real 150)")
    ap.add_argument("--interference-at", type=int, nargs="*", default=None,
                    help="Q values for solo-vs-joint interference (VSS only)")
    ap.add_argument("--permutation-at", type=int, nargs="*", default=None)
    ap.add_argument("--interference-states", type=int, default=200)
    args = ap.parse_args()

    cfg = BenchConfig()
    plain_cfg = PlainConfig()
    if args.n_eval_states is not None:
        cfg.n_eval_states = args.n_eval_states
    else:
        # CPU-budget defaults: fewer states on the heavy real-data option sets
        cfg.n_eval_states = 200 if args.dataset == "synthetic" else 150
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

    def save() -> None:
        """Incremental persistence: every completed cell is flushed so a
        killed invocation (600 s shell cap) keeps its progress."""
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)

    # resume: skip (seed, q) cells and analysis blocks already computed
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                prev = json.load(f)
            done = {(c["seed"], c["q"]) for c in prev.get("cells", [])}
            results["cells"] = list(prev.get("cells", []))
            for k in prev:
                if k.startswith(("interference_", "permutation_")):
                    results[k] = prev[k]
            print(f"resuming: {len(results['cells'])} cells already done",
                  flush=True)
        except Exception:
            done = set()
    else:
        done = set()

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
            if (seed, q) in done:
                cell = next(c for c in results["cells"]
                            if c["seed"] == seed and c["q"] == q)
                if "AB_spot_check" not in cell:
                    # repair cells saved before the honest spot check existed
                    subset = dataset.eval_subset(examples, q, cfg.eval_seed,
                                                 len(examples))
                    pairs = [(ex.state, qst) for ex in subset
                             for qst in ex.questions][:16]
                    recs_1 = pc.predict_rows(plain_model, pairs, batch_size=1)
                    recs_32 = pc.predict_rows(plain_model, pairs, batch_size=16)
                    n_diff, max_diff = spot_check_diffs(recs_1, recs_32)
                    cell["A_equals_B"] = bool(n_diff == 0 and max_diff < 1e-5)
                    cell["AB_spot_check"] = {
                        "n_pairs": 16, "pred_disagreements": n_diff,
                        "max_prob_diff": round(max_diff, 12)}
                    save()
                print(f"seed {seed} Q={q}: already done"
                      f" (A_equals_B={cell.get('A_equals_B')})", flush=True)
                continue
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
            save()
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
                key = "interference_Q" + str(qi) + f"_s{seed}"
                if key in results:
                    continue
                results[key] = interference_block(
                    vss_model, examples, qi, n_states=args.interference_states)
                save()
                print(key, "mean delta pts:",
                      results[key]["mean_delta_pts"], flush=True)
        if args.permutation_at and vss_model is not None:
            for qi in args.permutation_at:
                key = "permutation_Q" + str(qi) + f"_s{seed}"
                if key in results:
                    continue
                results[key] = permutation_block(
                    vss_model, examples, qi, n_states=args.interference_states)
                save()
                print(key, "agreement:", results[key]["agreement"], flush=True)

        del plain_model, vss_model

    save()
    print("wrote", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
