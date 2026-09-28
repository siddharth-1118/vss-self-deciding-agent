"""Latency scaling benchmark: how does answering N questions scale?

Methods:
  A  single request, all N questions co-asked   (VSS's core promise)
  B  N separate single-question requests        (classic per-question serving)
  C  N single-question requests, batched B=16   (batched classic serving)

Reported per N in {1, 5, 10, 20, 50, 100, 200}: wall-clock p50/p95/p99 over
repetitions after warmup, plus accuracy of method A vs solo gold-scored
questions (own question in slot 0).

Run:  python benchmarks/benchmark_latency.py \
          --model runs/clinc150-slot-ho/final --data data/clinc150/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vss.api import VSS  # noqa: E402
from vss.data.schema import load_jsonl  # noqa: E402
from vss.inference.batching import run_batch  # noqa: E402

INF = dict(abstain_threshold=0.55, enable_abstention=False, confidence_mode="blend")
NS = [1, 5, 10, 20, 50, 100, 200]


def _first_choice(ex):
    for q in ex.questions:
        if q.type == "choice":
            return q
    raise ValueError("no choice question")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-probes", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmarks/latency/clinc150.json")
    args = ap.parse_args()

    vss = VSS.from_pretrained(args.model)
    mdl = vss.model
    mdl.eval()
    examples = load_jsonl(args.data)
    rng = random.Random(args.seed)
    probes = rng.sample(examples, min(args.n_probes, len(examples)))
    all_choice = [ex for ex in examples if any(q.type == "choice" for q in ex.questions)]

    # build probe question sets: own question first (gold-scored), foreign after
    qsets, golds, states = [], [], []
    for ex in probes:
        own = _first_choice(ex)
        others = [o for o in rng.sample(all_choice, min(max(NS), len(all_choice)))
                  if _first_choice(o) is not own]
        qs = [dict(own.as_request())]
        for o in others:
            if len(qs) >= max(NS):
                break
            qs.append(dict(_first_choice(o).as_request()))
        for i, q in enumerate(qs):
            q["id"] = f"q{i}"
        qsets.append(qs)
        golds.append(str(own.answer))
        states.append(ex.state)

    def timed(fn, reps: int) -> dict:
        fn()  # warmup call
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        p = lambda q: ts[min(len(ts) - 1, int(q * (len(ts) - 1)))]
        return {"p50": round(p(0.50), 4), "p95": round(p(0.95), 4),
                "p99": round(p(0.99), 4), "min": round(ts[0], 4)}

    report: dict = {
        "model": args.model,
        "data": args.data,
        "n_probes": len(probes),
        "reps": args.reps,
        "batch_size_C": args.batch_size,
        "torch_threads": __import__("torch").get_num_threads(),
        "warm": {},
        "cold": {},
        "methodA_accuracy_by_N": {},
        "note": "warm = after 1 warmup call; cold = first timed call after model "
                "load (measured once at N=200). A = one co-asked request; "
                "B = N solo requests; C = N solo requests batched 16.",
    }

    # cold-start: first invocation of method A at the largest N
    t0 = time.perf_counter()
    run_batch(mdl, states[:4], [qs[:NS[-1]] for qs in qsets[:4]], device="cpu", **INF)
    report["cold"]["A_N200_first_call_s"] = round(time.perf_counter() - t0, 4)

    for N in NS:
        # --- A: one co-asked request per probe (batched across probes) --------
        def runA():
            out = []
            for i in range(0, len(probes), 4):
                out.extend(run_batch(mdl, states[i:i + 4],
                                     [qs[:N] for qs in qsets[i:i + 4]], device="cpu", **INF))
            return out

        resA = timed(runA, args.reps)

        # --- B: N solo requests per probe, sequential -------------------------
        # Measured directly for N <= B_DIRECT_MAX (sequential single-example
        # forwards are O(N * probes); larger N extrapolates from the largest
        # directly measured N and is flagged in the report.
        B_DIRECT_MAX = 50
        if N <= B_DIRECT_MAX:
            def runB():
                for i in range(len(probes)):
                    for slot in range(N):
                        run_batch(mdl, [states[i]], [[qsets[i][slot]]], device="cpu", **INF)

            resB = timed(runB, args.reps)
            resB["extrapolated"] = False
        else:
            base = report["warm"][f"N={B_DIRECT_MAX}"]["B"]
            scale = N / B_DIRECT_MAX
            resB = {k: round(v * scale, 4) for k, v in base.items()
                    if isinstance(v, (int, float))}
            resB["extrapolated"] = True

        # --- C: N solo requests, batched across probes ------------------------
        def runC():
            for slot in range(N):
                for i in range(0, len(probes), args.batch_size):
                    run_batch(mdl, states[i:i + args.batch_size],
                              [[qs[slot]] for qs in qsets[i:i + args.batch_size]],
                              device="cpu", **INF)

        resC = timed(runC, args.reps)

        # accuracy of A slot0 vs gold at this N (interference cost at scale)
        out = runA()
        acc = sum(1 for r, g in zip(out, golds) if r["answers"]["q0"]["value"] == g) / len(out)

        report["warm"][f"N={N}"] = {"A": resA, "B": resB, "C": resC,
                                    "speedup_A_vs_B_p50": round(resB["p50"] / resA["p50"], 2),
                                    "speedup_A_vs_C_p50": round(resC["p50"] / resA["p50"], 2)}
        report["methodA_accuracy_by_N"][str(N)] = round(acc, 4)
        print(f"N={N:3d}  A p50={resA['p50']:.3f}s  B p50={resB['p50']:.3f}s  "
              f"C p50={resC['p50']:.3f}s  A-acc@q0={acc:.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
