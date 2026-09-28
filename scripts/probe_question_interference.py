"""Question-interference probe: does co-asking questions change any answer?

VSS's core claim is one forward pass answers a variable set of typed
questions per state. Two side effects to rule out:

  1. Interference: with causal attention, a question's answer may depend on
     questions *earlier in the request* (later ones cannot affect it). We
     compare answers asked solo (one question per forward) vs co-asked at
     every slot of a K-question request, per slot.
  2. Efficiency: wall-clock for N answers solo vs co-asked, the
     single-request batching payoff.

With header-only serialization the predicted *logits* for a question come
from its own span mean-pool; the probe measures empirically how much (if at
all) preceding question tokens shift the same question's answer.

Run:  python scripts/probe_question_interference.py \
          --model runs/clinc150-slot-ho/final --data data/clinc150/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vss.api import VSS  # noqa: E402
from vss.data.schema import load_jsonl  # noqa: E402
from vss.inference.batching import run_batch  # noqa: E402

INF = dict(abstain_threshold=0.55, enable_abstention=False, confidence_mode="blend")


def _first_choice(ex):
    for q in ex.questions:
        if q.type == "choice":
            return q
    raise ValueError("example has no choice question")


def _run(mdl, batch_ex, question_lists):
    return run_batch(mdl, [ex.state for ex in batch_ex], question_lists,
                     abstain_threshold=INF["abstain_threshold"],
                     enable_abstention=INF["enable_abstention"],
                     confidence_mode=INF["confidence_mode"], device="cpu")


def _top_prob(answer: dict) -> float:
    if answer.get("probabilities"):
        return float(answer["probabilities"].get(answer["value"], 0.0))
    return float(answer.get("confidence", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--k", type=int, default=8, help="questions co-asked per probe")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="benchmarks/interference/clinc150.json")
    args = ap.parse_args()

    vss = VSS.from_pretrained(args.model)
    mdl = vss.model
    mdl.eval()
    examples = load_jsonl(args.data)
    rng = random.Random(args.seed)
    probes = rng.sample(examples, min(args.n_samples, len(examples)))

    all_choice = [ex for ex in examples if any(q.type == "choice" for q in ex.questions)]
    # per probe: K distinct choice questions (probe's own first, random others after);
    # track each question's gold answer (from its source example) to score accuracy
    qsets: list[list[dict]] = []
    golds: list[list[str]] = []
    for ex in probes:
        own = _first_choice(ex)
        others = [o for o in rng.sample(all_choice, min(args.k + 2, len(all_choice)))
                  if _first_choice(o) is not own]
        qs = [dict(own.as_request())]
        gs = [str(own.answer)]
        for o in others:
            if len(qs) >= args.k:
                break
            qs.append(dict(_first_choice(o).as_request()))
            gs.append(str(_first_choice(o).answer))
        for i, q in enumerate(qs):
            q["id"] = f"q{i}"
        qsets.append(qs)
        golds.append(gs)
    k = min(len(qs) for qs in qsets)

    # SOLO: each question asked alone (slot-0 position in its own request)
    t0 = time.perf_counter()
    solo: list[list[dict]] = []  # per probe, per slot
    for i in range(0, len(probes), args.batch_size):
        chunk = qsets[i : i + args.batch_size]
        for slot in range(k):
            res = _run(mdl, probes[i : i + args.batch_size], [[qs[slot]] for qs in chunk])
            for j, r in enumerate(res):
                if slot == 0:
                    solo.append([])
                solo[i + j].append(r["answers"][f"q{slot}"])
    solo_s = time.perf_counter() - t0

    # JOINT: all K questions co-asked in one request
    t0 = time.perf_counter()
    joint: list[dict] = []
    for i in range(0, len(probes), args.batch_size):
        res = _run(mdl, probes[i : i + args.batch_size], qsets[i : i + args.batch_size])
        joint.extend(res)
    joint_s = time.perf_counter() - t0

    n = len(probes)
    per_slot_agree: list[float] = []
    per_slot_drift: list[float] = []
    per_slot_solo_acc: list[float] = []
    per_slot_joint_acc: list[float] = []
    per_slot_hurt: list[int] = []
    per_slot_helped: list[int] = []
    for slot in range(k):
        agree = 0
        drift = []
        s_acc = 0
        j_acc = 0
        hurt = 0
        helped = 0
        for i in range(n):
            sv = solo[i][slot]["value"]
            jv = joint[i]["answers"][f"q{slot}"]["value"]
            g = golds[i][slot]
            agree += int(sv == jv)
            drift.append(abs(_top_prob(solo[i][slot]) - _top_prob(joint[i]["answers"][f"q{slot}"])))
            s_acc += int(sv == g)
            j_acc += int(jv == g)
            if sv == g and jv != g:
                hurt += 1
            if sv != g and jv == g:
                helped += 1
        per_slot_agree.append(agree / n)
        per_slot_drift.append(sum(drift) / n)
        per_slot_solo_acc.append(s_acc / n)
        per_slot_joint_acc.append(j_acc / n)
        per_slot_hurt.append(hurt)
        per_slot_helped.append(helped)

    n_answers = n * k
    report = {
        "model": args.model,
        "data": args.data,
        "n_probes": n,
        "k_coasked": k,
        "seed": args.seed,
        "answer_agreement_solo_vs_joint_per_slot": {
            f"slot{p}": round(a, 4) for p, a in enumerate(per_slot_agree)
        },
        "mean_answer_agreement": round(sum(per_slot_agree) / k, 4),
        "accuracy_solo_per_slot": {
            f"slot{p}": round(a, 4) for p, a in enumerate(per_slot_solo_acc)
        },
        "accuracy_joint_per_slot": {
            f"slot{p}": round(a, 4) for p, a in enumerate(per_slot_joint_acc)
        },
        "accuracy_solo_slot0_own_question": round(per_slot_solo_acc[0], 4),
        "accuracy_joint_slot0_own_question": round(per_slot_joint_acc[0], 4),
        "net_slot0_accuracy_effect_of_coasking": round(
            per_slot_joint_acc[0] - per_slot_solo_acc[0], 4),
        "flips_solo_right_joint_wrong_slot0": per_slot_hurt[0],
        "flips_solo_wrong_joint_right_slot0": per_slot_helped[0],
        "SCORING_NOTE": "per-slot accuracy beyond slot0 is NOT meaningful: slots>0 ask "
                        "foreign questions about this state, whose gold answers are "
                        "unknowable; slot0 (own question on own state) is the only "
                        "gold-scored comparison. All slots are valid for agreement/drift.",
        "mean_top_prob_drift_per_slot": {
            f"slot{p}": round(d, 6) for p, d in enumerate(per_slot_drift)
        },
        "mean_top_prob_drift_overall": round(sum(per_slot_drift) / k, 6),
        "max_slot_top_prob_drift": round(max(per_slot_drift), 6),
        "solo_wall_seconds": round(solo_s, 3),
        "joint_wall_seconds": round(joint_s, 3),
        "solo_answers_per_s": round(n_answers / solo_s, 1),
        "joint_answers_per_s": round(n_answers / joint_s, 1),
        "joint_speedup_per_answer": round((n_answers / joint_s) / (n_answers / solo_s), 2),
        "note": "encoder is BIDIRECTIONAL (is_causal=False): every question "
                "attends to all others in the request, so no slot is structurally "
                "immune; slot0 comparison is solo vs its position in the co-asked "
                "request. Batching invariance (same requests, different batch) is "
                "verified separately and holds to <1e-6.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
