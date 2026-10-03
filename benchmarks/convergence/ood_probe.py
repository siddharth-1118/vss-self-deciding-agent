"""Out-of-distribution abstention probe (Blocker D).

The quick-start example exposed a suspicious pattern: an unfamiliar-but-real
support message abstained at p=0.99, while pure nonsense ("quantum flux
capacitor warp drive") received a confident 0.88 answer with abstain_p=0.000.

That is the opposite of what an abstention head should do, so this script
measures it on a proper sample instead of anecdotes. It answers one question:

    Does the abstain head detect out-of-distribution *states*, or did it
    memorise lexical cues from the training distribution?

Three evaluation sets, all disjoint from training:

    in_distribution - the held-out eval split, unchanged.
    word_scramble   - real structure, destroyed lexical content.
    foreign_topic   - fluent text about a completely different domain.

If the head is doing genuine OOD detection the abstain rate should rise
monotonically across those three sets. If it is keying on surface tokens the
scrambled and foreign sets will be answered confidently.

Usage:
    python benchmarks/convergence/ood_probe.py --model runs/smoke_verify/final
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vss import VSS  # noqa: E402
from vss.data.schema import load_jsonl  # noqa: E402

FOREIGN = [
    "the quarterly ledger shows a variance against the forecast again",
    "please rotate the TLS certificate on the edge gateway tonight",
    "the compiler rejects the template due to an ambiguous overload",
    "sheep were grazing near the north field after the storm",
    "the sourdough starter needs feeding before the weekend bake",
    "orbital decay brings the satellite down within eighteen months",
    "my knee clicks when I descend the staircase after running",
    "the jury reached a verdict after four days of deliberation",
]

SCRAMBLE_SEED = 20241003

ORDER = ["in_distribution", "word_scramble", "foreign_topic"]


def _questions(row):
    """Return a row's question list for both TrainingExample and plain dicts."""
    return row.get("questions", []) if isinstance(row, dict) else row.questions


def _options(row) -> list[str]:
    for q in _questions(row):
        if getattr(q, "type", None) == "choice":
            return list(getattr(q, "options", None) or [])
    return ["billing", "sales", "shipping", "technical"]


def scramble(message: str, rng: random.Random) -> str:
    """Keep word count and casing shape, destroy lexical content."""
    words = message.split()
    rng.shuffle(words)
    return " ".join(words)


def _choice(question_id: str, options: list[str]) -> dict:
    return {"id": question_id, "type": "choice", "options": options}


def build_sets(eval_rows, rng: random.Random) -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {name: [] for name in ORDER}

    for row in eval_rows:
        questions = _questions(row)
        if not questions or getattr(questions[0], "type", None) != "choice":
            continue
        state = dict(row.state)
        opts = _options(row)

        sets["in_distribution"].append(
            {"state": state, "questions": [_choice("department", opts)]})

        scrambled = dict(state)
        if "message" in scrambled:
            scrambled["message"] = scramble(str(scrambled["message"]), rng)
        sets["word_scramble"].append(
            {"state": scrambled, "questions": [_choice("department", opts)]})

    default_opts = _options(eval_rows[0]) if eval_rows else \
        ["billing", "sales", "shipping", "technical"]
    sets["foreign_topic"] = [
        {"state": {"message": m}, "questions": [_choice("department", default_opts)]}
        for m in FOREIGN
    ]
    return sets


def score(model: VSS, rows: list[dict]) -> dict:
    n = abstain = 0
    conf_sum = abstain_p_sum = 0.0
    for row in rows:
        ans = model.decide(state=row["state"],
                           questions=row["questions"])["answers"]["department"]
        n += 1
        if ans["value"] == "ABSTAIN":
            abstain += 1
        conf_sum += float(ans.get("confidence", 0.0))
        abstain_p_sum += float(ans.get("abstain_probability", 0.0))
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "abstain_rate": round(abstain / n, 4),
        "mean_confidence": round(conf_sum / n, 4),
        "mean_abstain_probability": round(abstain_p_sum / n, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="data/generated/eval.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.eval)
    model = VSS.from_pretrained(args.model)
    sets = build_sets(rows, random.Random(SCRAMBLE_SEED))
    report = {name: score(model, rs) for name, rs in sets.items()}

    rates = [report[name]["abstain_rate"] for name in ORDER]
    monotonic = rates[0] <= rates[1] <= rates[2]
    foreign_rate = report["foreign_topic"]["abstain_rate"]

    report["_verdict"] = {
        "abstain_rate_ordering": dict(zip(ORDER, rates)),
        "monotonic_in_ood": monotonic,
        "detects_foreign_topic": foreign_rate >= 0.50,
        "ood_detection_works": bool(monotonic and foreign_rate >= 0.50),
    }

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())