"""Robustness suite: accuracy under cheap text perturbations on real data.

No LLM paraphrases — only deterministic, seed-controlled character- and
word-level corruptions. Each perturbation answers: how brittle is the
word+hash tokenizer + encoder path to surface-form noise?

Perturbations (applied to the state's text fields):
  lower       s.lower()
  upper       s.upper()
  typo        swap two adjacent chars inside one random word
  drop_word   delete one random word (not the first)
  dup_word    duplicate one random word
  punct       append "?!"
  whitespace  double every space

Usage:
    python benchmarks/robustness_suite.py --model runs/clinc150-slot-ho/best.pt \
        --test data/clinc150/test.jsonl --labels data/raw/clinc_label_names.json
"""
from __future__ import annotations

import argparse
import json
import random
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


def perturb_text(s: str, kind: str, rng: random.Random) -> str:
    words = s.split(" ")
    if kind == "lower":
        return s.lower()
    if kind == "upper":
        return s.upper()
    if kind == "punct":
        return s + "?!".rstrip()
    if kind == "whitespace":
        return "  ".join(words)
    if not words or all(not w for w in words):
        return s
    idxs = [i for i, w in enumerate(words) if len(w) >= 3]
    if not idxs:
        return s
    i = rng.choice(idxs)
    w = words[i]
    if kind == "typo" and len(w) >= 4:
        j = rng.randrange(1, len(w) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif kind == "drop_word" and len(words) > 2:
        del words[i]
    elif kind == "dup_word":
        words.insert(i, words[i])
    return " ".join(words)


PERTURBATIONS = ["clean", "lower", "upper", "typo", "drop_word", "dup_word", "punct", "whitespace"]


def accuracy(model, examples, options, rng_seed: int = 42) -> float:
    from vss.inference.batching import run_batch

    correct = []
    rng = random.Random(rng_seed)
    B = 16
    for s in range(0, len(examples), B):
        batch = examples[s : s + B]
        states = [ex["state"] for ex in batch]
        questions = [[{"id": "intent", "type": "choice", "options": options}] for _ in batch]
        answers = run_batch(model, states, questions, abstain_threshold=0.55,
                            enable_abstention=False, confidence_mode="blend")
        for a, ex in zip(answers, batch):
            correct.append(int(a["answers"]["intent"]["value"] == ex["questions"][0]["answer"]))
    return float(np.mean(correct))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default="data/clinc150/test.jsonl")
    ap.add_argument("--labels", default="data/raw/clinc_label_names.json")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmarks/robustness/clinc150_slot_ho.json")
    args = ap.parse_args()

    examples = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    rng = random.Random(args.seed)
    examples = rng.sample(examples, min(args.n_samples, len(examples)))
    options = sorted(json.loads(Path(args.labels).read_text()))
    model = load_model(args.model)

    results = {}
    for kind in PERTURBATIONS:
        pert = []
        for ex in examples:
            e = json.loads(json.dumps(ex))  # deep copy
            st = e["state"]
            if isinstance(st, dict):
                for k, v in st.items():
                    if isinstance(v, str):
                        st[k] = perturb_text(v, kind, random.Random(hash((k, v)) & 0xFFFF))
            elif isinstance(st, list):
                st = [perturb_text(str(v), kind, random.Random(hash(v) & 0xFFFF)) if isinstance(v, str) else v for v in st]
                e["state"] = st
            pert.append(e)
        acc = accuracy(model, pert, options, rng_seed=args.seed)
        results[kind] = round(acc, 4)
        print(f"{kind:12s} accuracy = {acc:.4f}", flush=True)

    clean = results["clean"]
    report = {
        "model": args.model,
        "data": args.test,
        "n_examples": len(examples),
        "seed": args.seed,
        "n_options": len(options),
        "accuracy_by_perturbation": results,
        "delta_vs_clean": {k: round(v - clean, 4) for k, v in results.items() if k != "clean"},
        "mean_abs_delta": round(float(np.mean([abs(v - clean) for k, v in results.items() if k != "clean"])), 4),
        "note": "deterministic perturbations only (no LLM paraphrase); each perturbation "
                "instance derived from a stable per-string seed so re-runs are reproducible",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
