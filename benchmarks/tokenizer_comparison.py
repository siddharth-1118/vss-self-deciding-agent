"""Tokenizer comparison: word+hash vs byte-pair encoding, identical recipe.

Hypothesis under test (from docs/current_state_audit.md #6): the word-level
tokenizer with FNV-1a OOV hashing hurts robustness and capacity. We train the
IDENTICAL CLINC150 slot-ho recipe (seed 13) with a drop-in BPETokenizer
(src/vss/model/tokenizer_bpe.py, selected via `tokenizer_type: bpe`) and
compare test accuracy + perturbation robustness against the committed
word+hash run. The model is tokenizer-agnostic, so this isolates the
tokenizer variable exactly.

Usage:
    python scripts/train.py --config configs/vss-prototype-clinc-slot-ho-bpe.yaml \
        --train data/clinc150/train.jsonl --eval data/clinc150/validation_1000.jsonl
    python benchmarks/tokenizer_comparison.py evaluate \
        --ckpt runs/clinc150-slot-ho-bpe/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vss.model.tokenizer import RESERVED_TAGS, VSSTokenizer  # noqa: E402


# BPETokenizer now lives in src/vss/model/tokenizer_bpe.py (shared with trainer).

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["evaluate"])
    ap.add_argument("--ckpt", default="runs/clinc150-slot-ho-bpe/best.pt")
    ap.add_argument("--test", default="data/clinc150/test.jsonl")
    ap.add_argument("--labels", default="data/raw/clinc_label_names.json")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="benchmarks/tokenizer/clinc150_bpe_vs_word.json")
    args = ap.parse_args()

    if args.mode == "prepare":  # kept for CLI compatibility
        print("config already exists: configs/vss-prototype-clinc-slot-ho-bpe.yaml")
        return 0

    # ---------- evaluate the BPE-trained checkpoint --------------------------
    from vss.inference.batching import run_batch
    from vss.model.tokenizer_bpe import BPETokenizer

    def load_model(path: str):
        import torch

        from vss.model.config import ModelConfig
        from vss.model.vss_model import VSSModel

        p = Path(path)
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        raw = ck["config"]
        raw = raw.get("model", raw) if isinstance(raw, dict) else raw
        model = VSSModel(ModelConfig(**raw))
        model.load_state_dict(ck["model"])
        tok_path = p.parent / "final" / "vocab.json"
        tok = BPETokenizer.load(tok_path)
        model.tokenizer = tok
        model.vss_encoder.tokenizer = tok
        model.eval()
        return model

    def perturb_text(s: str, kind: str, rng) -> str:
        words = s.split(" ")
        if kind == "lower":
            return s.lower()
        if kind == "upper":
            return s.upper()
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
        return " ".join(words)

    examples = [json.loads(l) for l in open(args.test, encoding="utf-8")]
    import random

    examples = random.Random(args.seed).sample(examples, min(args.n_samples, len(examples)))
    options = sorted(json.loads(Path(args.labels).read_text()))
    model = load_model(args.ckpt)

    def accuracy(perturb: str | None = None) -> float:
        correct = []
        for s in range(0, len(examples), 16):
            batch = examples[s : s + 16]
            states = [ex["state"] for ex in batch]
            if perturb:
                for st in states:
                    if isinstance(st, dict):
                        for k, v in st.items():
                            if isinstance(v, str):
                                st[k] = perturb_text(v, perturb, random.Random(hash((k, v)) & 0xFFFF))
            questions = [[{"id": "intent", "type": "choice", "options": options}] for _ in batch]
            answers = run_batch(model, states, questions, abstain_threshold=0.55,
                                enable_abstention=False, confidence_mode="blend")
            for a, ex in zip(answers, batch):
                correct.append(int(a["answers"]["intent"]["value"] == ex["questions"][0]["answer"]))
        return float(np.mean(correct))

    bpe = {"clean": round(accuracy(), 4),
           "typo": round(accuracy("typo"), 4),
           "drop_word": round(accuracy("drop_word"), 4)}
    print("BPE:", bpe)

    # word+hash reference from the committed robustness run (same subset seed)
    ref = json.loads(Path("benchmarks/robustness/clinc150_slot_ho.json").read_text())
    report = {
        "comparison": "identical CLINC150 slot-ho recipe (seed 13); only tokenizer differs",
        "bpe_trained": bpe,
        "word_hash_reference": {k: ref["accuracy_by_perturbation"][k]
                                for k in ("clean", "typo", "drop_word")},
        "note": "reference run used n=1000 seed 42 test subsample — same protocol",
        "conclusion": None,
    }
    gap_clean = bpe["clean"] - ref["accuracy_by_perturbation"]["clean"]
    report["conclusion"] = (
        f"BPE clean accuracy delta vs word+hash: {gap_clean:+.4f} "
        f"(typo delta {bpe['typo'] - ref['accuracy_by_perturbation']['typo']:+.4f}, "
        f"drop_word delta {bpe['drop_word'] - ref['accuracy_by_perturbation']['drop_word']:+.4f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
