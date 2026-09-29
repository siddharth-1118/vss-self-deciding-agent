"""Re-evaluate the encoder-classifier ablation checkpoint, in-scope only.

The trainer reported accuracy over all 5500 test examples (OOS included as a
class). VSS's headline number is in-scope-only (4500). This script computes
both from runs/ablation-cls/last.pt for the apples-to-apples comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    import random

    from benchmarks.encoder_classifier_ablation import ClassifierModel
    from vss.model.config import ModelConfig
    from vss.model.serialize import serialize_state
    from vss.model.tokenizer import VSSTokenizer
    from vss.model.vss_model import VSSModel

    ckpt_enc = "runs/clinc150-slot-ho/best.pt"
    ck = torch.load(ckpt_enc, map_location="cpu", weights_only=False)
    raw = ck["config"]
    raw = raw.get("model", raw) if isinstance(raw, dict) else raw
    cfg = ModelConfig(**raw)
    vss = VSSModel(cfg)
    vss.load_state_dict(ck["model"])
    tok = VSSTokenizer.load(str(Path(ckpt_enc).parent / "final" / "vocab.json"))

    labels = sorted(json.loads(Path("data/raw/clinc_label_names.json").read_text()))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    model = ClassifierModel(vss.vss_encoder, cfg.hidden_size, len(labels))
    st = torch.load("runs/ablation-cls/last.pt", weights_only=False, map_location="cpu")
    model.load_state_dict(st["model"])
    model.eval()

    test = [json.loads(l) for l in open("data/clinc150/test.jsonl", encoding="utf-8")]
    random.Random(42).shuffle(test)
    test = test[:1000]  # same subsample protocol as other comparisons

    inscope_correct = inscope_n = 0
    all_correct = 0
    with torch.no_grad():
        for s in range(0, len(test), 32):
            chunk = test[s : s + 32]
            ids = [tok.encode(serialize_state(ex["state"])) for ex in chunk]
            T = max(len(x) for x in ids)
            x = torch.zeros(len(chunk), T, dtype=torch.long)
            for j, v in enumerate(ids):
                x[j, : len(v)] = torch.tensor(v)
            pred = model(x).argmax(-1)
            for j, ex in enumerate(chunk):
                gold = ex["questions"][0]["answer"]
                ok = int(pred[j].item() == label_to_idx[gold])
                all_correct += ok
                if gold != "oos":
                    inscope_correct += ok
                    inscope_n += 1

    out = {
        "checkpoint": "runs/ablation-cls/last.pt",
        "protocol": "n=1000 test subsample seed 42 (matches baseline comparisons)",
        "accuracy_all_1000_including_oos_as_class": round(all_correct / len(test), 4),
        "accuracy_inscope_only": round(inscope_correct / max(1, inscope_n), 4),
        "n_inscope": inscope_n,
        "vss_reference_inscope": 0.7398,
        "vss_reference_protocol": "full 4500 in-scope (benchmarks/ood/clinc150_slot_ho_best.json)",
    }
    out_path = Path("benchmarks/ablation/encoder_classifier_inscope.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
