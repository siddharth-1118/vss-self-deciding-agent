"""Basic VSS example: one state, three typed questions, one forward pass.

The trained checkpoint is NOT committed to this repository (it is a build
artifact), so run the quick-start training command first, or pass an explicit
model directory:

    # 1. train once (~6 min on CPU), or
    python scripts/train.py --config configs/vss-prototype.yaml \
        --train data/generated/train.jsonl --eval data/generated/eval.jsonl \
        --out runs/prototype

    # 2. run this example, optionally pointing at your own checkpoint
    python examples/basic.py
    python examples/basic.py runs/my-run/final
"""
import json
import os
import sys
from pathlib import Path

from vss import VSS

# argv[1] wins; otherwise fall back to $VSS_MODEL, then the quick-start path.
model_dir = (sys.argv[1] if len(sys.argv) > 1
             else os.environ.get("VSS_MODEL", "runs/prototype/final"))

if not Path(model_dir).exists():
    sys.exit(
        f"No trained model at '{model_dir}'.\n"
        "Checkpoints are build artifacts and are not committed. Train one first:\n"
        "  python scripts/prepare_data.py --out data/generated\n"
        "  python scripts/train.py --config configs/vss-prototype.yaml "
        "--train data/generated/train.jsonl --eval data/generated/eval.jsonl "
        "--out runs/prototype\n"
        "Or pass an existing checkpoint: python examples/basic.py <model_dir>"
    )

model = VSS.from_pretrained(model_dir)

# The quick-start model is trained on data/generated, whose vocabulary is a small
# fixed set of support departments. Reusing that same option list keeps this
# example in-distribution; inventing a new option list or new vocabulary here
# measures out-of-distribution behaviour, which the model answers with ABSTAIN.
DEPARTMENTS = ["billing", "sales", "shipping", "technical"]

result = model.decide(
    state={
        "message": "my invoice shows a duplicate payment from last month",
    },
    questions=[
        {"id": "department", "type": "choice", "options": DEPARTMENTS},
        {"id": "duplicate_charge", "type": "noul"},
        {"id": "urgency", "type": "score", "min": 0, "max": 10},
    ],
)

print("--- in-distribution state -------------------------------------------")
print(json.dumps(result, indent=2))

# KNOWN LIMITATION -- read before relying on ABSTAIN.
#
# It is tempting to assume the model abstains on input it has never seen. It does
# not. A probe over 320 in-distribution, 320 word-scrambled and 8 foreign-topic
# states (benchmarks/convergence/ood_probe.py) measured abstain rates of 0.103,
# 0.106 and 0.000 respectively: destroying every lexical token left abstention
# unchanged, and fluent off-domain text was answered with *higher* confidence
# (0.942) than in-distribution text. The abstain head currently behaves like a
# roughly 10% prior, not an out-of-distribution detector.
#
# Treat ABSTAIN as a confidence signal to threshold, not as an OOD alarm. See
# docs/model_card.md ("Limitations") before routing on it.
ood = model.decide(
    state={"message": "quantum flux capacitor warranty void on the warp drive"},
    questions=[{"id": "department", "type": "choice", "options": DEPARTMENTS}],
)
print("\n--- off-domain input: note it is NOT detected as OOD -----------------")
print(json.dumps(ood["answers"]["department"], indent=2))