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
import sys
from pathlib import Path

from vss import VSS

# argv[1] wins; otherwise fall back to $VSS_MODEL, then the quick-start path.
model_dir = (sys.argv[1] if len(sys.argv) > 1
             else __import__("os").environ.get("VSS_MODEL", "runs/prototype/final"))

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

result = model.decide(
    state={
        "message": "My card was charged twice and I need a refund.",
        "customer_age_days": 421,
    },
    questions=[
        {
            "id": "department",
            "type": "choice",
            "options": ["billing", "technical", "sales", "shipping", "other"],
        },
        {"id": "refund_requested", "type": "noul"},
        {"id": "urgency", "type": "score", "min": 0, "max": 10},
    ],
)

print(json.dumps(result, indent=2))