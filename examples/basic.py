"""Basic VSS example: one state, three typed questions, one forward pass."""
from vss import VSS

model = VSS.from_pretrained("runs/prototype/final")

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

import json

print(json.dumps(result, indent=2))
