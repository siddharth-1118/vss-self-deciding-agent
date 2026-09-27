"""Agent-router example: many questions in one pass, one padded forward.

Demonstrates that adding questions is nearly free: the state is encoded
once; each question is a learned query attending to the shared state.
"""
from vss import VSS

model = VSS.from_pretrained("runs/prototype/final")

state = {
    "message": (
        "I cannot log in at all and your system says my account is locked. "
        "This has been broken for days and I am switching banks if it is not fixed."
    ),
    "customer_age_days": 180,
    "prior_department": "technical",
}

questions = [
    {
        "id": "department",
        "type": "choice",
        "options": ["billing", "technical", "sales", "shipping", "other"],
    },
    {"id": "refund_requested", "type": "noul"},
    {"id": "account_blocked", "type": "noul"},
    {"id": "urgency", "type": "score", "min": 0, "max": 10},
    {"id": "churn_risk", "type": "score", "min": 0, "max": 10},
]

result = model.decide(state, questions)
for qid, answer in result["answers"].items():
    print(f"{qid:>16}: {answer}")
