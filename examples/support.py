"""Support-routing example: dynamic option schemas per request.

The model never invents options: whatever subset the caller declares is
exactly what the probability distribution covers.
"""
from vss import VSS

model = VSS.from_pretrained("runs/prototype/final")

tickets = [
    {"message": "the mobile app crashes when I open the export screen"},
    {"message": "I want a quote for 25 seats on the annual plan"},
    {"message": "my package says delivered but it never arrived"},
]

# Same taxonomy question, three different declared option sets.
full = ["billing", "technical", "sales", "shipping", "other"]
narrow = ["shipping", "other"]  # e.g. a logistics-only intake queue

for i, state in enumerate(tickets):
    options = narrow if i == 2 else full
    result = model.decide(
        state,
        [{"id": "department", "type": "choice", "options": options}],
    )
    print(state["message"][:50], "->", result["answers"]["department"])
