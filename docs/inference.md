# Inference

## Python API

```python
from vss import VSS

model = VSS.from_pretrained("runs/prototype/final")
result = model.decide(
    state={"message": "My card was charged twice.", "customer_age_days": 421},
    questions=[
        {"id": "department", "type": "choice",
         "options": ["billing", "technical", "sales", "shipping", "other"]},
        {"id": "refund_requested", "type": "noul"},
        {"id": "urgency", "type": "score", "min": 0, "max": 10},
    ],
)
```

`decide_batch(states, questions_list)` runs chunked padded batches.

## Guarantees

Enforced in code, not by convention (`inference/schema.py` re-validates
every output):

- choice `value` is always one of the DECLARED options — the head only
  produces logits for those options; invalid options are impossible
- `probabilities` keys equal the declared options and sum to 1
- noul `value` in {0, 1} with explicit `probability`
- score `value` within [min, max]
- confidence in [0, 1] everywhere
- every request passes pydantic validation first (unknown fields rejected)

## Abstention

Two mechanisms:

1. **Threshold** — configured in YAML (`inference.abstain_threshold`,
   default 0.55) or per-call: when confidence falls below the threshold,
   the answer becomes `{"value": "ABSTAIN", "confidence": c}`.
2. **Learned abstain class** — the choice head carries a trained abstain
   logit (softmax over options + abstain, trained on `ABSTAIN` targets).
   When it wins the argmax, the model is saying "none of these options".
   The abstain entry reports the best declared option's support as
   `confidence` (how weak the concrete evidence is) and the model's
   P(abstaining is right) as `abstain_probability`:

```json
{"value": "ABSTAIN", "confidence": 0.0004, "abstain_probability": 0.9994}
```

Calibration metrics use `abstain_probability` for abstain rows so the
reported ECE reflects the model's actual stated belief.

## REST

```bash
vss serve --model runs/prototype/final --port 8000
# or: python -m vss.inference.server --model ...

curl -X POST localhost:8000/v1/decide -H 'content-type: application/json' -d '{
  "state": {"message": "My card was charged twice."},
  "questions": [
    {"id": "department", "type": "choice",
     "options": ["billing", "technical", "sales", "other"]},
    {"id": "refund_requested", "type": "noul"}
  ]
}'
```

`GET /health` reports liveness and model-loaded state. Invalid requests
return HTTP 422; an unloaded model returns 503.

## CLI

```bash
vss train --config configs/vss-prototype.yaml --train data/generated/train.jsonl
vss decide --model runs/prototype/final --state '{"message": "..."}' --questions questions.json
vss evaluate --model runs/prototype/final --data data/generated/eval.jsonl
vss validate --data data/train.jsonl
vss generate --out data/generated/train.jsonl
```

If another `vss` executable exists on your PATH, use
`python -m vss.cli <command>` instead.

## Determinism

`inference.deterministic: true` keeps the model in eval mode with no
sampling anywhere in the pipeline; identical inputs produce identical
outputs on the same hardware. Serialization and tokenization are fully
deterministic by construction.
