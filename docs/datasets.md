# Datasets

## Format

JSONL, one example per line:

```json
{"state": {...}, "questions": [{"id": "...", "type": "choice|noul|score", ..., "answer": ...}]}
```

Validation rules (enforced by `vss/data/schema.py`):

- `choice`: non-empty `options`; `answer` in options, or the literal
  `"ABSTAIN"` for no-defensible-option examples
- `noul`: `answer` in {0, 1}
- `score`: `max > min`; `answer` within [min, max]
- at least one question per example; unknown fields rejected
- `state` must be a JSON object or array (flattened deterministically)

Validate any file:

```bash
vss validate --data data/train.jsonl     # exit 1 + line-precise errors
```

## Synthetic generator

`src/vss/data/synthetic.py` produces the six curriculum stages with
programmatic labels (no LLM in the labeling path) and deliberate hard
cases: negation traps ("no refund needed" -> `refund_requested=0`),
irrelevant filler, typos, buried signals in long context, contradictory
context fields, out-of-domain text -> ABSTAIN targets, dynamic option
subsets.

```bash
python scripts/prepare_data.py --out data/generated --seed 13
```

Seeds fully determine the data; train/eval use disjoint seeds.

## Real public datasets

Planned conversions (permissive licenses only), via the normalizer
pipeline in `data/README.md`:

| source | VSS questions |
|---|---|
| CLINC150 | choice (intent), noul (out-of-scope) |
| banking77 | choice (77-way; subsampled schemas) |
| SNIPS | choice (intent), noul (slot presence) |
| GoEmotions | noul (per-emotion) |
| TREC | choice (question type) |
| AG News | choice (topic) |
| Jigsaw | score (severity), noul (toxic) |

Rules: never blindly merge incompatible datasets; validate everything;
report per-source metrics separately so the mixture is auditable.

## LLM teacher policy

An LLM may GENERATE additional states, never labels. Programmatic rules
produce labels; teacher outputs pass the same validator and are never
unquestioned ground truth.
