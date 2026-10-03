# VSS

**VSS** is an open-source **System-One decision intelligence model**: give it
a state (text and/or JSON) and a set of typed questions; it answers **all of
them in a single inference pass** with probability distributions, calibrated
confidence, and schema-constrained outputs. It is a real trainable PyTorch
model — not a prompt wrapper around an LLM.

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

```json
{
  "answers": {
    "department": {"value": "billing",
                   "probabilities": {"billing": 0.94, "technical": 0.02, "...": "..."},
                   "confidence": 0.92},
    "refund_requested": {"value": 1, "probability": 0.97, "confidence": 0.97},
    "urgency": {"value": 3.1, "confidence": 0.87}
  }
}
```

## What it guarantees

- **One forward pass for N questions** — the state is encoded once; each
  question is a learned query attending to it. Measured: 50 questions cost
  ~10x one question, not 50x.
- **Schema-constrained outputs** — a choice answer can ONLY be one of the
  declared options; the head scores exactly those options. Invalid values
  are impossible, not discouraged.
- **Three first-class question types** — `choice` (distribution over
  declared options), `noul` (binary probability), `score` (ordinal-binned
  value on any declared scale).
- **Calibrated confidence** — an auxiliary correctness head trained on
  empirical correctness, complemented by temperature scaling; evaluated
  with ECE / Brier / NLL / reliability buckets.
- **Abstention** — `{"value": "ABSTAIN", "confidence": 0.31}` instead of a
  forced guess, via threshold or the trained abstain logit.
- **Text AND JSON state** — deterministic canonical serialization of nested
  objects, arrays and scalars.

## Provenance

VSS targets publicly observable System-One behavior (typed decisions,
parallel output, calibration) but reproduces **no proprietary internals**.
Every design element in `docs/architecture.md` is tagged as
[public] documented concept, [std] standard ML technique, or [vss]
original engineering decision. VSS is not claimed to be equivalent to any
proprietary system unless measured.

## Install

```bash
pip install -e ".[dev]"        # core + tests
pip install -e ".[serve]"      # REST API
pip install -e ".[onnx]"       # ONNX export
```

## Quickstart (end-to-end, ~6 min on CPU)

Trained checkpoints are build artifacts and are **not committed** (`runs/` is in
`.gitignore`), so train one first. Each run must write to its own `--out`
directory; the trainer refuses to start if another live process owns it, and
writes a `manifest.json` recording the config hash, git revision, splits and an
explicit `done` / `failed` status.

```bash
python scripts/prepare_data.py --out data/generated          # synthetic data
python scripts/train.py --config configs/vss-prototype.yaml \
    --train data/generated/train.jsonl --eval data/generated/eval.jsonl \
    --out runs/prototype
python scripts/evaluate.py --model runs/prototype/final \
    --data data/generated/eval.jsonl --latency --sweep-thresholds
python examples/basic.py
```

Run the tests with `python -m pytest -q`.

Full environment, dataset preparation, tuning and evaluation commands:
[`docs/training_and_reproduction.md`](docs/training_and_reproduction.md).

## Release status

**Research preview (v0.1.0).** The software works end to end, the training
protocol is tuned and converged, and the engineering invariants are tested. It is
held back from release by one thing: **CLINC150 and the synthetic benchmark are
still single-seed**, and on the dataset that does have three seeds the plain
baseline is ahead.

VSS has **not** been shown to beat a plain classifier — and on the evidence
available, it loses. Measured against a tuned, converged baseline at per-dataset
learning rates, validation-selected checkpoints:

| dataset | n | VSS | plain | outcome |
|---|---:|---|---|---|
| Banking77 (77 classes) | 3 | 0.8300 (sd 0.065) | **0.8783** (sd 0.008) | plain +4.8 pts |
| CLINC150 (151 classes) | 1 | 0.675 | **0.915** | plain +24.0 pts |

VSS is also **much less stable across seeds** — 0.765 to 0.895 on Banking77,
against plain's 0.870 to 0.885. A single-seed run of VSS on Banking77 scored
0.895, above plain's best seed; at another seed it scored 0.765. If you are
choosing a model for real-data intent classification, the honest recommendation
from this evidence is the plain classifier.

Two further measured limitations: VSS's best checkpoint lands on the final epoch
where it is not stopped early (its numbers are floors, not ceilings), and
`ABSTAIN` is a confidence threshold rather than an out-of-distribution detector.

Gates, evidence and unresolved blockers:
[`docs/release_readiness.md`](docs/release_readiness.md).
Claim-by-claim status: [`docs/claims.md`](docs/claims.md).
Validated comparisons: [`docs/benchmark_report.md`](docs/benchmark_report.md).
Model card: [`docs/model_card.md`](docs/model_card.md).

## REST API

```bash
vss serve --model runs/prototype/final --port 8000
curl -X POST localhost:8000/v1/decide -H 'content-type: application/json' \
  -d '{"state": {"message": "My card was charged twice."},
       "questions": [{"id": "department", "type": "choice",
                      "options": ["billing", "technical", "sales", "other"]}]}'
```

## Model sizes

| preset | params | status |
|---|---|---|
| vss-prototype | 13.7M | trained (this repo, see `benchmarks/README.md`) |
| vss-small | ~52M | config ready |
| vss-base | ~250M | config ready |
| vss-large | ~600M | config ready |

Train the next tier only after experiments justify it. All architecture
values are config; nothing is hard-coded.

## Repository

```
src/vss/
  model/        # transformer, encoder, questions, heads, calibration, config
  inference/    # engine, schema validation, batching, REST server
  training/     # trainer, losses, curriculum, rlcd (experimental)
  data/         # schema/validator, synthetic generator
scripts/        # prepare_data, train, evaluate, calibrate, export
configs/        # vss-prototype / small / base / large YAML
benchmarks/     # measured results + reproduction commands
docs/           # architecture, training, inference, datasets
tests/          # 38 unit + integration tests
```

## Development rules (enforced by CI-able checks)

No fake benchmark numbers. No claims of equivalence to proprietary systems.
Poor results are reported, not hidden. Failed assumptions are replaced and
documented. Every major architectural choice has a benchmark command.

## License

MIT. See `MODEL_CARD.md` for intended use and limitations.
