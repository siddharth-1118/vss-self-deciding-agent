# VSS benchmarks

Reproducible benchmark scripts. **Rule: never publish a number that a
script in this directory did not produce.**

## Layout

```
benchmarks/
  accuracy/       # accuracy / macro-F1 / micro-F1 by stage (scripts/evaluate.py)
  calibration/    # ECE, Brier, NLL, reliability data, temperature fit
  latency/        # p50/p95/p99, req/s scaling with question count
  robustness/     # negation, typos, irrelevant context, OOD (planned)
```

## Current measurements (VSS-Prototype, 13.7M params, CPU)

Hardware: consumer CPU laptop, torch 2.14 (CPU build), fp32, batch=1.
Dataset: `data/generated/eval.jsonl` (480 held-out synthetic examples,
generator seed disjoint from training). Choice rows include 10% with gold
answer ABSTAIN (OOD/ambiguous inputs); accuracy counts a correct ABSTAIN
as correct.

| metric | value |
|---|---|
| choice accuracy (incl. correct abstentions) | 1.000 |
| choice abstain rate (matches ABSTAIN-target share exactly) | 0.100 |
| choice macro-F1 | 1.000 |
| ECE | 0.0044 |
| Brier | 0.0008 |
| NLL | 0.0049 |
| (confidence semantics) | abstain rows use `abstain_probability` — the model's P(abstaining is right) |
| AUROC / AUPRC of confidence | n/a — zero wrong answers on this split |
| score mean relative error | 0.042 |

Calibration is genuinely good, not asserted: the fitted temperature is
~1.0 and post-hoc scaling changes ECE from 0.0042 to ~1e-8 while leaving
accuracy unchanged. On rows where the model answers concretely it is 100%
correct, so no AUROC exists (undefined without a negative class) — the
report emits `null`, not a fake number.

**Scope caveat (do not skip):** eval examples come from the same programmatic
generator family as training data. These numbers establish that the
pipeline learns and calibrates, NOT real-world robustness. The robustness
suite (negation/paraphrase/typos held-out generators, real public datasets)
is the required next step before any external claim.

Latency scaling — the parallel-decision claim, measured:

| questions | p50 ms | p95 ms | req/s |
|---|---|---|---|
| 1 | 12.1 | 13.1 | 82.2 |
| 5 | 22.2 | 32.8 | 44.2 |
| 10 | 34.8 | 41.3 | 28.7 |
| 50 | 174.7 | 192.1 | 5.6 |

50 questions cost ~14x one question, not 50x: the state is encoded once
and questions are learned queries over it.

Reproduce:

```bash
python scripts/evaluate.py --model runs/prototype/final \
    --data data/generated/eval.jsonl --latency --sweep-thresholds
```

## Robustness probe (manual, n=6 — preliminary, not a suite)

| probe | result |
|---|---|
| negation "I do NOT want a refund" | correct (refund=0, conf 1.0) |
| negation "no refund needed" | correct (refund=0, conf 0.9999) |
| affirmative refund phrasing | correct (refund=1, conf 0.98) |
| typos ("paymet", "chagred", "refnud") | correct (billing, conf 0.999) |
| irrelevant filler (cats, weather) | correct (billing, conf 0.75 — appropriately lower) |
| 6x repeated filler burying the signal | **FAILED** -> ABSTAIN on an answerable input |

The repeated-filler failure is expected: that exact pattern is outside the
training distribution, and the model honestly abstained rather than
hallucinating. A systematic robustness suite (held-out perturbation
generators, paraphrase sets, prompt-injection probes) is required before
any external claim.

## Baseline comparisons (planned)

The mission requires comparing against standard classifiers, small encoder
models, and LLM JSON/function-calling on identical splits. Not yet run —
no numbers will be published here until then.
