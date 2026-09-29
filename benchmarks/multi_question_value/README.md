# benchmarks/multi_question_value

**The critical experiment**: does VSS (System C) provide measurable value
over a plain classifier called once per question (Systems A/B) when answering
multiple questions over the same state?

## Systems compared

| mode | system | inference for Q questions |
|------|--------|---------------------------|
| A | plain classifier, sequential | Q separate forward passes (batch 1) |
| B | plain classifier, batched | 1 padded forward over Q (state, question) rows |
| C | VSS (qmask, stable RoPE) | 1 forward with the whole request |

The plain classifier is deliberately simple **and deliberately strong**:
same word+hash tokenizer, same encoder architecture (hidden 256, 6 layers,
8 heads, SwiGLU 1024, RoPE), same serialization (`<STATE>` + one `<QUESTION>`
block), same typed head semantics (masked choice CE + trained ABSTAIN class,
noul BCE, score = 64 ordinal bins with expected-value decoding), same
optimizer/schedule/epochs/batch/seeds/splits as the VSS recipe. It differs
from VSS exactly where the architecture differs: no shared state
representation, no question masking, one decision per forward.

## Fairness guarantees

- **Identical data**: both systems train and evaluate on the same splits;
  at every Q the eval states and their question sets are identical across
  modes (`dataset.eval_subset` — fixed-prefix pools for synthetic, replicated
  real questions with distinct ids for Q>1 on single-question real data).
- **Per-question gold scoring**: metrics operate on flat per-question records
  (`metrics.py`); request accuracy (all questions correct) is reported
  separately. Never one score per request.
- **Paired evaluation**: mode A and B are the same trained network; a
  distribution-equality check (`A_equals_B`) verifies the batching does not
  change predictions. C-vs-B accuracy differences get a paired bootstrap CI
  (1000 resamples over per-question correctness).
- **Latency**: warmup + measured iterations (`config.BenchConfig`), percentiles
  p50/p95/p99, single-request path per mode, cold start (model load) reported
  separately. Environment (torch version, threads, precision, RAM, CPU) is
  recorded in every results file.

## Files

| file | purpose |
|------|---------|
| `config.py` | PlainConfig (mirrors VSS recipe) + BenchConfig |
| `config.yaml` | experiment-level settings (Q counts, seeds, latency budget) |
| `dataset.py` | synthetic generator (deterministic, verified gold) + real-data conversion + nested Q subsets |
| `plain_classifier.py` | System A/B model + training + per-row predictions |
| `vss_runner.py` | System C loader + per-question records + latency |
| `metrics.py` | per-question accuracy/macro-F1/ECE/Brier/NLL/AUROC/risk-coverage |
| `benchmark.py` | the CLI running modes A/B/C over Q × seeds |
| `plot.py` | SVG charts (no matplotlib in this environment) |
| `report.py` | aggregates results JSONs into markdown tables |

## Reproduce

```bash
# 1. synthetic dataset (deterministic; gold re-verified on load)
python -c "import sys; sys.path.insert(0,'benchmarks/multi_question_value'); \
  import dataset; \
  dataset.generate_synthetic(4000, 7, 'train'); \
  dataset.generate_synthetic(800, 7, 'validation'); \
  dataset.generate_synthetic(1200, 7, 'test')"

# 2. train both systems (3 seeds each; --resume survives the 600 s shell cap)
python benchmarks/multi_question_value/train_plain.py --dataset synthetic --seed 1
python benchmarks/multi_question_value/train_vss.py   --dataset synthetic --seed 1
# (also clinc150 / banking77 plain; VSS uses existing qmask checkpoints there)

# 3. run the comparison
python benchmarks/multi_question_value/benchmark.py --dataset synthetic \
    --questions 1 2 4 8 16 32 50 --seeds 1 2 3 \
    --plain-root runs/mqv-plain-synthetic-s{seed} \
    --vss-root runs/mqv-vss-synthetic-s{seed} \
    --interference-at 8 32 50 --permutation-at 8 32

# 4. charts + report tables
python benchmarks/multi_question_value/plot.py \
    --results benchmarks/multi_question_value/results/synthetic_results.json
python benchmarks/multi_question_value/report.py \
    --results benchmarks/multi_question_value/results/*.json \
    --out benchmarks/multi_question_value/report.md
```

## What this benchmark does NOT do

- It does not train a bigger VSS (scaling is gated on this result).
- It does not tune either system on test data; checkpoint selection is
  validation-loss only.
- It does not compare against an artificially slowed baseline: mode B is the
  hardware-honest version of the plain classifier, and VSS is benchmarked
  through its shipped inference path.
