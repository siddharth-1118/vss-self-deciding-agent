# Release Readiness — VSS 0.1.0

**Decision: RESEARCH PREVIEW.** Not release-ready. Two gates fail outright and
one is provisional; the rest pass. This is a deliberate, labelled release of
working software with honest evidence, not a claim that every question is
settled.

The model is not scaled. Every result below is at the current ~11M-parameter
size. No parameter increase is proposed or attempted.

---

## Gate A — Engineering integrity: **PASS**

| Requirement | Evidence |
|---|---|
| Tests pass | `python -m pytest -q` — **118 passed, 0 failed** (see §Evidence) |
| Data integrity | `train ∩ test = 0` utterances on CLINC150 and Banking77, verified by inspection |
| Label consistency | choice options enumerated per split; train uses 15 options, eval ranks the full label set — a deliberate, documented shift |
| Masks | question-mask and padding-mask behaviour covered by `tests/test_question_mask.py` |
| RoPE positions | per-example state length, pinned by `TestStablePositionsArePerExample` |
| Checkpoints | atomic writes (tmp + `os.replace`); a killed process never leaves a truncated `last.pt` |
| Run isolation | `benchmarks/convergence/manifest.py::claim_run_dir` raises `RunCollision` on a live owner; 16 tests in `tests/test_run_isolation.py` |
| Failed runs never read as successful | manifest `status` is explicit (`done`/`failed`); only `status == "done"` loads as a result |

**Fixed during this release pass:** a duplicate-supervisor collision that
corrupted a checkpoint is now impossible by construction; `scripts/train.py`
gained `--out` run isolation and manifest writing.

## Gate B — Training integrity: **FAIL**

Two defects were found and fixed, but the per-dataset tuning the protocol
requires is **not complete**.

| Requirement | Status |
|---|---|
| Per-dataset learning rates | **INCOMPLETE** — one global LR (3e-4, screened on synthetic) was applied everywhere. On CLINC150 that made the plain baseline collapse to 0.000 accuracy. A per-dataset screen (`--plan lr_screen_real`) now exists; CLINC150/VSS is complete, CLINC150/plain is in progress. |
| Warmup / decay / premature zero | Fixed (audit findings 2 and 5); warmup capped at 10% of the budget, step 0 no longer a no-op, and a spent budget is reported as budget exhaustion rather than an early stop |
| Gradient flow, clipping, weight decay | Verified — 75 parameter tensors, 0 with `None`/all-zero gradients |
| Checkpoint selection | Validation loss only; test split never read during training |
| Early stopping | Shared implementation, same patience/delta/min-epochs for both systems |

**Why FAIL:** a baseline that collapsed was reported as a comparison. Until the
plain CLINC150 arm is tuned and re-run, any real-data ranking is unsupported —
in either direction.

## Gate C — Evaluation integrity: **PROVISIONAL**

| Requirement | Status |
|---|---|
| Test isolation from selection | **Verified** — test is never read for LR, threshold, or stopping decisions |
| Per-seed reporting | **FAILING** — every result is a **single seed**. No variance estimate exists |
| Metric definitions | Documented; validation *loss* explicitly excluded from cross-system comparison |
| Macro-F1 | Implemented (`vss.eval.metrics.f1_scores`) but **not yet reported** for the convergence runs |
| Reproducibility of new runs | Each run carries a manifest with config hash, git commit, dirty flag, splits, params |
| **Reproducibility of legacy metrics** | **FAILING** — see below |

**Metric traceability failure (release-blocking).** Re-evaluating the pre-audit
plain checkpoints with current code, their own saved `vocab.json`, and the
current label mapping gives numbers that contradict their logged metrics:

| checkpoint | logged | re-measured |
|---|---:|---:|
| `mqv-plain-banking77-s13/best.pt` (epoch 6) | val 0.5206 | val **5.4961**, acc **0.075** |
| `mqv-plain-clinc150-s13/best.pt` (epoch 3) | val 0.395 | val **8.6164**, acc **0.000** |

Both load with **zero missing / unexpected tensors**, and using the checkpoint's
saved tokenizer does not resolve it. The most likely cause is an inconsistent
label-index mapping between the original training run and the current evaluation
path, but this is **not confirmed**. Until it is, the historical checkpoints
cannot serve as a convergence reference, and the "plain 0.8867 on Banking77"
figure has been removed from the argument in `docs/benchmark_report.md`.
Reproduced by `benchmarks/multi_question_value/diag_plain_clinc.py`.

**Why PROVISIONAL:** a single seed cannot support any claim that one
architecture beats another by a few points.

## Gate D — Functional completeness: **PASS**

Verified end to end on this machine:

| Requirement | Evidence |
|---|---|
| Install | `pip install -e ".[dev]"` |
| Generate data | `python scripts/prepare_data.py --out data/generated` → 2400/480 |
| Train from scratch | `python scripts/train.py ... --out runs/my-run` → converged (eval 0.936 → 0.271, choice 0.813 → 0.991) |
| Load in a fresh process | `VSS.from_pretrained(...)` |
| Inference | `examples/basic.py`, `vss decide` |
| REST API | `/health` 200, `/v1/decide` 200, malformed → 422 |
| Schema validation | `extra="forbid"`; unknown question type, empty option list, inverted score range, wrong answer type all → `ValidationError` |
| Malformed / degenerate input | Empty state, empty message, 50k-char input, single-option choice all handled; 19 tests in `tests/test_api_robustness.py` |
| Question isolation | Adding a question leaves an existing answer unchanged (test) |

**Fixed during this pass — three release-blocking bugs:**

1. **`questions: []` crashed the engine** with an opaque torch
   `RuntimeError` (tensor size mismatch) instead of returning a validation
   error. A client sending an empty question list got a server error. Fixed at
   the schema (`min_length=1`) and engine (explicit `ValueError`) layers.
2. **`/health` returned `model_loaded` as the string `"True"`**, because the
   handler was annotated `-> dict[str, str]`. Any client checking the JSON type
   would mis-read it. Fixed to return a real boolean.
3. **The quick-start example could not run on a fresh clone.**
   `examples/basic.py` hard-coded `runs/prototype/final`, but `runs/` is
   gitignored, so the path does not exist in a fresh checkout and the example
   died with a stack trace. It now accepts an explicit path / `$VSS_MODEL` and
   exits with the exact training command to run.

## Gate E — Claim integrity: **PASS**

Every public claim in `docs/claims.md` carries one of **Supported /
Provisional / Withdrawn / Unsupported** and names its evidence. The withdrawn
history is preserved rather than deleted: "plain beats VSS in 35/35 cells"
remains on the record as WITHDRAWN with the reason.

No claim asserts that synthetic accuracy demonstrates real-world
generalisation; the model card disclaims it explicitly.

## Gate F — Scope and limitations: **PASS**

`docs/model_card.md` documents intended and unintended use, training data,
limitations, risks, and abstention semantics — including that **confidence is
not a guarantee of correctness** and that thresholds must be re-fitted per
domain.

---

## Evidence summary

| | |
|---|---|
| Tests | 118 passed, 0 failed (`python -m pytest -q`) |
| Environment | Windows 10, Python 3.11.9, torch 2.14.0+cpu, CPU-only, fp32 |
| Seeds | **1** on every reported result |
| Runs with manifests | all convergence runs + smoke run |
| Valid comparisons | synthetic presentation-matched tie; question isolation; latency ratios within a session |
| Provisional | Banking77 0.870 vs 0.820 (step-matched, not convergence-matched) |
| Withdrawn | CLINC150 comparison; "plain wins 35/35"; all pre-audit real-data order-invariance / interference / selective-accuracy figures |

## Unresolved blockers

1. **Plain CLINC150 must be re-run under a protocol that matches the evaluation
   schema.** It reaches chance at both 3e-4 and 3e-5, so this is the
   15-option-train → 151-option-eval transfer, not a learning-rate choice. Until
   it trains sanely there is no valid CLINC150 comparison in either direction.
2. **The legacy plain checkpoints must be explained** (Gate C). Until their
   logged metrics are reproducible, no real-data convergence reference exists.
3. **Banking77 must be run convergence-matched** (~2000 steps for both systems).
4. **Multi-seed evaluation is missing.** 3 seeds per configuration is required
   before any ranking claim; currently 1.
5. **Macro-F1, OOD metrics and risk-coverage on real data are not reported.**
6. **A fresh-clone installation and quick-start run has not been executed end to
   end** on a clean environment — everything above ran in the development
   checkout.

## Why research preview rather than release

The engineering is sound and the software works end to end. What is missing is
*evidentiary*, not functional: a mis-tuned baseline produced a false real-data
comparison, and one seed cannot separate two architectures that differ by a few
points. Shipping as a research preview with those two facts stated is honest;
shipping as "release-ready" would not be.