# Release Readiness — VSS 0.1.0

**Decision: RESEARCH PREVIEW.** Not release-ready. All engineering, training and
scope gates now pass; **Gate C (evaluation integrity) is PROVISIONAL** because
the synthetic benchmark is still single-seed. This is a deliberate, labelled
release of working software with honest evidence, not a claim that every question
is settled.

**What the evidence now says about the model itself** (convergence-matched,
per-dataset tuned, validation-selected, three seeds each):

| dataset | n | VSS | plain classifier | outcome |
|---|---:|---|---|---|
| Banking77 (77 classes) | 3 | 0.8300 (sd 0.065) | **0.8783** (sd 0.008) | plain +4.8 pts |
| CLINC150 (151 classes) | 3 | 0.7050 (sd 0.052) | **0.9183** (sd 0.006) | plain +21.3 pts |

Under the corrected checkpoint-selection rule (validation accuracy rather than
eval loss, finding 7), re-reading the same recorded histories: Banking77
**0.8650 vs 0.8883** (plain +2.3), CLINC150 **0.7583 vs 0.9233** (plain +16.5).
The sign of the comparison does not change under either rule.

This is not a model that has been shown to beat its baseline. On the evidence
available **the plain classifier is better on both real datasets**, and VSS is
markedly less stable across seeds on both (spread 0.130 vs 0.015 on Banking77,
0.090 vs 0.010 on CLINC150). A single-seed run would have said the opposite on
Banking77 — VSS scored 0.895 at seed 13 and 0.765 at seed 7 — which is precisely
why the multi-seed requirement existed.

The model is not scaled. Every result below is at the current ~11M-parameter
(concretely: **11,164,483** for the benchmark configuration used by the
convergence sweep, and **13,655,364** for `configs/vss-prototype.yaml`, which
drives the verified end-to-end reference run in
`docs/benchmark_report.md` §10). Both are recorded in
`docs/current_state_audit.md`. No parameter increase is proposed or attempted.

---

## Gate A — Engineering integrity: **PASS**

| Requirement | Evidence |
|---|---|
| Tests pass | `python -m pytest -q` — **148 passed, 1 skipped, 0 failed** (see §Evidence) |
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

## Gate B — Training integrity: **PASS**

Per-dataset tuning is now complete for both architectures on both datasets, and
the earlier failures are fixed.

| Requirement | Status |
|---|---|
| Per-dataset learning rates | **DONE** — 20-run screen (5 LRs × 2 systems × 2 datasets), identical 600-step budget per run, validation-only selection. All four optima are **interior** (each beats both grid neighbours), so they are bracketed rather than pinned to a grid edge. Selected: Banking77 **3e-4** both systems; CLINC150 plain **1e-3**, VSS **3e-4**. |
| Consistent task definition across systems | **DONE** — the `header_only_choice` mismatch that had the plain baseline training a different objective was root-caused and fixed; 11 parity tests. |
| Warmup / decay / premature zero | Fixed (audit findings 2 and 5); warmup capped at 10% of budget, step 0 is not a no-op, a spent budget is reported as budget exhaustion rather than a fake early stop |
| Gradient flow, clipping, weight decay | Verified — 75 parameter tensors, 0 with `None`/all-zero gradients |
| Checkpoint selection | Validation loss only; test split never read during training. Accuracy is read at the *selected* (lowest-loss) epoch, not the best epoch by accuracy. |
| Early stopping | Shared implementation, same patience/delta/min-epochs for both systems; `early=True` and budget-exhaustion are reported distinctly |
| Convergence-matched budgets | **DONE** — 2000 steps per run. Both systems genuinely converged or exhaust-budget-declared. |

## Gate C — Evaluation integrity: **PROVISIONAL**

| Requirement | Status |
|---|---|
| Test isolation from selection | **Verified** — test is never read for LR, threshold, or stopping decisions |
| Per-seed reporting | **PARTIAL** — Banking77 **and CLINC150** now have **3 seeds per system** (real variance estimates on both). Synthetic remains single-seed. |
| Metric definitions | Documented; validation *loss* explicitly excluded from cross-system comparison (VSS's contains a calibration BCE and soft-ordinal term the plain loss lacks) |
| Macro-F1 | Implemented and reported for the verified smoke run (0.9928 synthetic); **not yet reported** for the real-data runs |
| Reproducibility of new runs | Each run carries a manifest with config hash, git commit, dirty flag, splits, params, pid, timestamps, and explicit `done`/`failed` status |
| **Reproducibility of legacy metrics** | **RESOLVED** — see below |

**Legacy-checkpoint discrepancy: root cause found, not guessed.** `header_only_choice`
was read from the **model config** by VSS but from the **question object** by the
plain baseline. `AnsweredQuestion` has no such field (`extra="forbid"`), so the
real-data loader silently left it absent for plain — it trained full-inventory
cross-entropy while VSS trained the header-only objective. Same checkpoint, same
data, only that flag changed:

| `header_only_choice` | choice accuracy |
|---|---:|
| `True` (as trained) | **0.8900** |
| `False` | 0.0750 |

Through the authoritative `load_plain()` path the legacy Banking77 checkpoint now
reproduces at **0.9400 test accuracy**. The checkpoints were always sound; the
*measurement* was wrong. Label-index mapping was verified identical, and
`tests/test_plain_header_only_parity.py` proves a known example maps to the same
class index in training and inference.

**Consequence:** every real-data comparison made before this fix was void,
because the plain arm was training a different task. They have been re-run.

**Why still PROVISIONAL:** the synthetic benchmark rests on one seed. Both real
datasets now have three seeds each, but Banking77's three seeds are what
overturned the earlier single-seed reading, so treating the remaining
single-seed synthetic number as settled would repeat the exact error this audit
exists to catch.

Measured variance estimates now exist on both real datasets: **Banking77 sd
0.008 (plain) / 0.065 (VSS)**, **CLINC150 sd 0.006 / 0.052**.

## Gate D — Functional completeness: **PASS**

### Verified in a genuinely clean checkout (Blocker E)

The dev checkout was not accepted as evidence. A fresh `git clone` of the
repository at commit `92e50ca` into an empty directory, with **no `runs/`** and
no shared state, was taken through the entire documented workflow:

| step | command | outcome |
|---|---|---|
| dependencies | import every declared dep from the clone's own `src` | numpy, torch, pydantic, yaml, safetensors, tqdm, fastapi all resolve |
| data prep | `python scripts/prepare_data.py` | train 2400 / eval 480, both `OK` |
| train from scratch | `python scripts/train.py --config configs/vss-prototype.yaml --train data/generated/train.jsonl --eval data/generated/eval.jsonl --out runs/clean_clone` | **status `done`**, 1093.52 s, eval 1.1107 → **0.2110**, choice 0.772 → **1.000** |
| manifest | `runs/clean_clone/manifest.json` | commit `92e50ca`, dirty flag, config hash `cefdddcfec8f115c`, params 13,655,364, pid, start/end timestamps, `splits {train: 2400, eval: 480, test_used: false}`, best checkpoint present |
| load in a fresh process | `python scripts/evaluate.py --model runs/clean_clone/final --data data/generated/eval.jsonl` | score mean relative error **0.0418** |
| example | `python examples/basic.py runs/clean_clone/final` | `billing` @ 0.9975, `noul` 0 @ 0.9998, `score` 1.017 @ 0.962 |
| CLI | `python -m vss.cli --help` | train, decide, serve, evaluate, validate, generate |
| REST | `TestClient` against the clone's own `src` | `/health` 200 `model_loaded: true` (boolean), `/v1/decide` 200, empty questions / bad type / unknown field all **422** |

`scripts/train.py` was verified byte-identical to the clone's own commit, so the
training run exercised committed code rather than a locally patched copy.

### Verified end to end on the development machine

| Requirement | Evidence |
|---|---|
| Install | `pip install -e ".[dev]"` |
| Generate data | `python scripts/prepare_data.py --out /tmp/vss_verify` → train 2400 / eval 480 |
| Train from scratch | `python scripts/train.py ... --out runs/smoke_verify` → **status `done`**, elapsed 1212 s, eval 1.4240 → **0.2096**, choice 0.703 → **0.9969**, `best.pt` recorded |
| Reproduce from its manifest | `runs/smoke_verify/manifest.json`: config hash `e0967919b7bc941e`, git commit, `dirty`, splits `{train: 2400, eval: 480, test_used: false}`, params, best checkpoint |
| Load in a fresh process | `python examples/basic.py runs/smoke_verify/final` → full typed answer set |
| Evaluate | `python scripts/evaluate.py --model runs/smoke_verify/final --data /tmp/vss_verify/eval.jsonl` → choice accuracy **0.99375**, **macro-F1 0.9928**, ECE **0.0064**, AUROC 0.9989, score mean relative error 0.0413 |
| Run isolation in practice | A second `scripts/train.py` into `runs/smoke_verify` exits **2** with `run directory ... is owned by live pid ...; refusing to write concurrently` |
| REST API | `/health` 200 with a real boolean, `/v1/decide` 200, malformed → 422 |
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

One further behavioural defect was found and fixed during Blocker E, recorded as
D27: **the abstain head does not detect out-of-distribution input.** Measured on
648 states, word-scrambling moved the abstain rate by 0.003 (0.1031 → 0.1062)
and fluent off-domain text drew **zero** abstentions at *higher* confidence
(0.9422) than in-distribution text (0.8904). The claim is withdrawn and the
behaviour is documented as a limitation in the model card, the benchmark report
and the example itself, and pinned by
`tests/test_ood_probe.py::test_abstain_head_is_not_an_ood_detector`.

**Extended to real data for v0.1.0.** D27 was measured on the synthetic
quick-start checkpoint. `scripts/eval_ood.py` now measures the same question on
CLINC150's genuine 1000-utterance OOS test split, with the abstain threshold
selected on the **validation** split only. That measurement found and fixed two
defects in the *measurement code* first (see below), and its result is more
nuanced than D27 alone:

| quantity | value |
|---|---:|
| abstain logit, AUROC vs OOS | 0.6642 (chance 0.5) — **D27 stands** |
| emitted confidence, AUROC vs OOS (test) | **0.8068** |
| selective accuracy at validation-selected threshold | 0.7662 (vs 0.6229 unfiltered) |
| false-abstention rate on in-scope traffic | **0.3309** |
| OOS recall on test vs 0.90 selection target | 0.8010 |
| ECE on answered in-scope | 0.1220 |

So the model exposes a **usable selective-prediction signal** but does **not**
ship a dependable OOD safeguard. Both model cards, the benchmark report (§8),
the claims ledger (D27) and the README state it that way, and
`tests/test_ood_metrics.py` fails the suite if any of them reintroduces a
guarantee claim.

Two defects were found *in the evaluation harness* while producing this, both
now regression-tested:

1. **OOS recall divided by the whole split instead of the OOS subset**
   (`(is_oos & ~keep).mean()`). With 100 OOS among 3100 validation rows that
   understated recall 31x — it reported 0.032 where the truth was 1.000, and
   declared a reachable 90%-recall operating point *unreachable*. It would have
   turned a working gate into a documented dead end.
2. **`auroc` broke ties by array position** (`argsort().argsort()`), so a
   perfectly uninformative score of four identical values reported 0.00 instead
   of 0.50. Discrete confidence estimates tie constantly.

A third, reporting-only defect: when the target operating point was declared
unreachable the script printed `in-scope coverage -1.000`, because the fallback
path never set the coverage it had chosen.

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
| Tests | **148 passed**, 1 skipped (opt-in slow OOD test), 149 collected, 0 failed (`python -m pytest -q`) |
| Fresh-clone verification | commit `92e50ca` in an empty directory: deps → data → train (`done`, 1093.52 s) → load → evaluate → example → CLI → REST |
| Real-data tuning | 20 screen runs + 4 convergence-matched runs; all four LR optima bracketed |
| Verified smoke run | `runs/smoke_verify`: status `done`, 1212 s, choice 0.9969 val, 0.99375 test accuracy, macro-F1 0.9928, ECE 0.0064 |
| Environment | Windows 10, Python 3.11.9, torch 2.14.0+cpu, CPU-only, fp32 |
| Seeds | **1** on every reported result |
| Runs with manifests | all convergence runs + smoke run |
| Valid comparisons | synthetic presentation-matched tie; question isolation; latency ratios within a session |
| Provisional | Banking77 0.870 vs 0.820 (step-matched, not convergence-matched) |
| Withdrawn | CLINC150 comparison; "plain wins 35/35"; all pre-audit real-data order-invariance / interference / selective-accuracy figures |

## Blocker status

| blocker | status |
|---|---|
| A — legacy checkpoints / evaluation discrepancy | **RESOLVED.** Root cause: `header_only_choice` was read from the model config by VSS but from the question object by plain, and `AnsweredQuestion` has no such field, so plain silently trained the *full-inventory CE* task while VSS trained the header-only task. Same checkpoint, same data, only that flag: **0.8900 vs 0.0750**. The checkpoints were always sound; the measurement was wrong. Fix + 11 parity tests in `tests/test_plain_header_only_parity.py`. |
| B — per-dataset LR tuning for both architectures | **DONE.** 20-run screen; all four optima bracketed. |
| C — fair convergence-matched real-data comparison | **DONE** — per-dataset tuned LRs, 2000 steps, **3 seeds on both real datasets**. The result is negative for VSS on both. |
| D — decision-specific behaviour | **DONE.** Heads, schemas, isolation, permutation, calibration and selective prediction exercised; the OOD-abstention claim was measured and withdrawn (D27). |
| E — fresh-environment verification | **DONE.** Full documented chain executed in a clean `git clone` at `92e50ca`; see Gate D. |

## Remaining blockers

1. **Synthetic is single-seed.** Banking77's three seeds are what overturned the
   earlier reading, so the synthetic tie cannot be treated as settled. Both real
   datasets are now at three seeds.
2. **VSS is unstable across seeds on both real datasets** (0.765–0.895 on
   Banking77, 0.675–0.765 on CLINC150) and the cause is undiagnosed. It
   early-stops on bad seeds and peaks on its final epoch on good ones, which
   points at the stopping rule interacting with the LR schedule rather than at
   data or capacity. Until that is understood, no single VSS number should be
   quoted as expected performance.
3. **Macro-F1 and per-class error analysis on real data are still not
   reported.** Risk-coverage *is* now measured on real data — `scripts/eval_ood.py`
   emits it for the CLINC150 checkpoint (`benchmarks/ood/vss-clinc150-s13.json`),
   along with coverage, selective accuracy and OOS AUROC.
4. **There is no dependable OOD safeguard** (D27, extended to real data). The
   confidence signal separates OOS at AUROC 0.807, but the operating point that
   catches most OOS rejects 33% of legitimate traffic and still misses ~20% of
   OOS. A product limitation rather than a release blocker, but binding on any
   out-of-domain deployment — an explicit novelty gate is needed before one is
   attempted.
5. **VSS is behind the plain baseline on both real datasets** and the
   architectural cause is not identified. The CLINC150 symptom (peak at epoch 1,
   then degrade) is documented at three seeds but not explained. This is the main
   research question the project now faces, and it argues firmly against scaling.

## Why research preview rather than release

The engineering is sound, the software works end to end from a clean checkout,
training is properly tuned and converged, and a long-standing measurement defect
was root-caused rather than worked around. What is missing is *evidentiary*, and
what the evidence says is negative: the plain baseline is better on both real
datasets and VSS is far less stable run to run. Shipping this as "release-ready"
would imply the model is fit to deploy as a decision model; it is not, on this
evidence. Shipping it as a research preview with the comparison reported as
found is.