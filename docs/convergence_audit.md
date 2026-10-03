# Convergence Audit — VSS Training Pipeline

**Scope.** An audit of the training pipeline *before* running any convergence
experiment, so that a measured "VSS is under-trained" conclusion could not be an
artifact of a broken trainer. Every check below was executed against the code as
it stood; probes are reproducible via `benchmarks/convergence/audit_probe.py`
and `audit_probe2.py`, and every defect is pinned by a regression test in
`tests/test_convergence_audit.py`.

**Outcome: 4 defects found, all fixed. Two of them (split leakage, warmup
domination) invalidate part of the earlier multi-question value test and are the
reason its synthetic numbers are being re-measured from scratch.**

---

## 1. Defects found

| # | Severity | Defect | Where it bit |
|---|---|---|---|
| 1 | CRITICAL | Synthetic splits were nested prefixes (`validation == train[:300]`, `test == train`) | synthetic selection + test |
| 2 | MAJOR | Fixed 150-step warmup ate 75% of the 200-step schedule; `lr_lambda(0) == 0` | VSS everywhere |
| 3 | MAJOR | stable-RoPE positions used example 0's state length for the whole batch | variable-length states (real data) |
| 4 | MODERATE | `evaluate()` averaged per-batch means, so selection depended on batch composition | validation signal |
| 5 | **CRITICAL** | epoch loop ignored the global step budget; post-budget epochs ran at `lr=0` and faked an early stop | real data only |
| 6 | MODERATE | mid-epoch checkpoints wrote `history=[]`, wiping a resumed run's per-epoch records | real data only, resumed runs |
| 7 | **MAJOR** | checkpoint selection used total eval **loss**, which for VSS includes calibration + ordinal terms that do not track choice accuracy | VSS on real data; cost up to 8 accuracy points |

Findings 5 and 6 were found **after** the synthetic study was concluded, while
re-running the real-data arms. Neither can appear in a short, uninterrupted run:
both need a dataset large enough that the step budget expires mid-run, and an
interrupt long enough to trigger a resume. That is exactly the profile of the
real-data arms, and exactly the profile the synthetic study did not have.

### Finding 1 — CRITICAL: the synthetic splits were nested prefixes of one another

`generate_synthetic(n_states, seed, split)` called `random.Random(seed)` for
*every* split and then drew `n_states` examples from it. Because each split
restarts the same RNG, the splits are prefixes of a single stream:

```
validation == train[:300]                (True)
test[:800] == train                      (True)
states in train ∩ test:  800 / 1000      (80% of the test split)
states in train ∩ validation: 300 / 300  (100% of the validation split)
```

So **every checkpoint in the multi-question value test was selected on states the
model had been trained on**, and 80% of the states it was then scored on were
also training states. Two consequences:

- the synthetic accuracy numbers (plain 0.970, VSS 0.762 at Q=1) are
  substantially memorisation, not generalisation;
- the "validation loss was still descending at the epoch cap" signal — the exact
  quantity this investigation was commissioned to interpret — was a *training*
  loss curve. A model can descend indefinitely on memorised states, so the
  observation that motivated the whole audit was not evidence of under-training.

The report's dataset section claimed "identical splits" and "gold re-verified at
load time" but never checked cross-split disjointness, and the module docstring
only promised "no state is duplicated within a Q setting".

**Fix.** Per-split RNG offsets (`SPLIT_SEED_OFFSET`), a separate `calibration`
split, and `verify_splits_disjoint()` which raises on any overlap. All four
splits were regenerated; `train ∩ test = 0/1000`. Tests use `out_dir=tmp_path`
so a test can never overwrite the real dataset again (it did, once, before this
guard existed).

### Finding 2 — MAJOR: a fixed warmup consumed 75% of the synthetic schedule

`warmup_steps=150` is a fixed count, but the synthetic VSS run has only
`(800 × 8) / 32 = 200` optimizer steps. The LR therefore ramped for 150 of 200
steps and had 50 steps left to decay:

| run | total steps | warmup | warmup fraction | mean LR over 2nd half |
|---|---:|---:|---:|---:|
| VSS synthetic, as previously run | 200 | 150 | **75.0%** | 2.01e-04 |
| plain synthetic, as previously run | 1600 | 150 | 9.4% | 6.47e-05 |
| VSS clinc150 | 2656 | 150 | 5.6% | 6.04e-05 |
| VSS banking77 | 2269 | 150 | 6.6% | 6.15e-05 |

VSS also received **8× fewer optimizer steps than the plain baseline** on the
same 800 states (200 vs 1600), because the plain trainer materialises one row per
(state, question) pair while VSS consumes a whole multi-question state per step.
That is a legitimate architectural difference in *cost per step*, but combined
with "same 8 epochs" it meant the two systems were not trained on a comparable
budget — the exact unfairness the task forbids.

A second, smaller bug in the same function: `lr_lambda(0)` returned `0`, and
`LambdaLR` applies the lambda *before* the first optimizer step, so **the first
update was a no-op**.

**Fix.** `warmup_steps` is now a cap at `warmup_frac` (default 10%) of the run:
150 → 20 steps on synthetic, and **unchanged on every real-data schedule** (150 is
under 10% of 2269-2656 steps), so this cannot be confused with a real-data
hyperparameter change. Step 0 now gets a non-zero LR. The schedule shape
(`total_steps`, `warmup`) is pinned in the checkpoint, so resuming with more
epochs no longer silently reshapes the LR curve.

### Finding 3 — MAJOR: stable-RoPE positions used example 0's state length for the whole batch

The per-example attention mask (fixed in `df5eb50`) isolates question blocks, but
the "stable positions" code that goes with it computed

```python
state_len = min(spans[0][0][0], seq_len)     # example 0 only
for b, spans_b in enumerate(spans):
    pos[b, s:e] = state_len + arange(e - s)   # every example uses L_0
    pos[b, :state_len] = arange(state_len)   # runs LAST, overwrites
```

For any example whose state is *shorter* than example 0's, the state assignment
runs last and **overwrites the first question block's positions**, and its
question-to-state RoPE distances are shifted by `L_0 - L_b`. That reintroduces
precisely the batch-composition dependence the mask fix was written to remove.

Measured exposure:

| dataset | state-block token length | distinct lengths |
|---|---|---:|
| synthetic | 32 (all) | 1 |
| clinc150 | 7 – 24 | 17 |
| banking77 | 10 – 30 | 16 |

So the defect is **inert on synthetic and live on both real datasets** — i.e. it
was silently degrading exactly the measurements used to characterise VSS's
order-invariance, and it is the most likely source of the residual 1.2-3.5% of
paired decision flips that the previous report attributed to "fp32 noise".

**Fix.** Each example uses its own state length. `tests/test_convergence_audit.py`
asserts both the assigned positions and batch-composition invariance on states
whose lengths genuinely differ.

### Finding 4 — MODERATE: validation loss averaged per-batch means

`evaluate()` averaged `parts["total"]` over batches, so a batch containing one
noul row counted as much as a batch of 32 choice rows. With mixed-type synthetic
batches the selection signal was therefore a function of batch composition, and
not comparable across epochs with different batch shapes.

**Fix.** Validation is row-weighted and reports per-task losses, Choice accuracy,
Noul accuracy, Score MAE, ECE, gradient norms and parameter-update ratios.
Early stopping (`patience`, `min_delta`, `min_epochs`) is configurable.

### Finding 5 — CRITICAL: the epoch loop ignored the global step budget

Found while re-running the real-data arms, after Findings 1–4 had already been
fixed and the synthetic study had been concluded.

`max_steps` is a **global** optimizer-step budget (Finding 2's fix), but only the
*inner* batch loop checked it. The `for epoch in range(...)` loop did not. On any
dataset where the budget is smaller than `steps_per_epoch x epochs`, the budget
runs out partway through an epoch and the outer loop keeps going:

- epoch 0 trains normally until the budget is hit;
- every subsequent epoch runs **one** batch, whose `lr_lambda` is already 0, and
  then evaluates.

banking77 is 9079 train examples at batch 32 = **283 steps/epoch**, so a 400-step
budget expires 1.4 epochs in. The observed banking77 runs reported
`steps=402, epochs_run=4, stopped_early=True`: epochs 2 and 3 were single-batch
no-ops at `lr=0.00e+00` with bit-identical validation loss
(2.1219796562194824 three epochs running), and because each no-op epoch still
incremented `epochs_without_improvement`, **early stopping fired on epochs that
could not have learned anything**. The runs were reported as having
"converged"; they had in fact simply run out of schedule.

The same fault is why the VSS arm's train loss fell 2.05 → 0.49 while eval loss
stayed flat at 2.3443: memorization with a dead learning rate, not convergence.

This also means the *real-data rows of the first sweep are void*, not merely
under-trained — the reported `best_loss`, `epochs_run` and `stopped_early` for
both banking77 arms describe a run that spent 3 of its 4 epochs doing nothing.
They are quarantined in `benchmarks/convergence/runs_invalid_budget400/` and
excluded from `tables.md`.

**Fix.** Both trainers (`src/vss/training/trainer.py`,
`benchmarks/multi_question_value/plain_classifier.py`) now break out of the
epoch loop as soon as `global_step >= max_steps`, printing an explicit
"step budget exhausted" line. Budget exhaustion is no longer reported as
`stopped_early`. Three regression tests in
`tests/test_convergence_audit.py::TestGlobalStepBudgetTerminatesEpochLoop` pin
this: two of them fail against the pre-fix trainer (verified by reverting the
guard).

**Lesson.** The synthetic study was immune because 800 examples at batch 32 = 25
steps/epoch, so a 400-step budget spans 16 epochs and the budget never expires
mid-run. The bug only fires on real datasets, i.e. exactly where the study was
about to make its scaling decision. A cheap guard against this class is to
assert, in any sweep harness, that `steps_run <= max_steps + batch_size`.

### Finding 6 — MODERATE: mid-epoch checkpoints wiped the run history

Also found on banking77, and only after a run survived several wedge/resume
cycles. The mid-epoch step hook called `_checkpoint(..., partial=True)` without
passing `history`, and `_checkpoint` defaults it to `[]`. Every partial
checkpoint therefore overwrote the accumulated per-epoch records with an empty
list, and `fit()` read that back as `history = []` on resume. The run JSON for
`vss-banking77-s13-lr0.0003-st1200-s13` therefore begins at **epoch 1**; epoch 0
(val 2.7865, choice 0.485) survives only in the log, not in the result file.

The model, the optimizer and the pinned schedule were unaffected, so the run's
*conclusions* stand — but the result file under-reports the run, which directly
violates the study's rule that every reported number is regenerated from the run
JSONs. The plain baseline was not affected (its partial save already passed
`history`).

**Fix.** The hook now forwards `history` and `epochs_without_improvement`. The
hard-coded 100-step checkpoint interval became `TrainingConfig.ckpt_every` so the
behaviour is testable. A regression test
(`TestMidEpochCheckpointPreservesHistory`) intercepts checkpoint writes and
asserts no partial checkpoint taken after a completed epoch carries an empty
history; against the pre-fix trainer it fails with
`[(1, True, []), (1, True, []), (2, True, []), (2, True, [])]`.

**Lesson.** This defect class only appears on runs long enough to be interrupted
and resumed — again, i.e. only on real data, and only because this box wedges
every ~26 min of CPU. Neither the synthetic study nor a single-shot run would
have surfaced it.

### Finding 7 — MAJOR: checkpoint selection used a loss that does not measure the reported metric

`best.pt` was chosen by minimum evaluation **loss**, and the early-stopping
counter used the same quantity. For the plain classifier that is nearly
equivalent to accuracy, because its validation loss *is* choice cross-entropy.
For VSS it is not: `combined_loss` sums choice CE with a calibration BCE and a
soft-ordinal term, so the total moves for reasons that have nothing to do with
whether the choice head is right.

Measured on the three-seed Banking77 runs, comparing the loss-selected epoch
with the best-accuracy epoch **inside the same run**:

| system | s7 | s13 | s21 | mean cost |
|---|---:|---:|---:|---:|
| plain | +0.015 | +0.015 | +0.000 | +0.010 |
| VSS | **+0.080** | +0.000 | +0.025 | **+0.035** |

VSS lost up to 8 accuracy points to its own selection rule; plain lost at most
1.5. The asymmetry is structural, not chance: VSS's loss curve is visibly
non-monotonic while its accuracy climbs monotonically. Seed 7 shows it —
accuracy `0.425 → 0.685 → 0.765 → 0.755 → 0.845 → 0.840` against eval loss
`3.05 → 1.75 → 1.548 → 1.96 → 1.564 → 1.77`. The minimum sits at epoch 2
(0.765) while 0.845 was available at epoch 4. The early-stopping counter
inherits the same noise, so the run halted at epoch 6 with accuracy still
improving.

**Fix.** `src/vss/training/selection.py` holds one shared implementation, and
both trainers import it — selecting on validation choice accuracy with loss as
the tie-break, with `loss` still selectable. Sharing one module is deliberate:
the two systems drifting apart in a protocol detail is exactly what produced
`header_only_choice` (D26).

Effect, recomputed from the already-recorded histories so no retraining was
needed (`python benchmarks/convergence/seed_table.py`):

| dataset | system | loss-selected | accuracy-selected |
|---|---|---:|---:|
| Banking77 | plain | 0.8783 (sd 0.0076) | **0.8883** (sd 0.0058) |
| Banking77 | VSS | 0.8300 (sd 0.0650) | **0.8650** (sd 0.0265) |
| CLINC150 | VSS | 0.6750 | **0.7300** |

Roughly half of VSS's Banking77 deficit and most of its seed instability were
its own selection rule. **The headline verdict does not change** — plain still
leads both datasets — but the gap is smaller and better understood, and the
stability claim needed qualifying.

This is validation-based selection only: the test split is never consulted and no
metric definition changes. The reported number is still choice accuracy on
held-out data, now taken from the checkpoint validation says is best for it.

---

## 2. Checks that passed

| Check | Result |
|---|---|
| Real-dataset split leakage | **clean** — clinc150 10625/3100/5500, banking77 9079/924/3080, `train ∩ test` utterances = **0** for both |
| All intended parameters receive gradients | **clean** — 75 parameter tensors, 0 with `grad is None`, 0 with all-zero grad, on a full multi-task batch |
| Multi-task losses finite and sensible | **clean** — total 5.7197 = choice 1.4056 + noul 0.6929 + score 1.8826 + 0.25 × ordinal 4.1609 + calibration 0.6983 |
| Block-isolation mask correctness | **clean** — per-example `[B,1,T,T]` since `df5eb50`, pinned by 2 tests |
| Batch-composition invariance (synthetic) | **clean** — max logit difference 0.000e+00 when the same 8 examples are re-batched against a different first example |
| Tokenizer fitted on training data only | **clean** — `_build_vocab` fits on `self.train[:20000]` |
| Resume preserves optimizer state | **clean**, with two caveats fixed below |
| LR schedule termination | **defect 2** (75% warmup); decay is otherwise correct and monotone |
| Validation metric consistency across checkpoints | **defect 4** |

### Resume caveats (now fixed)

- **Schedule reshaping.** `total_steps` was recomputed from the *current*
  `epochs` on every resume, so "resume with 40 epochs instead of 8" produced a
  different LR curve from an uninterrupted 40-epoch run. Pinned in the
  checkpoint now.
- **Dropout RNG not restored.** The global torch RNG is re-seeded but the
  generator's stream position is not restored, so a resumed run draws different
  dropout masks than an uninterrupted one. Not fixed (it would require storing
  RNG state); it is a reason to treat resumed and uninterrupted runs as
  *statistically* rather than *bitwise* identical, and it applies equally to both
  systems.
- The scheduler is a pure function of `step`, so fast-forwarding
  `sched.step() global_step` times reproduces it exactly — but only once the
  schedule shape is pinned (above).

---

## 3. Convergence criteria (fixed before looking at any new result)

Defined in `benchmarks/convergence/sweep.py` and applied identically to both
systems.

| Criterion | Value | Rationale |
|---|---|---|
| Selection metric | **validation loss**, row-weighted | never the test split |
| Early-stopping patience | 4 epochs | long enough to ride out a cosine shoulder |
| Material improvement | `eval_loss` must drop by > 5e-3 | below this the "best" checkpoint is noise-chasing on a 200-state slice |
| Minimum epochs | 8 | the schedule must have a chance to decay |
| Epoch cap (synthetic) | 40 (1000 steps) | 5× the previous 200-step budget |
| Epoch cap (real) | 16 | 2× the previous 8-epoch budget; real splits are ~2.3-2.7k steps per 8 epochs |
| Validation slice | first 200 states | keeps an epoch inside the shell cap; selection only, scoring is the benchmark's job |
| Seeds (final config) | 3 (1, 2, 3) | 5 was not affordable on CPU within the shell cap |

**Split discipline.** `train` fits weights; `validation` selects the checkpoint
and the stopping point; `calibration` is reserved for abstention thresholds and
temperature (the synthetic set now has a real calibration split; for the real
datasets the last 25% of validation is held out for it); `test` is touched only
by the benchmark. **No hyperparameter, checkpoint, epoch count or abstention
threshold in this study was chosen using a test-split number.**

**Equal-opportunity requirement.** Both systems use the *same* schedule
implementation (`vss.training.trainer.lr_lambda` / `effective_warmup`), the same
warmup cap, the same patience and delta, the same validation slice, and both are
screened over the same learning rates. The baseline is not handicapped by the
defects found in VSS's loop, and not given an advantage VSS does not receive.

---

## 4. Consequences for earlier conclusions

| Earlier claim | Status |
|---|---|
| "Plain beats VSS in 35/35 cells" | **must be re-measured.** 21 of those cells are synthetic, i.e. scored on 80% training states. The verdict may move in either direction: the plain baseline was equally inflated. |
| "VSS validation loss still descending at the epoch cap" | **void as evidence.** The validation split was the training split. |
| "Order agreement 0.964-0.993" | **re-measure.** The RoPE defect (finding 3) was live on both real datasets. |
| "Real-data interference is fp32 noise" | **re-measure.** Finding 3 offers a concrete mechanism for exactly that residual. |
| "Real-data splits are clean" | **confirmed**, independently. |

The full before/after comparison is in `docs/convergence_report.md`.
