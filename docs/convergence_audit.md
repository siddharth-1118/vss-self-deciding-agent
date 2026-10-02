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
