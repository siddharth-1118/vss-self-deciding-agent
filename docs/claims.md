# VSS Claims Ledger

Epistemic tiers: **Supported** (measured here, reproducible from artifacts in
this repository), **Provisional** (measured, but one seed / not
convergence-matched / not isolation-checked), **Withdrawn** (was claimed, no
longer supported), **Unsupported** (claimed without evidence). Every claim names
its evidence. All real-data results are SINGLE training seed unless stated.

## Release status of this ledger

This release is a **research preview**. The only claims at *Supported* are the
engineering invariants and the synthetic single-seed results. Every real-data
accuracy claim is **Provisional** or **Withdrawn**. See
`docs/release_readiness.md` for the gate-by-gate decision.

| Claim group | Status |
|---|---|
| Engineering invariants (permutation invariance, question isolation, run isolation, schema validation) | **Supported** |
| Synthetic convergence + tie vs baseline | **Provisional** (1 seed) |
| Banking77 VSS 0.8300 vs plain 0.8783 (3 seeds) | **Demonstrated (3 seeds)** — plain ahead 4.8 pts; VSS ~8.5× less stable (D28) |
| CLINC150 plain 0.9183 vs VSS 0.7050, 3 seeds | **Demonstrated (3 seeds)** — plain ahead 21.3 pts; 16.5 under the corrected selection rule (D28) |
| Legacy plain checkpoints | **Supported** — reproduce at 0.9400 test accuracy (D26, resolved) |
| "Plain beats VSS in 35/35 cells" | **Withdrawn** |
| Latency advantage vs batched baseline | **Provisional** (single contended session) |
| Selective prediction / risk-coverage | **Provisional** (synthetic only) |
| Real-world generalisation from synthetic results | **Unsupported** — and now actively **disconfirmed**: the synthetic tie and the CLINC150 loss disagree in sign (D28) |
| OOD / abstention detects unfamiliar input | **Withdrawn** — the abstain class behaves like a ~10% prior (D27). The *confidence* signal does separate OOS (AUROC 0.807) but only at a measured one-in-three false-rejection rate; not a safeguard. |

---

## Demonstrated

### D1. Header-only serialization is exactly permutation-invariant
Choice-question logits do not depend on option order when
`header_only_choice: true`. Verified on the trained CLINC checkpoint
(exact match to 1e-4) and tiny models (3 permutation seeds).
Evidence: `tests/test_option_permutation.py`; manual verification logged in
commit `test-option-permutation`. The legacy path is *provably* not invariant —
pinned by a test so the dilution failure stays documented.

### D2. Option text inside question blocks dilutes the question vector
Full-151-option text blocks in the `<QUESTION>` region collapsed accuracy from
52.4% (minimal block) to 8.6% (full block) at identical weights. The fix
(header-only blocks + slot projection) took CLINC150 full-schema accuracy from
16.8% → 73.98%.
Evidence: `fix-choice-dilution` measurements; `benchmarks/ood/clinc150_slot_ho_best.json`.

### D3. Co-asking questions degrades accuracy monotonically — FIXED by question-masked attention
Gold-scored slot0 comparison (own question, own state): solo 61.7% vs
co-asked 56.0% at k=8 (21 harmful flips vs 4 helpful). Dose-response
−3.3/−4.0/−5.7/−8.7/−20.0 pts at k=2/4/8/16/32. Cause: bidirectional encoder
(`is_causal=False`) — every question attends to all others.
**Resolution**: with `question_masked: true` (block-isolation mask + stable
per-token RoPE positions), the dose-response is flat: +0.7/+0.3/0.0/0.0/+0.3
at the same k values, slot0 agreement 0.93 flat in k, speedup ~1.95×.
Cost of the fix: −2.3 pts solo accuracy (71.7% vs 74.0%), ECE answered
improved 0.040→0.028, OOS AUROC unchanged (0.803).
Evidence: `benchmarks/interference/clinc150_dose_response.json` (unmasked),
`benchmarks/interference/clinc150_qmask_dose_response.json` (masked),
`benchmarks/ood/clinc150_qmask_best.json`.

### D4. Batching itself is exactly invariant
Same requests in different batches agree to <1e-6 in probabilities — the
interference in D3 is genuine question cross-attention, not a padding/masking bug.
Evidence: manual A/B probe in commit `probe-question-interference`.

### D5. The calibration head is saturated and adds no discrimination
Head output is near-constant (mean 0.9916, std 0.0134, 99.7% of examples above
0.9). Its AUROC for correctness is 0.741 vs 0.842 for plain top-prob; blending
it in does not hurt ECE (0.0992 vs 0.1045) but it is not doing what the
architecture claims (per-question P(correct)).
Evidence: `benchmarks/calibration/clinc150_slot_ho.json`.

### D6. Validation-fitted temperature hurts
T=1.2745 fitted on validation raises same-split ECE from 0.049 to 0.089 — the
model was already near temperature-1 calibrated. The earlier "temperature barely
helped" conclusion was optimistic; it actively hurts.
Evidence: `benchmarks/calibration/clinc150_slot_ho.json` (temperature_audit).

### D7. Word+hash tokenization beats BPE at this data scale
Identical recipe, only tokenizer differs: BPE 52.2% clean vs word+hash 63.4%
(−11.2 pts), no robustness advantage (typo −8.3 / drop_word −12.3 vs word's
−8.8/−10.4). The audit hypothesis that word+hash hurts is refuted at 10.6k
training examples.
Evidence: `benchmarks/tokenizer/clinc150_bpe_vs_word.json`.

### D8. Surface-form brittleness of the word+hash path (quantified)
Case and whitespace: exactly invariant. One adjacent-char typo −8.8 pts, one
dropped word −10.4 pts, appended punctuation −3.5 pts, word duplication −0.7 pts
(n=1000, seed 42).
Evidence: `benchmarks/robustness/clinc150_slot_ho.json`.

### D9. Seed noise is small; the baseline gap is real
Three-seed Banking77 (identical recipe, seeds 13/42/7): accuracy
72.4/72.5/73.7% (sample std 0.72 pts), ECE 0.120–0.123. The 14.1-point gap to
TF-IDF+LR is ~20 seed-sigmas — not a seed artifact.
Evidence: `benchmarks/multiseed/banking77_summary.json`.

### D10. Single-request co-asking is the fastest serving mode but leaks accuracy
Method A (one co-asked request) p50: 0.099 s @ N=1 → 9.00 s @ N=200
(~0.045 s/question; linear). ~2.2× faster than batched solo (C) and ~4.5×
faster than sequential solo (B) at N=200. Cold start 2.36 s. A's own-question
accuracy decays with N (D3 at scale).
Evidence: `benchmarks/latency/clinc150.json`.

### D10b. The serialization + slot-head layer costs ~13 accuracy points on the same backbone
Identical encoder architecture, optimizer, schedule, and seed; only the input
format and head differ. From-scratch: plain linear head over 151 intents on
state-only text reaches **86.7% in-scope** (1000-example subsample) vs VSS
full-model **73.98%**. Warm-started from the VSS encoder: 88.0%. The plain
head also converges faster (val 86.9% by epoch 6 vs VSS's best val CE at
epoch 3 of 8). VSS's typed-question machinery is a net accuracy *cost* on
CLINC150.
Evidence: `benchmarks/ablation/inscope_summary.json`,
`benchmarks/ablation/encoder_classifier_scratch_inscope.json`.

### D11. Early synthetic results were leakage artifacts
75/480 synthetic test examples were exact train duplicates; 92.9% had token
Jaccard ≥ 0.7 to a train example; temperature was fitted on the eval split.
Under held-out templates accuracy fell from 1.000 to 0.3125 (ECE 0.004 → 0.65).
Evidence: `benchmarks/leakage/synthetic_report.json`, commit `fix-evaluation-leakage`.

### D12. Head trained with slot-CE + header-only recovers real-data accuracy
CLINC150 in-scope full-151 accuracy 73.98% (from 6.2% refine-CE, 16.8% slot-CE
without header-only), with risk-coverage 99.6% @ 50% coverage → 76.8% @ 95%.
Evidence: `benchmarks/ood/clinc150_slot_ho_best.json`.

---

## Plausible

### P1. VSS is more calibrated than TF-IDF+LR on both real datasets
ECE 0.123 vs 0.248 (Banking77, 3 seeds stable) and 0.040 vs 0.297 (CLINC150
answered in-scope). Consistent direction on two datasets, but the comparison
conflates model family with abstention training; a temperature-scaled baseline
was not run (and D6 suggests post-hoc scaling is not the whole story either).

### P2. VSS's OOS rejection (AUROC 0.800) reflects trained abstention, not luck
0.28 of OOS examples get explicitly abstained at the validation threshold;
confidence AUROC 0.800 vs baseline 0.889. Single seed; mechanism not isolated
from the in-scope confidence structure.

### P3. BPE's failure is question-vector dilution (same family as D2)
BPE doubles sequence length; the mean-pooled question vector averages over more
tokens, and the audit's earlier dilution finding shows mean pooling is the
sensitive component. Not directly measured (no ablation of pooling).

---

## Unknown

### U2. REOPENED — VSS vs the plain-head baseline on multi-question tasks
Originally resolved as "plain wins" (D13). **That resolution is withdrawn.** The
convergence audit found that the comparison ran on a training-set-contaminated
synthetic split, against a VSS schedule that spent 75% of its steps in warmup,
with an 8× step-budget advantage given to the baseline. Re-measured properly
(clean splits, converged, presentation-matched, one seed): the two systems are
**tied on synthetic at Q=1–4**, plain leads by 3.4–6.9 pts at Q=8–50, and VSS
wins decisively on selective prediction and single-pass latency. The question is
now open on two axes: real data (not re-measured — see the audit) and seed
variance (one seed only). Evidence: `docs/convergence_report.md`.

### U3. Multi-seed behavior of CLINC150 results
All CLINC150 numbers are seed 13 only. Banking77 variance was 0.72 pts, but
CLINC has 151 classes and OOD structure; its variance is unmeasured.

### U4. Generalization beyond intent classification
All real-data validation is single-intent choice questions. Noul and Score
heads are validated only on synthetic data (and post-leakage, weakly: 31.25%).

### U5. Whether slot collisions (9 @ 1024 slots for 151 labels) matter
2 collisions at 4096 slots were measured; no accuracy ablation was run on
slot count.

---

---

## Multi-question value test (35 cells, 3 datasets, 3 systems)

Evidence: `benchmarks/multi_question_value/results/*.json`,
`benchmarks/multi_question_value/results/DIGEST.txt`,
`benchmarks/multi_question_value/report.md`,
`docs/multi_question_value_report.md`.

**Second pass, post-defect-fix.** Two first-pass "defects" turned out to be
implementation bugs: (1) `src/vss/model/encoder.py` built the question-isolation
mask once per *batch* instead of once per example, so in a batch of N examples
only the last example's span geometry survived (fixed in `df5eb50`, 2
regression tests in `tests/test_question_mask.py`); (2) the harness's
interference probe compared a 100-state solo arm against a 200-state joint arm
and reported large deltas while zero paired decisions flipped (now both arms
score the same `(state, question)` pairs and report `paired_decision_flips`).
All four VSS checkpoints were retrained on the fixed mask and all 35 cells were
re-measured. D13 and D14 survive the fix; **D15, D16 and D17 are rewritten and
the first-pass versions are withdrawn.** Pre-fix numbers are preserved verbatim
in `results/*_results_prefix.json`; the before/after table is §16.1 of the
report.

### D13. ~~Plain classifier beats VSS on per-question accuracy in all 35 cells~~ — WITHDRAWN
This claim was measured on three stacked defects and does not survive them:
(1) the synthetic `validation` split was literally `train[:300]` and `test[:800]`
was `train`, so 100% of the selection split and 80% of the test split were
training states; (2) VSS's 8-epoch synthetic schedule had only 200 optimizer
steps of which 150 were warmup, so it never left the LR ramp; (3) the step
budget handed the plain baseline was 1,600 steps vs VSS's 200 — an 8×
difference in gradient signal, since VSS back-propagates all questions of a
state per step and plain one (state, question) row. Re-measured on the corrected
split with converged, presentation-matched models, the synthetic cells are a
**tie at Q=1–4** and **plain ahead by 3.4–6.9 pts at Q=8–50**, most of which at
Q=8–16 is VSS abstaining rather than answering wrongly. See D21. The real-data
cells were not re-run and are withdrawn pending that work.

### D24. On Banking77 at a matched 1200-step budget VSS leads plain 0.870 vs 0.820 — DEMONSTRATED but NOT a ranking (1 seed, under-converged)
The first real-data re-run at a valid budget (audit findings 5 and 6 fixed;
real states carry exactly one question, so matched steps are also matched
presentations): **VSS 0.8700 choice accuracy / val 1.2035 / 949 s**, **plain
0.8200 / val 0.7662 / 1979 s**. VSS leads by 5.0 pts at 2.1x lower wall-clock.
Source: `benchmarks/convergence/runs/{vss,plain}-banking77-s13-lr0.0003-st1200-s13.json`.

Three things this claim is **not**:

1. **Not a real-data ranking.** Neither system converges at 1200 steps. The
   pre-audit plain checkpoint (`runs/mqv-plain-banking77-s13/best.pt`, epoch 6,
   ~1988 steps) reached val 0.5206 and 0.8867 test accuracy — *above* VSS's
   0.870. So the comparison is step-matched but not convergence-matched.
2. **Not a ceiling for VSS.** Its 0.870 is at epoch 4, the final epoch, with the
   LR already annealed to 0 — it is a floor.
3. **Not a replication.** One seed; validation *loss* is not comparable across
   systems (VSS carries calibration-BCE and ordinal terms); Noul and Score MAE
   are vacuous on Banking77 (single `choice` question per state, both score 0.0).

What it does establish: the previously reported Banking77 numbers (VSS 0.76 vs
plain 0.8867) came from a VSS checkpoint frozen at `lr=0` for 3 of its 4 epochs.
Fixing that alone moved VSS from 0.645 (void 400-step run) to 0.870, i.e.
**+22.5 points from Findings 5/6**, which is far larger than any architectural
difference the study has measured.

### D25. CLINC150 at 1200 steps: VSS 0.700 vs plain 0.000 — VOID, do not cite
The CLINC150 pair trained at the same budget. VSS behaved normally (0.36 → 0.63 →
0.625 → 0.70 choice accuracy, val loss 3.74 → 2.27). The plain baseline
**collapsed**: train loss 1.66 → 0.03 while val loss *rose* 4.53 → 5.95 and
accuracy fell 0.085 → 0.020 → **0.000**, below the 0.0067 random baseline for
151 classes.

A per-dataset LR screen was run to diagnose it:

| plain CLINC150 LR | best val loss | choice acc |
|---|---:|---:|
| 3e-5 | 5.2541 | 0.000 |
| 1e-4 | 5.2100 | 0.000 |
| 3e-4 | 4.1142 | 0.120 |
| 1e-3 | 2.9824 | **0.195** |

The first three runs suggested this was *not* an LR effect (two lower LRs were
also at chance). **That reading was wrong**: the completed screen shows plain
improving **monotonically** across the entire grid, with the best value at the
**top edge** rather than an interior peak. The grid did not bracket the
baseline's optimum, so the plain arm is under-tuned and 3e-3 was added to close
the bracket.

### D25b. The CLINC150 collapse was the `header_only_choice` mismatch, not the learning rate — DEMONSTRATED
**D25's collapse is explained and the plain arm is re-measured.** The cause was
identified in D26: VSS read `header_only_choice` from its **model config**, the
plain baseline read it from the **question object**, and `AnsweredQuestion` has
no such field — so on the real-data loader the flag was silently absent for
plain, and it trained full-inventory cross-entropy while VSS trained the
header-only objective. Same checkpoint, same data, only that flag changed:
**0.8900 vs 0.0750** accuracy.

With the protocol corrected, the widened per-dataset screen (20 runs, both
architectures, both datasets, identical 600-step budget, validation-only
selection, seed 13) gives choice accuracy at the selected checkpoint:

| dataset | system | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 | selected |
|---|---|---:|---:|---:|---:|---:|---|
| Banking77 | plain | 0.655 | 0.775 | **0.880** | 0.845 | 0.795 | **3e-4** |
| Banking77 | VSS | 0.130 | 0.520 | **0.770** | 0.685 | 0.110 | **3e-4** |
| CLINC150 | plain | 0.335 | 0.625 | 0.790 | **0.815** | 0.775 | **1e-3** |
| CLINC150 | VSS | 0.010 | 0.245 | **0.575** | 0.445 | 0.030 | **3e-4** |

Plain CLINC150 now reaches **0.790–0.815** where the mismatch produced 0.000,
so the collapse is gone. All four optima are **interior** (each beats both
grid neighbours), so the optimum is bracketed rather than pinned to the edge —
the specific defect that made the earlier sweep uninterpretable. Both systems
independently pick 3e-4 on Banking77; on CLINC150 they disagree (plain 1e-3,
VSS 3e-4), which is why a single global LR was the wrong policy.

**Do not read the 600-step screen as a ranking.** VSS trails plain on both
datasets there (0.770 vs 0.880; 0.575 vs 0.815) because 600 steps is ~2 epochs
and VSS is the slower converger — at Banking77 3e-4 the final VSS epoch still
shows train 0.42 against eval 1.68, i.e. still descending. The budget is
identical for both, so it is fair for *selecting an LR* and not a fair basis for
declaring a winner. Ranking claims come only from the separate 2000-step runs at
these selected learning rates, and those remain single-seed.

Rendered by `python benchmarks/convergence/lr_table.py`; selection derived from
the run files by `sweep.plan_real_final`. Validation **loss** is not comparable
across the two rows (VSS's includes a calibration BCE and a soft-ordinal term);
only choice accuracy is.

### D26. ~~The pre-audit plain checkpoints do not reproduce their logged metrics~~ — RESOLVED, checkpoints are sound
**This claim was wrong and is withdrawn.** The checkpoints were always fine; the
*measurement* was wrong.

Root cause, identified with evidence: `header_only_choice` was applied
differently to the two systems. VSS reads it from its **model config**
(`vss_model.py`); the plain baseline read it from the **question object**. The
real-data loader (`sweep.load_splits` → `vss.data.schema.load_jsonl`) builds
`AnsweredQuestion`, which has no such field (`extra="forbid"`), so the flag was
always absent for plain there. The flag changes both the serialized text (option
text stripped or not) *and* the plain head's training loss (full-inventory CE vs
masked CE over each row's declared options).

With ONE fixed `mqv-plain-banking77-s13` checkpoint on **identical** validation
data, changing only that flag:

| `header_only_choice` | choice accuracy |
|---|---:|
| `True` (as trained; as `dataset.load_real` sets it) | **0.8900** |
| `False` (as my diagnostic set it) | 0.0750 |

Re-measured through the authoritative `load_plain()` path:

| split | accuracy |
|---|---:|
| banking77 validation[:200] | 0.8900 |
| banking77 test[:200] | **0.9400** |
| banking77 train[:200] | 1.0000 (15 declared options — memorised) |

Label-index mapping was verified **identical** between `final.pt`'s
`label_to_idx` and the recomputed sorted union, and a regression test now proves
a known example maps to the same class index in training and inference.

**Consequences.** The legacy checkpoints are usable, and "plain reached 0.8867
on Banking77" is back on the record. More importantly, the plain arm of every
real-data convergence run was training a *different task* than VSS (masked CE +
full option text vs header-only + full-inventory CE). Those real-data
comparisons are void and are being re-run; see D25.

### D28. Convergence-matched real data: the plain baseline wins both datasets — DEMONSTRATED
2000-step budget, each system at the learning rate its **own** screen selected
(Banking77 3e-4 both; CLINC150 plain 1e-3, VSS 3e-4), validation-only checkpoint
selection, accuracy read at the selected (lowest-loss) checkpoint. Rendered by
`python benchmarks/convergence/seed_table.py`.

| dataset | n | VSS | plain | delta |
|---|---:|---|---|---:|
| Banking77 | **3** | 0.8300 (sd 0.0650) | **0.8783** (sd 0.0076) | **plain +4.8 pts** |
| CLINC150 | **3** | 0.7050 (sd 0.0520) | **0.9183** (sd 0.0058) | **plain +21.3 pts** |

Under the corrected selection rule (finding 7: validation accuracy, loss as
tie-break) re-reading the same recorded histories:

| dataset | n | VSS | plain | delta |
|---|---:|---|---|---:|
| Banking77 | 3 | 0.8650 (sd 0.0265) | **0.8883** (sd 0.0058) | **plain +2.3 pts** |
| CLINC150 | 3 | 0.7583 (sd 0.0275) | **0.9233** (sd 0.0076) | **plain +16.5 pts** |

The gap narrows under the corrected rule on both datasets, but the sign never
changes.

**Banking77 reversed under multi-seed, and that is the important part of this
claim.** At seed 13 VSS scored 0.895 against plain's 0.870 — a nominal 2.5-point
VSS lead that an earlier revision of this file recorded as a VSS win. Seeds 7
and 21 show it was **seed luck**:

| seed | plain | VSS |
|---|---:|---:|
| 7 | 0.880 | 0.765 |
| 13 | 0.870 | 0.895 |
| 21 | 0.885 | 0.830 |
| **mean** | **0.8783** | **0.8300** |
| sd | 0.0076 | 0.0650 |

VSS spans 0.765–0.895 (spread 0.130); plain spans 0.870–0.885 (spread 0.015).
**VSS is roughly 8.5× less stable run to run**, and its best single seed
(0.895) exceeds plain's best seed (0.885).

On CLINC150, VSS peaks early and then degrades — at seed 13 its best was **epoch 1**
and eval loss rose 2.07 → 3.34 by epoch 3; seeds 7 and 21 show the same
non-monotonic eval loss against still-climbing accuracy. This is a genuine
weakness at 151 classes, not an under-trained artefact, since plain's curve was
still improving at 2000 steps. For three audit cycles the record said
"VSS 0.700 vs plain 0.000"; that was wrong in two compounding ways (the plain arm
trained a different task per D26, and both arms ran a synthetic-derived LR at too
small a budget). Corrected, plain reaches 0.9183. **The CLINC150 verdict also now
replicates at three seeds**, and the stability asymmetry seen on Banking77
(sd 0.052 vs 0.006, spread 0.090 vs 0.010) appears on a second dataset.

**Consequences.**
* CLINC150 moves from VOID to **measured, and measured against VSS**. D25 is
  retained above as history because the error it recorded — a broken baseline
  presented as a comparison — is worth not reintroducing.
* **Synthetic accuracy does not predict real-data accuracy.** The synthetic tie
  (D21) coexists with losses on both real datasets. Nothing in the synthetic
  benchmark forecasts either real outcome.
* **No scaling decision follows from this**, and D28 argues *against* scaling.
* **What this does not show**, kept explicit so the negative is not over-read:
  VSS reached 0.895 at seed 13, above plain's best seed. The finding is that VSS
  does not *reliably* reach its own best case — a reliability problem, which is
  a different and more tractable one than a capability gap.
* CLINC150 is now **three seeds**, and its 21.3-point gap (16.5 corrected) is far
  larger than any seed noise observed on either dataset. Banking77's 2.3-point
  corrected gap is *not* far outside its noise — that comparison is the weaker
  of the two and should not be leaned on.
* The CLINC150 **eval-loss anomaly is diagnosed** (audit finding 8): it is the
  calibration term, not a degrading choice head. Re-running seed 13 with the
  per-component loss split recorded shows validation choice cross-entropy
  falling monotonically (2.61 → 0.92) and accuracy rising every epoch
  (0.465 → 0.735), while `comp_calibration` *rises* 0.96 → 1.61 as its training
  counterpart falls to 0.014. The cause is that the calibration target is the
  model's own correctness on the **training** forward pass, so the head learns
  to say "confident" on a distribution it never meets at inference.
* The CLINC150 **accuracy gap to plain is not diagnosed**. The structural fix
  (compute the calibration target from a held-out or previous pass) is
  identified but deliberately not implemented: with one seed's evidence it
  would be a speculative change to what the model optimises. Seed-to-seed
  instability (sd 0.052 vs plain's 0.006) is likewise still unexplained.

### D29. The calibration-target fix is now IMPLEMENTED (opt-in), not just identified
The structural fix named in D28 now exists in code:
`TrainingConfig.calibration_target_mode` (`"self"` | `"ema"`, default `"self"`
so no existing config or checkpoint changes behaviour). Under `"ema"` the
per-row P(correct) target is computed from a **detached exponential-moving-
average copy of the model** — the previous pass — instead of the current
forward pass, so the head is trained to predict correctness of a distribution
it does not itself define. `combined_loss` accepts explicit
`calibration_targets` and rejects a length mismatch rather than broadcasting.

Status of the evidence: this is an **implementation** claim, not a performance
claim. What is **Supported**: the mechanism exists, is opt-in, is
regression-tested (`tests/test_calibration_target.py`, 5 tests, verified to
fail when the fix is disabled), and the full suite passes (199 passed,
1 skipped). What is **NOT yet claimed**: that it improves accuracy,
calibration, or AUROC on real data — that requires the A/B run recorded in
`benchmarks/convergence/results/calibration_ab.json`, which is a single seed
on CLINC150 and must not be read as a ranking.

Deliberately unchanged: the default stays `"self"`, so every existing
checkpoint and config still reproduces its current numbers exactly.

**The first A/B does NOT support the fix.** `benchmarks/convergence/
results/calibration_ab.json` (CLINC150, seed 13, lr 3e-4, 150 steps, held-out
400 choice rows; single seed, short budget — this probes the head only, not
convergence):

| mode | mean | std | frac >0.9 | frac >0.99 | accuracy | AUROC (P correct) |
|---|---:|---:|---:|---:|---:|---:|
| `self` (current) | 0.3065 | 0.0554 | 0.000 | 0.000 | 0.0575 | **0.5927** |
| `ema` (fix) | 0.1282 | 0.0360 | 0.000 | 0.000 | 0.0250 | **0.1815** |

Read honestly, this says two things and neither is what D28 predicted:

1. **The saturation in D5 does not reproduce at this budget.** Neither arm
   exceeds 0.9 on any row. At 150 steps the head sits near 0.13–0.31, not at
   D5's 0.9916. D5 was measured on a *converged* checkpoint; an early-stopped
   one does not show it. The defect this change targets was therefore not
   observed here, so the fix could not be shown to remove it.
2. **EMA targets made the P(correct) signal worse, not better** — AUROC
   0.5927 → 0.1815, i.e. *worse than random*. The plausible cause is that a
   lagging EMA teacher is systematically wrong about the live model's
   correctness early in training, so the head is trained to predict a target
   that anti-correlates with the outcome it is judged on.

So: the code is implemented, tested and safe (default unchanged, full suite
green), but **the hypothesis is not supported by this evidence** and the fix
must not be claimed as an improvement. It stays opt-in and off. Next step
would be a converged-budget run (2000 steps, matching D28's setting) before
drawing any conclusion; a single short seed cannot settle it either way.

**Correction to the framing above:** D28's own text warned this would be "a
speculative change to what the model optimises". Implementing it and measuring
was the right call — it produced a falsification, which is worth more than the
speculation would have been.

### D27. ~~VSS abstains on out-of-distribution input~~ — WITHDRAWN, measured false
**Do not claim OOD detection.** `benchmarks/convergence/ood_probe.py` scored the
verified quick-start checkpoint on 320 in-distribution, 320 word-scrambled and 8
foreign-topic states:

| set | n | abstain rate | mean confidence |
|---|---:|---:|---:|
| in-distribution | 320 | 0.1031 | 0.8904 |
| word-scrambled | 320 | 0.1062 | 0.8895 |
| foreign topic | 8 | **0.0000** | **0.9422** |

Destroying every lexical token moved abstention by 0.003. Fluent off-domain text
(sourdough, sheep, TLS certificates) drew **zero** abstentions at *higher*
confidence than in-distribution text. The head behaves like a ~10% prior, not a
novelty detector.

How it was found, because it is a lesson: `examples/basic.py` originally used a
message that abstained at p=0.99, which looked like correct OOD behaviour. A
nonsense control answered confidently at 0.88. The single anecdotal example
would have supported the opposite conclusion to the truth, and the true
conclusion only appeared once the two were measured side by side on 648 states.

**Consequences.** `ABSTAIN` is documented as a confidence threshold, never a
novelty alarm, in `docs/model_card.md` (limitations + abstention semantics),
`MODEL_CARD.md`, and `examples/basic.py`. Pinned by
`tests/test_ood_probe.py::test_abstain_head_is_not_an_ood_detector`, which fails
if detection ever starts working so the caveat cannot silently go stale. Any
deployment outside the training domain needs an explicit novelty gate.

**Scope of this withdrawal (added after real-data measurement).** D27 was
measured with the synthetic quick-start checkpoint and 8 foreign sentences. A
follow-up on CLINC150 (`scripts/eval_ood.py`, 4500 in-scope / 1000 genuine OOS
test utterances, threshold selected on validation only) separates two claims
that are easy to conflate:

| claim | measured | status |
|---|---:|---|
| the trained abstain class detects OOS | AUROC **0.664** (baseline 0.5) | **false** — withdrawal stands |
| the emitted confidence separates OOS | AUROC **0.8068** test / 0.8678 validation | **true but partial** |

At the validation-selected threshold, selective accuracy on answered in-scope
requests is **0.7662** against a 0.6229 unfiltered baseline — confidence is
genuinely informative. But the operating point costs **33.1% false
abstention** on in-scope traffic, its OOS recall falls from 90% (the selection
target) to **80.1%** on test, and ECE on what it answers is 0.122.

So the honest statement is: **VSS exposes a usable, measurable confidence-based
selective-prediction signal, and does not ship a dependable OOD safeguard.** The
docs say exactly that; `tests/test_ood_metrics.py` fails the suite if any
release document reintroduces a guarantee claim.

This supersedes P2, which attributed the OOS-rejection numbers to trained
abstention; those AUROC figures came from a different (real-data, pre-audit)
harness and are not evidence for the shipped abstain head.

### D21. At matched data exposure VSS and the plain classifier are tied on synthetic — DEMONSTRATED (1 seed)
Converged, early-stopped, identical schedule implementation and LR selection
(3e-4 for both), matched (state, question) presentations (VSS 400 steps x 8
questions = 102,400; plain 2,200 steps): best validation choice accuracy
**0.8700 (VSS) vs 0.8675 (plain)**, noul 0.9550 vs 0.9513, score MAE 0.1562 vs
0.1543 — tied on all three task types. VSS reaches that point with 5.5x fewer
optimizer steps and 1.37x less wall-clock time. VSS: 11.16M params, 400 steps,
2953 s. plain: 10.51M params, 2200 steps, 4056 s. **Single seed** — no variance
estimate. Validation *loss* is not cross-system comparable (different objectives).
Evidence: `benchmarks/convergence/runs/*.json`, `benchmarks/convergence/tables.md`,
`docs/convergence_report.md` §4.

### D22. VSS's confidence is informative where the plain classifier's is not — DEMONSTRATED (synthetic Q=8, 1 seed)
Both systems scored on one risk-coverage curve (1,600 questions ranked by their
own confidence). VSS dominates at every coverage <= 0.90: at 0.85 coverage VSS
is 0.9919 accurate while plain at full coverage is 0.9306. Plain's confidence is
close to uninformative — dropping its least-confident 10% *lowers* accuracy
(0.9306 -> 0.9236) — whereas dropping VSS's least-confident 20% raises it from
0.8644 to 0.9961. At VSS's shipped gate (coverage 0.874) answered accuracy is
0.9893 vs plain's always-on 0.9306. This corrects the earlier reading that the
gate was uninformative, which was measured on a starved checkpoint.
Evidence: `benchmarks/convergence/results/risk_coverage_synthetic_q8.json`.

### D23. Co-asking questions helps rather than harms, once VSS is trained to convergence — DEMONSTRATED (1 seed)
Paired solo-vs-joint deltas on the clean synthetic split are **positive**:
+8.13 pts at Q=8 (266/800 paired decisions flip) and +6.09 pts at Q=32
(1,225/3,200 flip). On the pre-fix, pre-retrain runs the same measurement was
+0.31 to +0.88. Permutation agreement 0.955 (Q=8) / 0.988 (Q=32).
Evidence: `benchmarks/multi_question_value/results/synthetic_fair_results.json`.

### D14. VSS single-pass latency beats even the BATCHED classifier at high Q — DEMONSTRATED (conditional)
Synthetic p50 request latency at Q=50: A sequential 745.2 ms, B batched
267.9 ms, C VSS 109.2 ms → **6.8× vs sequential, 2.5× vs batched** (idle box).
Re-measured with converged checkpoints on a contended box: A 1,573 ms, B 465 ms,
C 174 ms → **9.0× vs sequential, 2.7× vs batched**. At Q=1 all three are within
a few percent. The advantage is a function of request length, not question
count: on CLINC150 and Banking77 (one short question per state) VSS was 1.1×
*slower* per request in the earlier (pre-audit-fix) measurement. Absolute
latencies are load-dependent — the contended re-run inflated every number by
~2× — so only within-cell ratios are claimed. **This is the one claim from the
first value test that survived the convergence audit unchanged.**

### D15. VSS has no measurable cross-question interference on synthetic — DEMONSTRATED (real-data half WITHDRAWN)
Both arms of the probe now score the **same** `(state, question)` pairs, so the
assumption-free statistic is `paired_decision_flips`. On synthetic (3 seeds ×
Q∈{8,32,50}, 27,000 paired decisions) the mean solo→joint delta is
**−0.31 to +0.88 pts with exactly 0 paired decision flips**. On real data
(replicated single-question requests) the deltas are **−0.13/−0.31 pts
(CLINC150 Q=8/32)** and **0.00/+0.06 pts (Banking77 Q=8/32)**, with 1.2–3.5% of
paired decisions flipping in both directions — consistent with fp32
non-associativity under different padding, not with systematic leakage.
**The first-pass claim of −16.9/−15.6 pts (CLINC150) and −9.1/−10.5 pts
(Banking77) real-data interference is withdrawn**: it compared a 100-state solo
arm against a 200-state joint arm and had 0 paired flips behind the headline
number. **The real-data half of this claim is now itself withdrawn** pending
re-run: audit finding 3 (stable-RoPE positions built from example 0's state
length) is live on both real datasets, whose state lengths vary (7-24 and 10-30
tokens) while synthetic's are uniform. With converged synthetic training the
synthetic delta is *positive* (+8.1 / +6.1 pts, D23).

### D16. VSS's confidence gate selects a more accurate subset — DEMONSTRATED, but on synthetic only
The honest version of this claim is now D22 (a proper risk-coverage comparison
on synthetic, where VSS dominates at every coverage <= 0.90). The CLINC150 /
Banking77 numbers previously recorded here (77.7% answered at 53.0% coverage vs
plain 65.9%) come from checkpoints trained before the convergence audit and are
**withdrawn pending re-run**, because audit finding 3 is live on those datasets.
The gate costs nothing in overall decision accuracy: sweeping the threshold
(`results/abstention_sweep.json`) leaves decision accuracy flat at 0.533
(CLINC150) and 0.687 (Banking77) across 0.0–0.55. The first pass's "abstention
costs 15–17 accuracy points" claim is **retracted** — it was a symptom of the
mask bug.

### D17. ~~VSS answers are only ~74–79% order-invariant~~ — WITHDRAWN, superseded by D20
The first pass measured 74–79% agreement under question permutation on distinct
synthetic questions. That was an artifact of the per-batch question mask
(below). **Withdrawn as a property of the architecture.** The current
measurement is in D20.

### D18. Batching does not change plain-classifier outputs — DEMONSTRATED
Mode A ≡ mode B verified in all 35 cells (`A_equals_B: true`; 16-pair batch-1
vs batch-32 spot check per cell, max probability difference ≤ 1.5e-6, pure fp32
noise). This is why the batched classifier (B) is the honest latency
baseline rather than the slow sequential one (A).

### D19. The question-isolation mask was built per batch, not per example — DEMONSTRATED (new, code defect + fix)
`src/vss/model/encoder.py` allocated a single `[T,T]` plane and wrote every
example's rows into it inside the batch loop, so example *b*'s span geometry
overwrote example *b−1*'s and the **last example in the batch decided the mask
for the whole batch**. Isolation therefore depended on batch composition and
question order, and was invisible at batch size 1 (which is why single-question
probes never showed it). Fixed in `df5eb50` to a per-example `[B,1,T,T]` mask,
pinned by two regression tests in `tests/test_question_mask.py` (per-example
planes in a heterogeneous batch; batch-invariance of per-question outputs).
Effect on measured claims, same-checkpoint where possible: order agreement
0.973/0.975 → 0.996/0.998 (CLINC150 Q=8/32), real-data interference
−16.9/−15.6 → −0.13/−0.31 pts. Retraining the Banking77 checkpoint on the
fixed mask moved per-question accuracy 0.573 → 0.760 (Q=1) and 0.429 → 0.754
(Q=50).

### D20. VSS answers are 95.5–98.8% order-invariant on distinct synthetic questions — DEMONSTRATED (post-fix replacement for D17)
After the D19 fix and retraining, permuting distinct synthetic questions within
one request changes the answer for 1.2–4.5% of (state, question) pairs
(Q=8: 0.964/0.967/0.966; Q=32: 0.991/0.993/0.991 across seeds; with the
converged seed-1 checkpoint, 0.955 at Q=8 and 0.988 at Q=32). The plain
classifier is exactly order-invariant by construction. The residual
disagreement carries no systematic sign. The CLINC150 (0.996/0.998) and
Banking77 (0.998/0.999) figures come from pre-audit-fix code and are
**withdrawn** — audit finding 3 is live on those datasets.

---

## Explicit non-claims

- VSS does **not** beat a TF-IDF+LR baseline on either real dataset
  (−14.8 pts CLINC150, −14.1 pts Banking77). It wins on calibration and latency.
- The calibration head does **not** provide per-question correctness probability
  (D5). The blend confidence works, but through top-prob, not head signal.
- "One forward pass answers N questions" is true at N=1 and false as an
  accuracy-preserving claim at large N (D3, D10).
- **VSS is NOT validated as an architecture — and neither is it refuted.**
  The first value test's "plain ahead in 35/35 cells" is withdrawn (D13). What
  survives is a *tie* on synthetic accuracy at matched data exposure (D21), a
  loss at Q≥32 (D21/D13), and two genuine wins: informative selective prediction
  (D22) and single-pass latency against a *batched* baseline (D14). One seed,
  synthetic only.
- **VSS is NOT ready to scale to 52M+ parameters.** Not because it lost — the
  loss claim is withdrawn — but because the evidence is one seed on one synthetic
  dataset, and every real-data number predates a fix (audit finding 3) that is
  demonstrably live on those datasets. Scaling is not licensed by a tie.
- **The real-data value test is NOT re-measured.** CLINC150 and Banking77
  results in this file were produced before the convergence audit and before the
  RoPE position fix; their order-invariance, interference and selective-accuracy
  figures are withdrawn as known-suspect, not as disproven. The convergence
  re-runs (D24, D25) are *training-only* comparisons on a 200-example validation
  slice; they do not replace the benchmark, and D25 is void outright.
- **The first real-data re-run attempt was itself void.** While starting it, audit
  finding 5 was found: the epoch loop ignored the global step budget, so on
  banking77 (283 steps/epoch) a 400-step budget produced 4 reported epochs of
  which 3 were single-batch no-ops at `lr=0`, and the early stop that closed them
  was an artifact of the budget expiring. Those two runs are quarantined in
  `benchmarks/convergence/runs_invalid_budget400/` and are **not evidence in
  either direction**. The synthetic study is unaffected by finding 5 (400 steps
  spans 16 epochs there). Re-runs are at 1200 steps.
