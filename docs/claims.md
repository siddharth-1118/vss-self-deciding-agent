# VSS Claims Ledger

Epistemic tiers: **Demonstrated** (measured here, reproducible from committed
artifacts), **Plausible** (consistent with evidence but not isolated), **Unknown**
(no evidence either way). Every claim names its evidence. All real-data results
except where noted are SINGLE training seed unless stated.

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

### U2. RESOLVED — VSS does NOT beat the plain-head baseline on multi-question tasks
Question-masked attention is meant to fix interference exactly (D3 resolution)
at −2.3 pts solo accuracy. **The multi-question value test answered this: the
plain classifier wins** — and it still wins after the per-batch mask bug was
found, fixed, and every VSS checkpoint retrained (D13). See the
"Multi-question value test" section below (D13–D20) and
`docs/multi_question_value_report.md`. The single-pass co-asking machinery did
not beat a plain classifier on per-question accuracy at any Q, on any of the
three datasets; the architecture is not justified as it stands **at this
budget**, which is the confound that must be closed next.

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

### D13. Plain classifier beats VSS on per-question accuracy — DEMONSTRATED
At matched training budget (same tokenizer family, same header-only
serialization, same full-inventory choice head, 8 epochs AdamW, batch 32,
best-val checkpoint selection, ~10.5M vs 11.2M parameters), the plain
classifier (systems A/B) beats VSS (system C) on per-question accuracy in
**all 35 measured (dataset, Q) cells** — synthetic Q∈{1,2,4,8,16,32,50} over
3 seeds, CLINC150 and Banking77 at Q∈{1,…,50}. Paired bootstrap (1000
resamples) of C−B is negative everywhere with CIs excluding zero (synthetic
Q=1 [−0.265,−0.157] → Q=50 [−0.170,−0.151]; CLINC150 Q=50 [−0.258,−0.238];
Banking77 Q=50 [−0.107,−0.088]). VSS request accuracy reaches 0.000 from Q=16
on synthetic (all-questions-right requirement); plain decays gracefully.
**Re-measured after the mask fix with retrained VSS checkpoints, so this is not
a bug artifact** — it is a statement about this budget, not about the
architecture (see U2 and the report's §20).

### D14. VSS single-pass latency beats even the BATCHED classifier at high Q — DEMONSTRATED (conditional)
Synthetic p50 request latency at Q=50: A sequential 745.2 ms, B batched
267.9 ms, C VSS 109.2 ms → **6.8× vs sequential, 2.5× vs batched**
(throughput 460 vs 187 questions/s). At Q=1 all three are within 2% of each
other. The advantage is a function of request length, not question count: on
CLINC150 and Banking77 (one short question per state) VSS is 1.1× *slower*
per request and 0.82–0.92× the batched classifier's questions/s. Absolute
latencies are load-dependent (the post-fix re-run measured 2–3× slower absolute
values than the pre-fix run); only within-cell ratios are claimed.

### D15. VSS has no measurable cross-question interference on any dataset — DEMONSTRATED (rewritten post-fix)
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
number.

### D16. VSS's confidence gate selects a more accurate subset — DEMONSTRATED (rewritten post-fix)
With the shipped threshold (0.55, blend confidence) on CLINC150, VSS's answered
questions are **77.7% accurate at 53.0% coverage** where the plain classifier is
65.9% accurate while always answering (Q=50) — an 11.8-pt selective-accuracy
advantage. The advantage does not transfer to Banking77 at Q=1 (0.857 answered
vs 0.887 plain) but appears at Q=50 (0.867 vs 0.851).
**The gate costs nothing in overall accuracy**: sweeping the threshold
(`results/abstention_sweep.json`), decision accuracy is flat at 0.533
(CLINC150) and 0.687 (Banking77) across thresholds 0.0–0.55. The first pass's
"abstention costs 15–17 accuracy points" claim is **retracted** — it was a
symptom of the mask bug, not a property of the gate. The first pass's 17.5-pt
selective-accuracy figure is also superseded (it was 85.5% answered @ 66.3%
coverage on a buggy-mask checkpoint).

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

### D20. VSS answers are 96.4–99.3% order-invariant on distinct synthetic questions — DEMONSTRATED (post-fix replacement for D17)
After the D19 fix and retraining, permuting distinct synthetic questions within
one request changes the answer for 0.7–3.6% of (state, question) pairs
(Q=8: 0.964/0.967/0.966; Q=32: 0.991/0.993/0.991 across seeds; CLINC150
0.996/0.998, Banking77 0.998/0.999 on replicated requests). The plain
classifier is exactly order-invariant by construction. The residual
disagreement carries no consistent direction (see D15), which is consistent
with fp32 non-associativity rather than a slot-position effect.

---

## Explicit non-claims

- VSS does **not** beat a TF-IDF+LR baseline on either real dataset
  (−14.8 pts CLINC150, −14.1 pts Banking77). It wins on calibration and latency.
- The calibration head does **not** provide per-question correctness probability
  (D5). The blend confidence works, but through top-prob, not head signal.
- "One forward pass answers N questions" is true at N=1 and false as an
  accuracy-preserving claim at large N (D3, D10).
- **VSS is NOT validated as an architecture.** The multi-question value test
  found the plain classifier ahead on accuracy in 35/35 cells (D13) — measured
  on defect-free code with retrained checkpoints. The specialized architecture
  is not justified over a plain classifier called once per question at this
  budget; only the single-pass latency mechanism (D14), the verified absence of
  interference (D15) and the selective-accuracy gate (D16) survive as genuine,
  narrower benefits.
- **VSS is NOT ready to scale to 52M+ parameters, but not for the reason first
  stated.** The order-sensitivity (D17) and real-data interference (D15)
  objections were implementation bugs and are withdrawn. What remains is the
  accuracy gap (D13) *confounded by an unconverged VSS training budget* — its
  synthetic validation loss was still descending at the 8-epoch cap while plain
  had plateaued. The gap is therefore a statement about this budget, and the
  budget must be closed before any scaling decision.
