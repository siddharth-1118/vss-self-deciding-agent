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

### D3. Co-asking questions degrades accuracy monotonically
Gold-scored slot0 comparison (own question, own state): solo 61.7% vs
co-asked 56.0% at k=8 (21 harmful flips vs 4 helpful). Dose-response
−3.3/−4.0/−5.7/−8.7/−20.0 pts at k=2/4/8/16/32. Cause: bidirectional encoder
(`is_causal=False`) — every question attends to all others.
Evidence: `benchmarks/interference/clinc150_dose_response.json`.

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

### U1. Whether the question-masked-attention fix preserves accuracy
Candidate fix for D3 (block attention between question spans, keep
state ↔ question attention) is untested. Note the D10b ablation removes the
motivation for multi-question co-asking on accuracy grounds; U2 is now mainly
about whether VSS can keep its single-pass serving story at ANY accuracy level.

### U2. Whether interference (D3) can be fixed without losing single-pass batching
Candidate fixes (question-masked attention, per-question pooling at every layer,
causal question blocks) are untested.

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

## Explicit non-claims

- VSS does **not** beat a TF-IDF+LR baseline on either real dataset
  (−14.8 pts CLINC150, −14.1 pts Banking77). It wins on calibration and latency.
- The calibration head does **not** provide per-question correctness probability
  (D5). The blend confidence works, but through top-prob, not head signal.
- "One forward pass answers N questions" is true at N=1 and false as an
  accuracy-preserving claim at large N (D3, D10).
