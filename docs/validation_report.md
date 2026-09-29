# VSS Validation Report

Validation pass, September 2026. Every number below is measured and backed by a
committed artifact. Single-seed results are labeled as such. The companion
epistemic ledger is `docs/claims.md`; the pre-pass audit is
`docs/current_state_audit.md`.

---

## 1. What changed

Code (in git order):
- `fix-training-resume` — mid-epoch step checkpoints (every 100 steps) with
  partial-epoch replay; made CPU training chunkable across session limits.
- `fix-choice-dilution` — `header_only_choice` config + serialization + slot-CE
  regime (the single biggest fix; see §5).
- `add-clinc150` — CLINC150 converters (151 intents), real-data eval tooling,
  TF-IDF+LR baseline.
- `add-banking77` — Banking77 converters, trained run, TF-IDF baseline.
  Also fixed `eval_realdata.py`, which called a nonexistent `decide_batch`.
- `test-option-permutation` — invariance contract tests for option order.
- `probe-question-interference` — solo vs co-asked answer stability probe.
- `benchmark-latency` — serving-mode latency profiles (methods A/B/C).
- `audit-calibration` — calibration-head audit, temperature audit, adaptive ECE.
- `multiseed-banking77` — 3-seed stability estimate.
- `robustness-suite` — deterministic text perturbations on real data.
- `compare-tokenizers` — drop-in BPE tokenizer + identical-recipe comparison.

Protocol changes: all evaluation moved to honest splits (leakage audited,
§3); thresholds and temperatures are selected on validation only; OOD and
calibration report separately from accuracy.

## 2. Existing architecture

As validated here (see `docs/current_state_audit.md` for the pre-pass state):
state + typed questions are serialized into one token sequence
(`<STATE>` block + one `<QUESTION>` block per question), encoded by a
bidirectional 6-layer transformer (hidden 256), and mean-pooled per question
span into question vectors. Heads: Choice (slot projection over a shared
option-slot table, + trained abstain logit), Noul (binary), Score (ordinal
bins). Confidence = blend of top option probability and the calibration head.
Inference is a single padded forward per request, any number of questions.
Parameter count (slot-ho config): 11,164,483 (94.0% encoder).

## 3. Data leakage audit

`benchmarks/leakage/audit_leakage.py` on the original synthetic benchmark:
- 75/480 test examples were **exact duplicates** of train examples.
- 92.9% had token-Jaccard ≥ 0.7 to some train example.
- The temperature used in earlier reports was fitted **on the eval split**.
Consequence: prior synthetic results (1.000 accuracy, ECE 0.004) were
invalidated. Held-out-template synthetic accuracy: **0.3125, ECE 0.65**
(`benchmarks/leakage/synthetic_report.json`). All subsequent real-data
evaluation uses official splits with no overlap and validation-only fitting.

## 4. Synthetic benchmark results

After the leakage fix, synthetic data is no longer treated as a headline
benchmark — it is a unit-scale sanity check. The held-out-template collapse
(§3) shows the original generator rewarded memorization. No architecture
conclusion is drawn from synthetic results in this pass.

## 5. CLINC150 results

Regimes (identical backbone, seed 13, full official 151-option schema):

| regime | in-scope accuracy | notes |
|---|---|---|
| refine-CE (original) | 6.2% | dilution bug active |
| slot-CE | 16.8% | subset-15 eval only 81.7% |
| **slot-CE + header-only** | **73.98%** | val CE 2.5682 |

Headline run `runs/clinc150-slot-ho` (single seed):
- in-scope full-151 accuracy **73.98%**, OOS AUROC **0.800**, ECE answered
  **0.040**, validation-selected threshold 0.3306 → coverage 0.9556,
  OOS abstention recall 0.28.
- risk–coverage monotone: 99.6% @ 50% cov → 76.8% @ 95% cov.
- TF-IDF+LR baseline: **88.8%** accuracy, AUROC 0.889, ECE 0.297.
  **VSS trails by 14.8 accuracy points and 0.089 AUROC; leads on ECE by 0.26.**

## 6. Banking77 results

Trained `runs/banking77-slot-ho` (same recipe, 77 intents, seed 13, ~33 min CPU):
- full test split (n=3080): **73.21%** accuracy, macro-F1 0.742,
  ECE 0.118, abstain rate 0.161.
- matched 1000-example subsample (seed 42): **72.4%** vs TF-IDF+LR **87.0%**
  (−14.1 pts, ≈20 seed-sigmas, §10). ECE **0.123** vs baseline **0.248**.
- OOD abstention: N/A — Banking77 defines no `oos` split (not faked).

## 7. Robustness results

CLINC150 test subsample (n=1000, seed 42), deterministic perturbations
(`benchmarks/robustness/clinc150_slot_ho.json`):

| perturbation | accuracy | Δ vs clean |
|---|---|---|
| clean | 63.4% | — |
| lower / upper / whitespace | 63.4% | 0.0 (exact invariance) |
| one adjacent-char typo | 54.6% | **−8.8** |
| one dropped word | 53.0% | **−10.4** |
| appended "?!" | 59.9% | −3.5 |
| duplicated word | 62.7% | −0.7 |

Case/whitespace invariance is a tokenizer property; character- and
word-level edits expose real fragility. See §9 for the BPE test of the
obvious remedy.

## 8. OOD-abstention results

CLINC150 (protocol in `scripts/eval_ood.py`; artifact
`benchmarks/ood/clinc150_slot_ho_best.json`): confidence separates OOS from
in-scope at AUROC 0.800 (baseline 0.889). At the validation-selected
threshold, 95.6% of in-scope questions are answered with selective accuracy
76.8%, and 28% of OOS questions are abstained. The abstain logit is trained
(joint options+abstain softmax), and its mass is what drives explicit
abstention; ranking by confidence adds the rest. Single seed.

## 9. Calibration results

`benchmarks/calibration/clinc150_slot_ho.json` (validation audit + test ECE):
- **Calibration head is saturated**: mean 0.9916, std 0.0134, 99.7% of
  validation examples above 0.9. It does not track per-question correctness;
  its AUROC for correctness is 0.741 vs **0.842** for plain top-prob.
  The blend confidence still works (ECE 0.0992 vs 0.1045 for top-prob) but
  the head contributes nothing discriminative.
- **Temperature audit**: T=1.2745 fitted on validation *raises* same-split
  ECE 0.049 → 0.089. The model is already near temperature-1; post-hoc
  temperature scaling hurts. The original pipeline's temperature step should
  be removed or refitted on a separate calibration split.
- Adaptive ECE matches equal-width ECE closely (0.0992 both for blend);
  reliability curve committed in the artifact.
- Multi-seed (Banking77, 3 seeds): ECE 0.120–0.123 — stable (§10).

## 10. Question-scaling results

Latency (`benchmarks/latency/clinc150.json`, warm p50 over 5 reps):

| N questions | A: co-asked (1 call) | B: solo seq. | C: solo batched | A acc @ q0 |
|---|---|---|---|---|
| 1 | 0.099 s | 0.199 s | 0.069 s | 68.8%* |
| 10 | 0.427 s | 2.019 s | 0.718 s | 56.2%* |
| 50 | 1.956 s | 10.061 s | 3.570 s | 25.0%* |
| 200 | 8.997 s | 40.2 s (extrap.) | 14.467 s | 6.2%* |

\* n=16 probes → noisy; authoritative interference numbers use 300 probes
(`benchmarks/interference/`). Cold start 2.36 s.

Interference (`benchmarks/interference/clinc150_dose_response.json`, n=300):
co-asking degrades the gold-scored own-question accuracy monotonically —
−3.3 / −4.0 / −5.7 / −8.7 / −20.0 pts at k=2 / 4 / 8 / 16 / 32 — while speedup
per answer saturates at ~2.2×. Batching invariance verified (<1e-6), so the
cause is the **bidirectional** encoder letting questions attend to each other
(`is_causal=False`); no slot is structurally immune.

## 11. Baseline comparisons

| dataset | VSS acc | TF-IDF+LR acc | Δ | VSS ECE | baseline ECE |
|---|---|---|---|---|---|
| CLINC150 (in-scope, single seed) | 73.98% | 88.8% | **−14.8** | 0.040 | 0.297 |
| Banking77 (3-seed mean) | 72.9% | 87.0% | **−14.1** | 0.121 | 0.248 |

**Same-encoder ablation (U1)** — identical backbone, optimizer, schedule,
seed; only input format and head differ (`benchmarks/ablation/`):

| model | in-scope acc (1000-sub) |
|---|---|
| VSS full model (slot-CE + questions) | 73.98% (full 4500) |
| plain linear head, state-only, scratch | **86.7%** |
| plain linear head, state-only, warm-start | **88.0%** |

The serialization + slot-head layer **costs ~13 accuracy points** on the same
backbone. The plain scratch head also matches the TF-IDF+LR baseline. VSS
loses on accuracy and AUROC on both datasets and to its own backbone without
the question machinery; it wins on calibration and (at k≥2 co-asking)
latency. The honest reading: VSS's typed-question layer is a net cost for
single-intent classification accuracy; its remaining value proposition is
calibration + abstention + serving shape, two of which are themselves
qualified by §9 and §10.

## 12. Statistical confidence

- Banking77 accuracy: 3 seeds, 72.4 / 72.5 / 73.7% (sample std 0.72 pts).
  All reported gaps exceed 19 seed-sigmas.
- CLINC150: **single seed** (13) for all runs; variance unmeasured (the
  Banking77 estimate suggests ~0.7 pts, but 151 classes may differ).
- Interference dose-response: n=300 probes per k; slot0 flips 21 vs 4 at k=8
  (sign-test p ≈ 0.001).
- Latency: p50/p95/p99 over 5 warm reps; single machine, 6 torch threads;
  method B extrapolated beyond N=50 (flagged in artifact).
- No confidence intervals on ECE (single eval split, no bootstrap yet).

## 13. Failures

1. **Original synthetic benchmark was self-deceiving** (§3) — fixed by audit
   + held-out generators, at the cost of the earlier 100% claims.
2. **Option-dilution bug shipped unnoticed** — 6.2% accuracy looked like "model
   can't learn"; it was serialization. Found by a 3-line ablation.
3. **`eval_realdata.py` never ran** — called a nonexistent API; fixed in
   `add-banking77`.
4. **Calibration head does not work as designed** (§9) — saturated; blend
   confidence works but not for the documented reason.
5. **Temperature scaling hurts** (§9) — the fitting step is a net negative.
6. **BPE hypothesis refuted** (§12→`compare-tokenizers`): −11.2 pts clean.
7. **Single-pass multi-question accuracy is not preserved** (§10) — the
   architecture's marquee property degrades monotonically with k.
8. **VSS trails a fair classical baseline on both real datasets** (§11).
9. **Ablation harness bug (caught)**: the first encoder-classifier scratch
   run silently trained only the linear head — `VSSEncoder` is a deliberate
   non-Module, so encoder params never joined the optimizer. Detected by
   checkpoint forensics (state dict had 2 tensors), fixed with an nn.Module
   encoder box, and retrained. An intermediate "79.7%" warm-start number was
   a frozen-encoder artifact and was discarded, not reported as a result.

## 14. Architecture conclusions

1. **The typed-question layer is a net accuracy cost for single-intent
   classification.** The same encoder with a plain linear head beats the full
   VSS model by ~13 points from scratch (§11). If VSS is to be justified, it
   must be on calibration, abstention, or serving grounds — and each of
   those is currently qualified (saturated head §9; interference §10).
2. **Mean-pooled question spans are the system's load-bearing weakness.**
   The major failures (option-text dilution; BPE regression; the ablation
   gap) all point at information loss between serialization and the pooled
   per-question vector. Header-only serialization works *because* it
   shortens and purifies the pooled region — and the best head is one that
   needs no question block at all.
3. **Bidirectional attention across questions is a design bug for the
   multi-question claim.** With per-question spans pooled into independent
   vectors, cross-question attention only adds noise (§10). Either the
   encoder needs question-masked attention or the serving story must change.
4. **The slot-projection choice head trained stably** where refine-CE
   collapsed, and permutation-invariance is a verified contract — but the
   head is outclassed by a linear head over a state-only pool (§11).
5. **The calibration head should be redesigned or dropped** — as built it is
   a constant. Blend confidence ≈ top-prob in practice.
6. **Word+hash tokenization is adequate and robust-in-variance** at this
   scale; the OOV hash is doing real work that BPE could not replace.

## 15. What remains unproven

- ~~That VSS's architecture (vs the same encoder + a plain classifier head)
  contributes anything on real data~~ — now measured: it costs ~13 points
  (§11). What remains unproven is any *offsetting* real-data benefit
  (calibration/abstention gains attributable to the architecture rather than
  to training details).
- That multi-question single-pass serving can be made accuracy-preserving
  (U2).
- CLINC150 multi-seed stability (U3) — including for the ablation result.
- Any real-data validation of the Noul and Score heads (U4 — synthetic-only,
  and the synthetic benchmark itself was invalidated).
- That slot-table size 1024 is adequate (U5; 9 label collisions at 151 labels).

## 16. Recommended next experiment

**Question-masked attention, evaluated against the ablation bar.**

The U1 ablation (§11) removed the original motivation for co-asking: a plain
head on state-only text is both more accurate and cheaper. The remaining
scientific case for VSS's question layer must therefore show a *benefit the
plain head cannot match*. The cleanest such test:

1. Add question-masked attention (block attention between question spans,
   keep state ↔ question attention) and re-run the committed interference
   dose-response (`benchmarks/interference/`). Acceptance: co-asked accuracy
   ≈ solo accuracy at every k (the −3.3…−20.0 pts decay disappears).
2. Then require the fixed multi-question model to beat the §11 plain-head
   baseline on a task the plain head cannot do at all: N questions over a
   shared state with per-question gold (e.g., intent + slot-filling + score
   jointly). If it cannot, the honest conclusion is that VSS's serialization
   should be reduced to a plain classifier with auxiliary heads, and the
   multi-question machinery retired.
