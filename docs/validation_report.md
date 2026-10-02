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
3. **Interference is a mask bug, not an architecture fate.** Question-masked
   attention + stable RoPE positions eliminate the co-asking decay exactly
   (dose-response flat ±0.7 pts at k≤32) at a cost of −2.3 pts solo accuracy
   and an ECE improvement (0.040→0.028). The remaining question is whether
   single-pass co-asking can justify that trade against a plain classifier
   with auxiliary heads.
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

**Question-masked attention: DONE — passed its acceptance test.**

Implemented (`question_masked: true`; block-isolation mask + stable per-token
RoPE positions) and re-trained on the identical CLINC150 recipe. The
committed dose-response flipped from −3.3/−4.0/−5.7/−8.7/−20.0 pts (k=2/4/8/
16/32) to **+0.7/+0.3/0.0/0.0/+0.3** — co-asking is now accuracy-preserving,
with ~1.95× throughput at k=32. Costs: −2.3 pts solo accuracy (71.7% vs
74.0%), ECE answered improved 0.040→0.028, OOS AUROC unchanged (0.803).
Evidence: `benchmarks/interference/clinc150_qmask_dose_response.json`,
`benchmarks/ood/clinc150_qmask_best.json`.

**Next: the multi-question value test.** With interference eliminated, the
last open question is whether single-pass co-asking beats a plain classifier
with auxiliary heads on a task where per-question gold over a shared state is
the task (e.g., intent + slot-fill + score asked jointly, scored per
question). Compare: (a) plain head + N forward passes, (b) qmask VSS, one
pass. If (a) wins on accuracy×latency, the multi-question machinery should
be retired in favor of a plain classifier with auxiliary heads; if (b) wins,
VSS has its justified niche.

---

# 17. The multi-question value test (RESOLVED — plain classifier wins)

Research trail so far: unmasked VSS (interference −3.3…−20.0 pts at
k=2…32) → qmask VSS (dose-response flat: +0.7/+0.3/0.0/0.0/+0.3) → **this
value test**. Nothing above is deleted or rewritten; this section only adds
what the value test measured.

## 17.1 What was run

Three inference modes over identical per-question gold: **A** plain classifier
called once per question (sequential forwards), **B** the same network batched
over the request's questions, **C** qmask VSS answering the whole request in
one pass. Q ∈ {1,2,4,8,16,32,50}; synthetic (3 seeds, 200 eval states,
mixed choice/noul/score), CLINC150 and Banking77 (seed 13, 150 eval states,
real 151-/77-label intent tasks). 35 cells; every question scored
individually; ECE/Brier/NLL per question; p50/p95/p99 latency; paired
bootstrap C−B; solo-vs-joint interference per template; order permutation.
Artifacts: `benchmarks/multi_question_value/` (harness, results JSON,
`report.md`, `results/DIGEST.txt`, 15 SVG plots),
`docs/multi_question_value_report.md` (21 sections), `docs/claims.md` D13–D18.
Test suite: 63 passing at this pass, 65 after the §17.5 fixes.

> §17.1–§17.4 record the **first** pass. §17.5 is the current result: two of
> the findings below (order-invariance, real-data interference) were bugs and
> are withdrawn.

## 17.2 Result

**Accuracy — plain wins everywhere.** Plain beats VSS in 35/35 cells. Paired
bootstrap C−B CI excludes zero in every cell (synthetic Q=1 [−0.380,−0.252]
… Q=50 [−0.200,−0.180]; CLINC150 Q=50 [−0.272,−0.252]; Banking77 Q=50
[−0.435,−0.410]). Synthetic Q=1: 0.970 vs 0.655. CLINC150: 0.680 vs 0.447.
Banking77: 0.887 vs 0.573. VSS request accuracy collapses to 0.000 from Q=8 on
synthetic (all-questions-right).

**Calibration — mixed, mostly plain.** Plain better at Q=1–2 (synthetic ECE
0.049 vs 0.237); VSS better on synthetic ECE at Q≥4 (0.11–0.15 vs 0.15–0.22);
Brier/NLL favor plain in nearly every cell. The earlier "ECE answered 0.028"
does not reproduce under this test protocol and is not comparable.

**Latency — VSS wins only where requests are long.** Synthetic Q=50 p50:
A 778.6 ms, B 304.4 ms, C 110.8 ms (7.0× vs sequential, **2.8× vs the
batched** classifier). On real data VSS is 1.1–1.3× *slower* per request
(CLINC150 Q=50: A 17.1 / B 17.2 / C 21.0 ms) and 0.82× the batched
classifier's questions/s.

**Interference — zero on synthetic, real on real data.** Per-template
solo-vs-joint deltas: synthetic −0.13…+2.7 pts across 3 seeds × Q∈{8,32,50};
CLINC150 −16.9/−15.6 pts, Banking77 −9.1/−10.5 pts at Q=8/32 (of which, on
CLINC150 Q=8, ≈−14.8 pts is lost coverage 66.3%→51.5% and −8.3 pts is
answered-accuracy loss 85.5%→77.2%).

**Order — VSS is not order-invariant.** Distinct synthetic questions
reordered: 74–79% agreement (Q=8 and Q=32, all seeds); plain is exactly
order-invariant by construction. New defect, not visible in the dose-response
work.

**What VSS does own.** A confidence gate that selects an 85.5%-accurate subset
at 66.3% coverage on CLINC150 where the always-on plain classifier is 68.0%
(+17.5 pts selective advantage), and single-pass latency that beats even the
*batched* classifier by 2.8× at Q=50 on long-text requests.

## 17.3 Honesty notes on this experiment

Two harness defects were found and fixed rather than reported as results:
(1) the first plain real-data baselines used per-row masked CE over 15 declared
options against 151/77 at test time (5.3% CLINC "plain"); those runs are
quarantined as `runs/mqv-plain-*_maskedce_v1` and all reported numbers use the
retrained full-inventory head (CLINC val loss 4.66 → 0.395);
(2) real-data question replication initially renamed slot 0's id, pushing both
systems onto unseen text and faking a Q=1→Q=2 collapse; slot 0 now keeps the
canonical id. Also disclosed: the only pre-existing banking77 checkpoint is
*unmasked* (not System C), so a qmask banking77 checkpoint was trained for
this test (best eval 1.3206 @ epoch 3); VSS synthetic training had not
converged at the 8-epoch cap, so its accuracy deficit is confounded by budget.

## 17.4 Decision

Per the decision tree, this resolves to **"plain wins" → redesign**, not
scale. VSS's specialized architecture does not currently justify its
complexity over a plain classifier called once per question. Preserved for a
redesign: the single-pass multi-question mechanism (2.8× vs batched at Q=50),
the typed per-question heads, and the confidence-gated abstention (+17.5 pts
selective accuracy on CLINC150). To be fixed first: order-invariance and the
abstention calibration that costs 15–17 pts of real-data accuracy. VSS is
**not** ready to scale to 52M+ parameters.

> **SUPERSEDED IN PART — read §17.5 first.** The two defects named for fixing
> above (order-invariance, and the "15–17 pt abstention cost") turned out to be
> bugs, not architecture. They are fixed and the whole test was re-measured;
> §17.5 is the current result. The accuracy verdict (plain wins 35/35 cells)
> survived the fix.

## 17.5 Second pass — the two "defects" were bugs, and the verdict survived

Acting on §17.4's "fix the defects first" item found **two implementation
bugs**, not architectural properties. Both are fixed, regression-tested, and the
entire 35-cell value test was re-measured on corrected code with retrained VSS
checkpoints. Pre-fix results are preserved verbatim in
`benchmarks/multi_question_value/results/*_results_prefix.json`.

**Bug 1 — the question mask was built per batch, not per example**
(`src/vss/model/encoder.py`, fixed in `df5eb50`). The encoder allocated a single
`[T,T]` plane and wrote every example's rows into it inside the batch loop, so
example *b*'s span geometry overwrote example *b−1*'s — the last example in the
batch decided the isolation mask for all of them. Isolation depended on batch
composition and question order, and was invisible at batch size 1, which is why
every earlier single-question probe missed it. Fix: per-example
`m = torch.zeros(len(spans), T, T)`, writes `m[b, …]`, `qmask = m.unsqueeze(1)`
→ `[B,1,T,T]`. Two regression tests pin it in `tests/test_question_mask.py`
(per-example planes in a heterogeneous batch; batch-invariance of per-question
outputs, atol 2e-3 for fp32 padding noise). Test suite 63 → 65.

**Bug 2 — the interference probe compared different state sets.** The solo arm
was capped at 100 states while the joint arm scored 200, so it reported large
deltas with **zero** paired decisions actually flipping. Both arms now score the
same `(state, question)` pairs, the joint arm is computed once over the union of
selected states, and every per-question entry carries
`paired_decision_flips`.

**Measured effect (before → after):**

| Measurement | Pre-fix | Post-fix |
|---|---:|---:|
| Order agreement, synthetic Q=8 (3 seeds) | 0.776 / 0.794 / 0.788 | 0.964 / 0.967 / 0.966 |
| Order agreement, synthetic Q=32 (3 seeds) | 0.740 / 0.752 / 0.748 | 0.991 / 0.993 / 0.991 |
| Order agreement, CLINC150 Q=8 / Q=32 (same ckpt, code only) | 0.973 / 0.975 | 0.996 / 0.998 |
| Interference, CLINC150 Q=8 / Q=32 (same ckpt, code only) | −16.9 / −15.6 pts | **−0.13 / −0.31 pts** |
| Interference, Banking77 Q=8 / Q=32 | −9.1 / −10.5 pts | **0.00 / +0.06 pts** |
| Paired decision flips, synthetic (27,000 pairs) | not measured | **0** |
| VSS accuracy, Banking77 Q=1 / Q=50 (retrained) | 0.573 / 0.429 | **0.760 / 0.754** |
| VSS answered-accuracy, CLINC150 Q=1 | 0.761 @ 58.7% | 0.736 @ 60.7% (plain 0.680) |

**One of my own claims was simply wrong and is retracted.** §17.2/§17.4 blamed
abstention for "15–17 points" of real-data accuracy. Sweeping the threshold
(`benchmarks/multi_question_value/results/abstention_sweep.json`) shows
*decision* accuracy (abstentions scored wrong) is **flat at 0.533 on CLINC150**
across thresholds 0.0–0.55 (coverage 100% → 68%) and 0.687 on Banking77 across
0.0–0.5. The gate costs nothing; the loss was a symptom of the mask bug.

**What did not change: the accuracy verdict.** Plain still beats VSS in 35/35
cells with paired-bootstrap CIs excluding zero, now on defect-free code with
retrained checkpoints (synthetic Q=1 [−0.265,−0.157] … Q=50 [−0.170,−0.151];
CLINC150 Q=50 [−0.258,−0.238]; Banking77 Q=50 [−0.107,−0.088]). Retraining on
the fixed mask also *helped* VSS substantially on Banking77 (0.573 → 0.760 at
Q=1) and still left it behind plain, which is the strongest form of the
finding: the gap is not a bug artifact.

**Revised decision.** Still **do not scale**, but the remaining reason is
narrower: the accuracy gap is confounded by VSS's unconverged training budget
(synthetic validation loss still descending at the 8-epoch cap), so "plain wins"
is a statement about *this budget*, not the architecture. The next experiment is
therefore §16/§21 item 2 — a converged, multi-seed budget — plus a genuinely
multi-question real task, not another bug hunt.
