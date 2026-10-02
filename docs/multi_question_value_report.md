# Multi-Question Value Test — Does VSS Justify Its Architecture?

**Status:** complete. 35 (seed, Q) cells across 3 datasets, 4 analyses per dataset,
63/63 tests passing. All numbers below are measured, not estimated.

**Bottom line up front:** at this budget (11M parameters, CPU, 800–9500 training
states) the plain classifier wins on per-question accuracy on **all three datasets**
with bootstrap CIs that exclude zero, and VSS loses to it on calibration in most
cells. VSS's real advantages are narrower and still real: **zero cross-question
interference on the synthetic task**, **single-pass latency that beats even the
*batched* classifier once Q is large and question text is long** (2.8× at Q=50),
and **selective accuracy** — when VSS answers, it is far more often right than the
plain classifier is (85.5% vs 68.0% answered-accuracy on CLINC150). VSS is not
ready to scale; the architecture is not yet justified over the simpler baseline.

---

## 1. Experimental Setup

| Item | Value |
|---|---|
| Hardware | AMD64 Family 25 Model 68 (Zen-class), 12 logical / 6 physical cores, 16.4 GB RAM |
| GPU | none (CPU-only) |
| PyTorch | 2.14.0+cpu, fp32, `torch.set_num_threads(6)` (measured optimum for this box) |
| Python | 3.11.9, Windows 10.0.26200 |
| Latency protocol | cold start measured separately; 5 warmup + 20 measured iterations per cell (largest practical on CPU; a mode-A iteration at Q=50 is 50 sequential forwards) |
| Seeds | synthetic: 1, 2, 3 (both systems trained per seed). Real data: seed 13 (one plain + one VSS checkpoint each) |
| Question counts | Q ∈ {1, 2, 4, 8, 16, 32, 50} |
| Scoring | **per question**, against independently verified gold; request accuracy also reported |
| Raw results | `benchmarks/multi_question_value/results/{synthetic,clinc150,banking77}_results.json` |
| Generated report | `benchmarks/multi_question_value/report.md` |
| Plots | `benchmarks/multi_question_value/plots/*.svg` (hand-rolled SVG, no matplotlib) |

Cold start (model load + first request): plain 2.1–2.4 s, VSS 3.4–4.1 s.

## 2. Systems

**System A/B — plain classifier.** `STATE + QUESTION → one typed decision`.
Same tokenizer family and canonical serialization as VSS (`src/vss/model/`),
header-only choice blocks (exactly what VSS sees), 6 layers, hidden 256,
vocab 16384, mean-pooled encoder, linear choice head over the full option
inventory + trained ABSTAIN class, Noul sigmoid head, 64-bin score head.
10.53M trainable parameters.

**System C — VSS, unmodified.** Existing `qmask` checkpoints
(`question_masked: true`, stable per-token RoPE), shipped inference engine
(`abstain_threshold: 0.55`, `confidence_mode: blend`). No VSS file was changed
to win this benchmark. 11.16M parameters. Three checkpoints:

| Dataset | VSS run | Note |
|---|---|---|
| synthetic | `runs/mqv-vss-synthetic-s{1,2,3}` | trained here, identical recipe |
| clinc150 | `runs/clinc150-slot-ho-qmask` | pre-existing validated qmask run |
| banking77 | `runs/mqv-vss-banking77-s13` | **trained here** — the only pre-existing banking77 checkpoint was *unmasked*, which is not System C; see §16 |

**Fairness guards.** Mode A and mode B are the *same network*. Batching is
verified not to change outputs on every cell (`A_equals_B: true` in all 35
cells; spot check = 16 pairs at batch 1 vs batch 32, max probability
difference ≤ 1e-6, i.e. fp32 noise). A is reported as measured-equivalent to B
rather than recomputed, which is why `A_equals_B` is a checked claim rather
than an assumption.

## 3. Datasets

| Dataset | Split sizes | Eval | Questions |
|---|---|---|---|
| synthetic | 800 train / 300 val / 1000 test states | 200 test states, 3 seeds | 64-question pool per state (16 base templates + rephrasings `_b/_c/_d` sharing the same answer function); mixed `choice` / `noul` / `score` |
| clinc150 | repo splits under `data/clinc150/` | 150 test states | 1 real choice question per state over the real 151-label inventory |
| banking77 | repo splits under `data/banking77/` | 150 test states | 1 real choice question per state over the real 77-label inventory |

Synthetic gold is a deterministic function of the state's facts (never stored
prose, never model-generated) and is re-verified at load time. Real gold is the
datasets' own labels; no real result is fabricated.

**Eval > Q=16 includes rephrased surface forms** (`_b/_c/_d` variants of the
same underlying fact). This is a deliberate surface-form generalization probe
and is disclosed wherever synthetic Q≥32 numbers appear.

## 4. Question Construction

Nested prefixes by construction: each state owns an ordered question pool;
the Q=q evaluation is `pool[:q]`, so Q=2 ⊂ Q=4 ⊂ … ⊂ Q=50, with the *same*
states at every Q and identical sets for modes A, B and C. State subsampling
happens once, independent of Q, seeded from the run seed.

Real datasets have exactly one natural question per state, so Q-scaling
replicates it with distinct ids. **Slot 0 keeps the canonical trained id**
(`intent`); extra copies get suffixed ids (`intent#7`) only because the
inference engine keys answers by id within a request. An earlier version of
the harness renamed slot 0 as well; that pushed both systems onto unseen
question text and produced artificial Q=1→Q=2 collapses. The fix is disclosed
here because it changed real-data numbers materially (§16).

## 5. Training Fairness

Identical splits, identical optimizer family, identical budget, identical
checkpoint-selection rule (best validation loss, never final epoch).

| | plain | VSS |
|---|---|---|
| Parameters | 10.53M | 11.16M |
| Epochs | 8 | 8 |
| Optimizer | AdamW, warmup + cosine | AdamW, warmup + cosine |
| Batch size | 32 | 32 |
| Seeds | synthetic 1/2/3, real 13 | synthetic 1/2/3, real 13 |
| Checkpoint | best val (CLINC 0.395, Banking77 0.5206) | best val (synthetic 2.21/2.48/2.26, Banking77 1.3206 @ epoch 3) |
| Per-epoch validation slice | first 200 states (selection signal only) | first 100 states (selection signal only) |

Per-epoch histories and full hyper-parameters are in
`benchmarks/multi_question_value/train_logs/*.json`.

**One protocol defect was found and fixed mid-experiment.** The first plain
real-data baselines trained with per-row masked cross-entropy over each row's
*declared* 15 options while test rows declare all 151/77 — a protocol artifact
that produced a 5.3% "plain" baseline on CLINC. The baseline was quarantined
(`runs/mqv-plain-*_maskedce_v1`, kept for audit), retrained with header-only
text and full-inventory cross-entropy — the same recipe as the repository's own
U1 ablation — and every real-data number in this report comes from the fixed
baseline. The quarantined runs are never cited as results.

## 6. Accuracy Results

Per-question accuracy (mean ± std over seeds), synthetic (3 seeds):

| Q | A/B plain | C VSS | plain−VSS (paired bootstrap C−B, 95% CI) |
|--:|---:|---:|---|
| 1 | 0.970 ± 0.009 | 0.655 ± 0.149 | −0.315 [−0.380, −0.252] |
| 2 | 0.985 ± 0.004 | 0.828 ± 0.075 | −0.158 [−0.196, −0.123] |
| 4 | 0.963 ± 0.008 | 0.886 ± 0.034 | −0.078 [−0.099, −0.057] |
| 8 | 0.928 ± 0.006 | 0.690 ± 0.030 | −0.239 [−0.261, −0.218] |
| 16 | 0.934 ± 0.020 | 0.719 ± 0.019 | −0.215 [−0.230, −0.200] |
| 32 | 0.802 ± 0.017 | 0.611 ± 0.009 | −0.191 [−0.202, −0.179] |
| 50 | 0.760 ± 0.017 | 0.570 ± 0.010 | −0.190 [−0.200, −0.180] |

CLINC150 (seed 13): plain 0.680 → 0.659 across Q; VSS 0.447 → 0.397;
C−B = −0.233 [−0.293, −0.167] at Q=1, −0.262 [−0.272, −0.252] at Q=50.

Banking77 (seed 13): plain 0.887 → 0.851; VSS 0.573 → 0.429;
C−B = −0.313 [−0.393, −0.240] at Q=1, −0.422 [−0.435, −0.410] at Q=50.

Macro-F1 and request accuracy move the same way. Request accuracy is the
starkest number: VSS needs *every* question in the request right, so it reaches
0.000 from Q=8 on synthetic while plain decays gracefully (0.53 at Q=8).

**Selective accuracy is where VSS is genuinely strong.** Counting an abstention
as wrong makes VSS look weak, but VSS's confidence gate is informative:

| Dataset | plain (always answers) | VSS answered-accuracy @ coverage |
|---|---:|---|
| CLINC150 Q=1 | 0.680 | **0.855 @ 66.3% coverage** |
| CLINC150 Q=8 | 0.663 | 0.772 @ 51.5% coverage |
| Banking77 Q=1 | 0.887 | 0.789 @ 72.7% coverage |
| Banking77 Q=50 | 0.851 | 0.756 @ 56.7% coverage |

On CLINC150 VSS's answered subset is **17.5 points more accurate** than the
plain classifier's always-on output. On Banking77 it is not — plain is simply
stronger there. This asymmetry is the honest headline: VSS's gating buys
accuracy on the hard, 151-way task and nothing on the 77-way one.

## 7. Calibration Results

ECE / Brier / NLL per question, all 15 metric-Q combinations in
`report.md`. Summary:

- **Q=1–2 (short requests): plain is better calibrated.** Synthetic Q=1 ECE
  0.049 vs 0.237; CLINC150 0.204 vs 0.216; Banking77 0.107 vs 0.153.
- **Q≥4 on synthetic, VSS's ECE is lower** (0.106–0.146 vs plain's
  0.153–0.221) — VSS degrades more gracefully with Q, while plain becomes
  overconfident on the harder rephrased variants.
- **Brier and NLL favor plain in nearly every cell**, including synthetic
  Q≥4 (e.g. Q=50: plain Brier 0.351/NLL 1.253 vs VSS 0.415/1.197).
- The historical "ECE answered 0.028" figure is an answered-only number from
  the CLINC validation set under a different protocol; it does **not**
  reproduce on this test protocol and is not comparable to the numbers above.

Conclusion: VSS's *selective* behavior is informative, but its raw
distributions are not better calibrated than a plain classifier's.

## 8. Question Scaling: Latency

p50 request latency (ms), synthetic:

| Q | A sequential | B batched | C VSS | C/A | C/B |
|--:|--:|--:|--:|--:|--:|
| 1 | 14.9 | 14.5 | 15.3 | 1.02 | 1.06 |
| 2 | 26.9 | 20.0 | 16.2 | 0.61 | 0.81 |
| 4 | 59.7 | 33.2 | 20.1 | 0.34 | 0.61 |
| 8 | 111.3 | 49.3 | 26.4 | 0.24 | 0.53 |
| 16 | 221.8 | 89.7 | 38.4 | 0.17 | 0.43 |
| 32 | 488.0 | 204.5 | 76.2 | 0.16 | 0.37 |
| 50 | 778.6 | 304.4 | 110.8 | **0.14** | **0.36** |

On real data the picture inverts: single-question-per-state requests are
overhead-bound, and VSS is *slower* than both plain modes at every Q
(CLINC150 Q=50: A 17.1 / B 17.2 / C 21.0 ms; Banking77 Q=50: A 14.1 /
B 13.3 / C 15.6 ms). The single-pass advantage is real but conditional on
request length, not on question count alone.

Plots: `plots/{dataset}_latency_p50_vs_Q.svg`, `plots/{dataset}_latency_p95_vs_Q.svg`.

## 9. Question Scaling: Accuracy

Plain decays smoothly with Q (synthetic 0.970 → 0.760 over 1→50) because each
question is an independent (state, question) input. VSS decays faster
(0.655 → 0.570, with a non-monotone bump at Q=4) and its *request* accuracy
collapses to zero by Q=8. On CLINC150 VSS's per-question accuracy is nearly flat
(0.447 → 0.397) while plain is flat too (0.680 → 0.659); on Banking77 VSS
recovers from 0.347 (Q=2) to 0.429 (Q=50).

Plots: `plots/{dataset}_accuracy_vs_Q.svg`.

## 10. Throughput

Questions per second at p50, synthetic: Q=50 → A 65.6, B 166.3, **C 458.3**
(VSS is 2.8× the batched classifier, 7.0× the sequential one). CLINC150 Q=50 →
A 2917, B 2905, C 2386 (VSS is 0.82× batched). Banking77 Q=50 → A 3541,
B 3757, C 3201 (0.85× batched).

Throughput per *request* is a different story: VSS wins requests/s only on
synthetic (9.2 vs 1.3 sequential at Q=50) and loses on both real datasets
(47.7 vs 58.3 on CLINC150).

Plots: `plots/{dataset}_throughput_vs_Q.svg`.

## 11. Resource Usage

CPU-only throughout; VRAM not applicable (no CUDA device present in the
environment block). Resident memory: plain ≈ 480 MB, VSS ≈ 640 MB at peak
during training; inference peak < 1 GB for both. All latency figures were
measured with 6 torch threads on an otherwise idle machine; the reported p95
and p99 values on real data are close to p50 (no batching queueing), which is
expected for single-request micro-benchmarks.

## 12. Plain Classifier: What It Is and How Strong It Is

The baseline is not a strawman, and that was enforced rather than assumed:

- It is verified to be the reason VSS previously looked good. The repository's
  own U1 ablation (plain head beating the VSS head 86.7% vs 74.0% on CLINC)
  predicted this outcome, and the value test reproduced it.
- Its only defect found during this work was a harness protocol defect
  (per-row masked CE) that made it artificially weak; that run was quarantined
  and the model retrained under the proven recipe (§5). Post-fix CLINC150
  validation loss dropped 4.66 → 0.395.
- It uses the same tokenizer, the same serialization path, the same hidden
  size class, the same splits, the same epochs, and a full-inventory head.

## 13. Option Permutation

For real datasets, all Q questions in a request are replicas of one canonical
question, so permuting them cannot change meaning; agreement measures
slot-position sensitivity only. Disclosed in the results JSON via a `note` key.

| Dataset | Q=8 | Q=32 |
|---|---:|---:|
| synthetic (distinct questions) | 0.776 / 0.794 / 0.788 (3 seeds) | 0.740 / 0.752 / 0.748 |
| clinc150 (replicated) | 0.973 | 0.975 |
| banking77 (replicated) | 0.923 | 0.956 |

## 14. Question Order

On the synthetic task — the only place with genuinely distinct questions —
**VSS's answers are only 74–79% order-invariant**: reordering the questions in
a request changes the answer for roughly a quarter of (state, question) pairs.
Plain is exactly order-invariant by construction (each question is an
independent input). This is a genuine limitation of the current VSS
implementation that the dose-response experiments did not surface, and it
explains part of VSS's request-accuracy collapse.

## 15. Statistical Results

- **Paired bootstrap (1000 resamples) of C−B per-question accuracy**: negative
  in all 21 synthetic cells and all 14 real-data cells. Synthetic CIs:
  [−0.380, −0.252] (Q=1) to [−0.200, −0.180] (Q=50). CLINC150 Q=50:
  [−0.272, −0.252]. Banking77 Q=50: [−0.435, −0.410]. Plain beats VSS on
  accuracy with high confidence everywhere measured.
- **Seed variance (synthetic, n=3)**: plain std ≤ 0.020 at every Q; VSS std
  ≤ 0.149 at Q=1 (abstention-heavy, high variance) and ≤ 0.075 elsewhere.
- **Real data is single-seed (13)**: no seed CI is available for CLINC150 or
  Banking77. Their confidence intervals come from the within-sample paired
  bootstrap only and therefore understate true uncertainty. This is a known
  limitation, not a claim of statistical power.
- **A≡B verification**: 35/35 cells, max probability difference ≤ 1e-6.

## 16. Failure Cases

1. **Real-data Q>1 measurement artifact (found and fixed).** Renaming slot 0's
   question id during replication pushed both systems onto unseen text. Before
   the fix, VSS appeared to collapse 0.68 → 0.28 (Banking77) and permutation
   "agreement" read 0.000; after the fix, CLINC150 is flat (0.447 → 0.397) and
   agreement reads 0.97. The pre-fix numbers are not reported anywhere as
   results.
2. **The pre-existing banking77 checkpoint was unmasked.** `runs/banking77-slot-ho`
   has no `qmask` flag; measured against it, VSS degraded 0.68 → 0.01 with
   Q and interference reached −43 pts. That is System *not*-C. A qmask
   Banking77 checkpoint was trained for this test (8 epochs, best eval 1.3206);
   its numbers are the ones reported.
3. **Abstention dominates real-data VSS accuracy.** VSS abstains on 27–49% of
   real questions at the shipped threshold 0.55. Since real test gold is
   always an in-scope label, every abstention is scored wrong. Decomposed
   interference (CLINC150 Q=8): solo 0.566 @ 66.3% coverage → joint 0.398 @
   51.5% coverage; of the −16.9 pt total, roughly −14.8 pts is lost coverage
   and −8.3 pts is answered-accuracy loss.
4. **Order sensitivity** (§14): ~25% of synthetic (state, question) pairs
   change answer under reordering.
5. **Synthetic VSS never converges at this data budget.** Its validation loss
   was still descending at the 8-epoch cap (2.21/2.48/2.26) while plain had
   plateaued (0.186/0.196/0.200). VSS's synthetic deficit is therefore partly
   an optimization-budget artifact, not necessarily a capacity ceiling.
6. **Harness limitation**: latency used 5 warmup + 20 measured iterations
   rather than the requested 100 + 500, because a mode-A iteration at Q=50 is
   50 sequential forwards on a CPU-only box. Percentiles are from 20 samples,
   so p95/p99 are coarse.

## 17. VSS vs Plain Classifier

Directly, on the primary metric (per-question accuracy, abstention counted as
incorrect), the plain classifier wins on all three datasets and every question
count, with CIs excluding zero. VSS does not win accuracy, does not win
calibration in most cells, and does not win latency on the real datasets.

VSS does win three things that the plain architecture cannot offer at all:
zero interference on the synthetic task, 2.8× the batched classifier's
question throughput at Q=50 on long-text requests, and a confidence gate that
produces an 85.5%-accurate subset where the plain classifier is 68.0% accurate.

## 18. Does VSS Provide Real Value?

**Partially, and not enough to justify the architecture as it stands.**

The value proposition decomposes into three separable claims:

1. *Multi-question efficiency* — **supported where questions are long and
   numerous** (2.8× batched at Q=50 synthetic), **not supported** on short
   real-world requests where VSS is 1.1–1.3× slower per request.
2. *No cross-question interference* — **supported on synthetic** (mean delta
   −0.13 to +2.7 pts across 3 seeds × 3 Q), **not supported on real data**
   (−9.1 to −16.9 pts, partly abstention-driven).
3. *Better decisions per question* — **not supported** (plain wins accuracy and
   NLL everywhere), **partially supported through selectivity** (answered-only
   accuracy 0.855 vs 0.680 on CLINC150).

The specialized architecture does not currently pay for its complexity on the
primary metric. What survives scrutiny is narrower and worth keeping: the
single-pass multi-question mechanism, the typed per-question heads, and the
confidence-gated abstention behavior.

## 19. What Is Proven

- **DEMONSTRATED**: at matched budget, a plain classifier beats VSS on
  per-question accuracy on synthetic, CLINC150 and Banking77 at every Q from
  1 to 50 (35 cells; paired bootstrap CIs exclude zero in all).
- **DEMONSTRATED**: VSS single-pass latency beats both sequential *and batched*
  plain inference by 7.0× and 2.8× respectively at Q=50 on long-text
  synthetic requests, and loses to both on short real-data requests.
- **DEMONSTRATED**: VSS shows no systematic cross-question interference on the
  synthetic task (mean |delta| ≤ 2.7 pts over 3 seeds × Q ∈ {8, 32, 50}), and
  does show interference on real data (−9 to −17 pts).
- **DEMONSTRATED**: VSS's confidence gate selects a substantially more accurate
  subset than the plain classifier delivers unconditionally on CLINC150
  (0.855 answered vs 0.680 always-on).
- **DEMONSTRATED**: VSS's answers are only ~74–79% order-invariant on distinct
  synthetic questions.
- **DEMONSTRATED**: A ≡ B for the plain network in all 35 cells
  (batching changes nothing on this deterministic model).

## 20. What Remains Unproven

- **UNKNOWN**: whether VSS closes the accuracy gap with a larger training
  budget or more parameters. Its synthetic validation loss was still falling
  at the 8-epoch cap; the experiment cannot separate "needs more training" from
  "worse architecture".
- **UNKNOWN**: whether VSS's real-data interference is intrinsic or an artifact
  of replicated identical questions with suffixed ids (the only real-data
  multi-question structure available).
- **UNKNOWN**: seed variance on real data (single seed 13 per system).
- **UNKNOWN**: whether the calibration gap closes with temperature scaling or
  per-head calibration, which was not attempted here.
- **UNKNOWN**: GPU/accelerator behaviour, batched-serving throughput under
  concurrency, and memory scaling — all CPU-only, single-request measurements.
- **PLAUSIBLE but unproven**: that the single-pass mechanism plus typed heads
  becomes the right choice at large Q with long, genuinely distinct questions.

## 21. Next Step

**Do not scale VSS to 52M parameters yet.** The decision tree resolves to
*"plain wins" → redesign*, with two narrow carve-outs worth preserving.

1. **Fix the two measured architectural defects before any scaling decision:**
   order-invariance (~25% of answers change when questions are reordered) and
   the abstention calibration that costs 15–17 points of real-data accuracy
   while providing only partial selectivity benefit.
2. **Re-run the value test with a matched, converged budget** (VSS trained to
   a flat validation curve, 3 seeds on both real datasets) so the accuracy
   comparison is not confounded by the optimization-budget gap.
3. **Add a genuinely multi-question real-data task.** Every real result here is
   limited by the fact that CLINC150 and Banking77 offer one natural question
   per state; the replicated-question protocol measures slot stability, not
   multi-question competence.
4. **Only then** decide whether a larger model is justified. The evidence
   available today does not support scaling.
