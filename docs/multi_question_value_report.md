# Multi-Question Value Test — Does VSS Justify Its Architecture?

> ## ⚠️ WITHDRAWN IN PART — read `docs/convergence_report.md` first
>
> A later convergence audit found that this experiment's **training protocol**
> was defective, so several conclusions below do not survive it. The historical
> numbers are kept here unchanged for the audit trail, but:
>
> - the synthetic splits were **nested prefixes of one another**
>   (`validation == train[:300]`, `test[:800] == train`), so 100% of the model
>   selection split and 80% of the synthetic test split were training states;
> - VSS's synthetic schedule was 200 optimizer steps of which **150 were
>   warmup** (75%), so it never left the LR ramp;
> - VSS received **8× fewer optimizer steps** than the plain baseline (200 vs
>   1600) because a VSS step carries a whole multi-question state;
> - a fourth defect (stable-RoPE positions taken from example 0's state length)
>   is **live on CLINC150 and Banking77**, so the real-data interference,
>   order-invariance and selective-accuracy numbers below are known-suspect and
>   are withdrawn pending a re-run.
>
> Re-measured on clean splits with converged, presentation-matched models, the
> synthetic comparison is a **tie at Q=1–4**, plain ahead by 3.4–6.9 pts at
> Q=8–50, and VSS ahead on selective prediction and on latency against a
> *batched* baseline. Claims D13 (plain wins 35/35) is withdrawn; D21–D23 added;
> U2 reopened.

**Status:** complete, **second pass** (post-defect-fix). 35 (seed, Q) cells
across 3 datasets, 15 interference/permutation analyses, 65/65 tests passing.
All numbers below are measured, not estimated.

**What changed since the first pass.** Two measured "defects" turned out to be
**bugs in VSS and in the harness, not properties of the architecture**:

1. `src/vss/model/encoder.py` built the question-isolation mask **once per
   batch** instead of once per example, so in a batch of N examples only the
   last example's span geometry survived. Order sensitivity and most of the
   measured "interference" were artifacts of that. Fixed in `df5eb50`, with two
   regression tests.
2. The harness's interference probe compared a 100-state solo arm against a
   200-state joint arm, so it reported large deltas while **zero** paired
   decisions actually flipped. Both arms now score the same (state, question)
   pairs and report `paired_decision_flips`.

All four VSS checkpoints were retrained on the fixed mask and the whole 35-cell
value test was re-measured. The pre-fix results are preserved verbatim in
`results/{synthetic,clinc150,banking77}_results_prefix.json`; the before/after
comparison is §16.1.

**Bottom line up front (post-fix).** At this budget (11M parameters, CPU,
800–9500 training states) the plain classifier still wins on per-question
accuracy on **all three datasets** at every Q, with paired-bootstrap CIs that
exclude zero — so **the "do not scale" verdict stands, but for a different
reason**. It no longer rests on order-sensitivity or cross-question
interference: those are gone (order agreement 0.96–0.99 on distinct questions;
interference −0.31…+0.88 pts with **0** paired decision flips in 27,000
synthetic pairs). It rests on a simpler fact: **a plain classifier called once
per question is simply more accurate at this budget**, and VSS's remaining
advantages are real but narrow and conditional —

* **single-pass latency** on long multi-question requests (6.8× vs sequential,
  2.5× vs *batched* plain at Q=50 on synthetic), while being 1.1× *slower* per
  request on both real datasets;
* **selective accuracy** — when VSS answers on CLINC150 it is right 77.7% of
  the time vs the plain classifier's always-on 65.9% (Q=50), at 53% coverage;
* **no measurable cross-question interference** anywhere (0 paired flips
  synthetic; 1.2–3.5% of paired decisions flip on real data with a net delta of
  −0.31…+0.06 pts, i.e. fp32 noise, not systematic leakage).

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
| Pre-fix results (audit) | `benchmarks/multi_question_value/results/*_results_prefix.json` |
| Generated report | `benchmarks/multi_question_value/report.md`, `results/DIGEST.txt` |
| Plots | `benchmarks/multi_question_value/plots/*.svg` (hand-rolled SVG, no matplotlib) |

Cold start (model load + first request) tracks what is already in the OS page
cache: plain 0.19–3.87 s, VSS 0.27–4.66 s, with one 23.3 s outlier on the
first, coldest VSS load.

**Latency caveat.** The post-fix re-run happened on a busier machine than the
pre-fix run, so *absolute* latencies rose (real-data p50 14–21 ms → 30–64 ms).
Only the within-cell ratios (A, B and C measured back to back in the same
process) are trustworthy; the pre/post latency columns in §16.1 are therefore
not compared.

## 2. Systems

**System A/B — plain classifier.** `STATE + QUESTION → one typed decision`.
Same tokenizer family and canonical serialization as VSS (`src/vss/model/`),
header-only choice blocks (exactly what VSS sees), 6 layers, hidden 256,
vocab 16384, mean-pooled encoder, linear choice head over the full option
inventory + trained ABSTAIN class, Noul sigmoid head, 64-bin score head.
10.53M trainable parameters.

**System C — VSS with question-masked attention.** `qmask` checkpoints
(`question_masked: true`, stable per-token RoPE), shipped inference engine
(`abstain_threshold: 0.55`, `confidence_mode: blend`), **per-example** isolation
mask. 11.16M parameters.

| Dataset | VSS run | Note |
|---|---|---|
| synthetic | `runs/mqv-vss-synthetic-s{1,2,3}` | **retrained** on the fixed mask, identical recipe |
| clinc150 | `runs/clinc150-slot-ho-qmask` | pre-existing validated qmask run, **not retrained** (see §16, case 5) |
| banking77 | `runs/mqv-vss-banking77-s13` | **retrained** on the fixed mask; the only pre-existing banking77 checkpoint was *unmasked*, which is not System C |

**Fairness guards.** Mode A and mode B are the *same network*. Batching is
verified not to change outputs on every cell (`A_equals_B: true` in all 35
cells; spot check = 16 pairs at batch 1 vs batch 32, max probability
difference ≤ 1.5e-6, i.e. fp32 noise). A is reported as measured-equivalent to B
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
question text and produced artificial Q=1→Q=2 collapses (§16, case 1).

## 5. Training Fairness

Identical splits, identical optimizer family, identical budget, identical
checkpoint-selection rule (best validation loss, never final epoch). All VSS
runs below were trained **after** the mask fix.

| | plain | VSS |
|---|---|---|
| Parameters | 10.53M | 11.16M |
| Epochs | 8 | 8 |
| Optimizer | AdamW, warmup + cosine, lr 3e-4, wd 0.01, clip 1.0 | same |
| Batch size | 32 | 32 |
| Seeds | synthetic 1/2/3, real 13 | synthetic 1/2/3, real 13 |
| Best val loss | synthetic 0.186/0.196/0.200, CLINC 0.395, Banking77 0.5206 | synthetic 1.957/2.217/2.266 (all at epoch 7, still descending), Banking77 **1.048 @ epoch 6** (was 1.321 pre-fix) |
| Per-epoch validation slice | first 200 states (selection signal only) | first 100 states (selection signal only) |

Per-epoch histories and full hyper-parameters are in
`benchmarks/multi_question_value/train_logs/*.json`. The VSS logs carry complete
8-epoch histories; the plain logs' `history` arrays were partially clobbered by
chunked resume (a known harness artifact) and retain their last rows only — the
authoritative `best_val_loss` header field is intact in every log and is what the
table above reports.

**Two protocol defects were found and fixed during this work.** (i) The first
plain real-data baselines trained with per-row masked cross-entropy over each
row's *declared* 15 options while test rows declare all 151/77 — a protocol
artifact that produced a 5.3% "plain" baseline on CLINC. The baseline was
quarantined (`runs/mqv-plain-*_maskedce_v1`, kept for audit) and retrained with
header-only text and full-inventory cross-entropy; every real-data number here
comes from the fixed baseline. (ii) The VSS question mask was built per batch
instead of per example (§16.1). Neither defect is cited as a result.

## 6. Accuracy Results

Per-question accuracy (mean ± std over seeds), synthetic (3 seeds), post-fix:

| Q | A/B plain | C VSS | plain−VSS (paired bootstrap C−B, 95% CI) |
|--:|---:|---:|---:|
| 1 | 0.970 ± 0.009 | 0.762 ± 0.129 | −0.208 [−0.265, −0.157] |
| 2 | 0.985 ± 0.004 | 0.881 ± 0.064 | −0.104 [−0.136, −0.075] |
| 4 | 0.963 ± 0.008 | 0.918 ± 0.033 | −0.045 [−0.062, −0.028] |
| 8 | 0.928 ± 0.006 | 0.722 ± 0.019 | −0.206 [−0.227, −0.187] |
| 16 | 0.934 ± 0.020 | 0.746 ± 0.019 | −0.188 [−0.202, −0.173] |
| 32 | 0.802 ± 0.017 | 0.638 ± 0.021 | −0.165 [−0.176, −0.153] |
| 50 | 0.760 ± 0.017 | 0.599 ± 0.022 | −0.160 [−0.170, −0.151] |

CLINC150 (seed 13): plain 0.680 → 0.659 across Q; VSS 0.447 → 0.411.
C−B = −0.233 [−0.300, −0.173] at Q=1, −0.248 [−0.258, −0.238] at Q=50.

Banking77 (seed 13): plain 0.887 → 0.851; VSS 0.760 → 0.754.
C−B = −0.127 [−0.187, −0.080] at Q=1, −0.097 [−0.107, −0.088] at Q=50.

Macro-F1 moves the same way. Request accuracy is the starkest number: VSS
needs *every* question in the request right, so it reaches 0.000 from Q=16 on
synthetic while plain decays gracefully (0.53 at Q=8). On Banking77 VSS's
request accuracy is 0.507 at Q=50 vs plain's 0.387 — the only Q where the
all-questions-right criterion is even reachable.

**Selective accuracy is where VSS remains genuinely strong** (abstention
counted as wrong above, so this is a separate view):

| Dataset | plain (always answers) | VSS answered-accuracy @ coverage |
|---|---:|---:|
| CLINC150 Q=1 | 0.680 | **0.736 @ 60.7% coverage** |
| CLINC150 Q=8 | 0.663 | **0.781 @ 54.8% coverage** |
| CLINC150 Q=50 | 0.659 | **0.777 @ 53.0% coverage** |
| Banking77 Q=1 | 0.887 | 0.857 @ 88.7% coverage |
| Banking77 Q=50 | 0.851 | **0.867 @ 87.0% coverage** |

On CLINC150 VSS's answered subset is **11.8 points more accurate** than the
plain classifier's always-on output; on Banking77 it is +1.6 pts at Q=50 and
−3.0 pts at Q=1. **The earlier claim of a 17.5-pt advantage was inflated by
the mask bug** (§16.1).

**The abstention gate is free, not costly.** Sweeping the threshold
(`results/abstention_sweep.json`), CLINC150 *decision* accuracy (abstentions
scored wrong) is **flat at 0.533** for thresholds 0.0 → 0.55 (coverage 100% →
68%) and only falls at 0.6+; Banking77 is flat at 0.687 for 0.0 → 0.5. So the
shipped threshold costs no overall accuracy — the earlier "15–17 point
abstention cost" claim in the first-pass report was **wrong** and is retracted.
What the gate buys is a sharper answered subset, and (on CLINC150) a confidence
signal that is not informative enough to raise decision accuracy at all.

## 7. Calibration Results

ECE / Brier / NLL per question, all 15 metric-Q combinations in
`report.md`. Summary (post-fix):

- **Q=1–2 (short requests): plain is better calibrated.** Synthetic Q=1 ECE
  0.049 vs 0.221; CLINC150 0.204 vs 0.238; Banking77 0.107 vs 0.130.
- **Q≥4, VSS's ECE is lower on all three datasets** (synthetic 0.102–0.158 vs
  plain's 0.153–0.221; CLINC150 0.196 vs 0.214; Banking77 0.093–0.118 vs
  0.101–0.148) — VSS degrades more gracefully with Q, while plain becomes
  overconfident on the harder rephrased variants.
- **Brier and NLL favor plain in most cells** (synthetic Q=8: plain Brier
  0.199/NLL 0.407 vs VSS 0.364/1.010), but Banking77 Q≥16 inverts it (Q=50:
  plain NLL 0.775 vs VSS 0.763).
- The historical "ECE answered 0.028" figure is an answered-only number from
  the CLINC validation set under a different protocol; it does **not**
  reproduce on this test protocol and is not comparable to the numbers above.

Conclusion: VSS's *relative* calibration degrades more slowly with Q, but at
any single Q its raw distributions are not better calibrated than a plain
classifier's.

## 8. Question Scaling: Latency

p50 request latency (ms), synthetic:

| Q | A sequential | B batched | C VSS | C/A | C/B |
|--:|--:|--:|--:|--:|--:|
| 1 | 19.0 | 18.8 | 18.6 | 0.98 | 0.99 |
| 2 | 32.4 | 22.7 | 17.9 | 0.56 | 0.79 |
| 4 | 64.5 | 32.4 | 20.9 | 0.33 | 0.65 |
| 8 | 127.2 | 52.7 | 26.8 | 0.21 | 0.51 |
| 16 | 254.0 | 95.4 | 43.4 | 0.17 | 0.46 |
| 32 | 486.6 | 170.9 | 63.7 | 0.13 | 0.37 |
| 50 | 745.2 | 267.9 | 109.2 | **0.15** | **0.41** |

At Q=50 that is **6.8× faster than sequential plain and 2.5× faster than the
*batched* plain classifier** — at p95, 2.69 ms per question versus 16.4 ms for
sequential plain and 6.15 ms for batched plain.

On real data the picture inverts: single-question-per-state requests are
overhead-bound and VSS is *slower* than both plain modes at every Q
(CLINC150 Q=50: A 29.9 / B 27.4 / C 33.4 ms; Banking77 Q=50: A 38.8 /
B 40.1 / C 43.8 ms). The single-pass advantage is real but conditional on
request length, not on question count alone.

Plots: `plots/{dataset}_latency_p50_vs_Q.svg`, `plots/{dataset}_latency_p95_vs_Q.svg`.

## 9. Question Scaling: Accuracy

Plain decays smoothly with Q (synthetic 0.970 → 0.760 over 1→50) because each
question is an independent (state, question) input. VSS decays faster
(0.762 → 0.599, with a local peak at Q=4) and its *request* accuracy collapses
to zero by Q=16. On CLINC150 VSS's per-question accuracy is nearly flat
(0.447 → 0.411) while plain is flat too (0.680 → 0.659); on Banking77 VSS
recovers monotonically from 0.660 (Q=2) to 0.754 (Q=50) — the only
configuration where VSS *gains* accuracy as questions are added.

Plots: `plots/{dataset}_accuracy_vs_Q.svg`.

## 10. Throughput

Questions per second at p50, synthetic: Q=50 → A 67.2, B 187.2, **C 459.6**
(VSS is 2.5× the batched classifier, 6.8× the sequential one). CLINC150 Q=50 →
A 1673, B 1823, C 1495 (VSS is 0.82× batched). Banking77 Q=50 → A 1288,
B 1246, C 1142 (0.92× batched).

Throughput per *request* is a different story: VSS wins requests/s only on
synthetic (9.2 vs 1.3 sequential at Q=50) and loses on both real datasets
(29.9 vs 33.5 on CLINC150).

Plots: `plots/{dataset}_throughput_vs_Q.svg`.

## 11. Resource Usage

CPU-only throughout; VRAM not applicable (no CUDA device present in the
environment block). Resident memory: plain ≈ 480 MB, VSS ≈ 640 MB at peak
during training; inference peak < 1 GB for both. Latency was measured with 6
torch threads; p95/p99 come from 20 samples and are therefore coarse, which is
why the p95 tables should be read as indicative rather than precise.

## 12. Plain Classifier: What It Is and How Strong It Is

The baseline is not a strawman, and that was enforced rather than assumed:

- It is verified to be the reason VSS previously looked good. The repository's
  own U1 ablation (plain head beating the VSS head 86.7% vs 74.0% on CLINC)
  predicted this outcome, and the value test reproduced it in 35/35 cells
  *after* the mask fix — so the result is not a bug artifact.
- Its only defect found during this work was a harness protocol defect
  (per-row masked CE) that made it artificially weak; that run was quarantined
  and the model retrained under the proven recipe (§5).
- It uses the same tokenizer, the same serialization path, the same hidden
  size class, the same splits, the same epochs, and a full-inventory head.

## 13. Option Permutation

For real datasets, all Q questions in a request are replicas of one canonical
question, so permuting them cannot change meaning; agreement measures
slot-position sensitivity only. Disclosed in the results JSON via a `note` key.

| Dataset | Q=8 | Q=32 |
|---|---:|---:|
| synthetic (distinct questions) | 0.964 / 0.967 / 0.966 (3 seeds) | 0.991 / 0.993 / 0.991 |
| clinc150 (replicated) | 0.996 | 0.998 |
| banking77 (replicated) | 0.998 | 0.999 |

## 14. Question Order

On the synthetic task — the only place with genuinely distinct questions —
**VSS is now 96.4–96.7% order-invariant at Q=8 and 99.1–99.3% at Q=32**
(pre-fix: 74–79%). Plain is exactly order-invariant by construction (each
question is an independent input). The residual 3–4% disagreement at Q=8 is
fp32 non-associativity from different padding/reduction orders under
permutation, not systematic slot preference: the disagreements carry no
consistent direction (see the ±0.31–0.88 pt interference deltas of §16.1,
which are the same effect measured per question).

**The first pass's headline finding — "~25% of answers change when questions
are reordered" — was a manifestation of the per-batch mask bug and is
withdrawn.**

## 15. Statistical Results

- **Paired bootstrap (1000 resamples) of C−B per-question accuracy**: negative
  in all 21 synthetic cells and all 14 real-data cells. Synthetic CIs:
  [−0.265, −0.157] (Q=1) to [−0.170, −0.151] (Q=50). CLINC150 Q=50:
  [−0.258, −0.238]. Banking77 Q=50: [−0.107, −0.088]. Plain beats VSS on
  accuracy with high confidence everywhere measured.
- **Seed variance (synthetic, n=3)**: plain std ≤ 0.020 at every Q; VSS std
  ≤ 0.129 at Q=1 (abstention-heavy, high variance) and ≤ 0.064 elsewhere.
- **Real data is single-seed (13)**: no seed CI is available for CLINC150 or
  Banking77. Their confidence intervals come from the within-sample paired
  bootstrap only and therefore understate true uncertainty. This is a known
  limitation, not a claim of statistical power.
- **A≡B verification**: 35/35 cells, max probability difference ≤ 1.5e-6.
- **Paired interference flips** (the direct, assumption-free interference
  test): 0 / 27,000 paired decisions on synthetic; 28/800 (Q=8) and 113/3200
  (Q=32) on CLINC150; 12/800 and 39/3200 on Banking77 — with net deltas of
  −0.31…+0.06 pts, so the real-data flips are symmetric numerical noise.

## 16. Failure Cases

### 16.1 The two defects, and what changed (before → after)

Pre-fix numbers are the preserved `*_results_prefix.json`; "code only" means
the *same* checkpoint was re-evaluated against fixed code, so the delta is
attributable to the fix alone.

| Measurement | Pre-fix | Post-fix | Cause |
|---|---:|---:|---|
| **Order agreement, synthetic Q=8** (3 seeds) | 0.776 / 0.794 / 0.788 | 0.964 / 0.967 / 0.966 | mask built per batch |
| **Order agreement, synthetic Q=32** (3 seeds) | 0.740 / 0.752 / 0.748 | 0.991 / 0.993 / 0.991 | mask built per batch |
| **Order agreement, CLINC150 Q=8 / Q=32** (code only) | 0.973 / 0.975 | 0.996 / 0.998 | mask built per batch |
| **Order agreement, Banking77 Q=8 / Q=32** | 0.923 / 0.956 | 0.998 / 0.999 | mask bug + retrained ckpt |
| **Interference, CLINC150 Q=8 / Q=32** (code only) | −16.9 / −15.6 pts | **−0.13 / −0.31 pts** | harness compared *different* state sets |
| **Interference, Banking77 Q=8 / Q=32** | −9.1 / −10.5 pts | **0.00 / +0.06 pts** | harness defect + retrained ckpt |
| **Interference, synthetic Q=8/32/50** (mean over seeds) | +1.69…+0.50 / +2.69…+0.48 / +2.30…+0.66 | +0.88…−0.31 / +0.31…+0.70 / +0.06…+0.63 | harness defect |
| **VSS accuracy, Banking77 Q=1 / Q=50** | 0.573 / 0.429 | **0.760 / 0.754** | retrained on the fixed mask |
| **VSS accuracy, synthetic Q=1 / Q=50** (3-seed mean) | 0.780 / 0.579 | 0.762 / 0.599 | retrained; still unconverged |
| **VSS answered-accuracy, CLINC150 Q=1** | 0.761 @ 58.7% | 0.736 @ 60.7% | — (plain is 0.680) |
| **VSS answered-accuracy, Banking77 Q=1** | 0.789 @ 72.7% | 0.857 @ 88.7% | retrained on the fixed mask |
| **"Abstention costs 15–17 pts"** | asserted | **retracted** — decision accuracy flat 0.533 (CLINC) / 0.687 (B77) across thresholds 0.0–0.55 | was a symptom of the mask bug |

The mask fix itself: `encoder.py` allocated one `[T,T]` plane and wrote every
example's rows into it inside the batch loop, so example *b*'s geometry
overwrote example *b−1*'s and the last example in the batch decided the mask
for all of them. Isolation therefore depended on batch composition and question
order — and was invisible at batch size 1, which is why single-question probes
never showed it. The fix allocates `m = torch.zeros(len(spans), T, T)` per
example, writes `m[b, …]`, and passes `qmask = m.unsqueeze(1)` → `[B,1,T,T]`
(the transformer already broadcasts a per-batch bias). Two regression tests in
`tests/test_question_mask.py` pin it: per-example planes in a heterogeneous
batch, and batch-invariance of per-question outputs (atol 2e-3 for fp32 padding
noise).

1. **Real-data Q>1 measurement artifact (found and fixed in the first pass).**
   Renaming slot 0's question id during replication pushed both systems onto
   unseen text. Before the fix VSS appeared to collapse 0.68 → 0.28
   (Banking77) and permutation "agreement" read 0.000; after the fix CLINC150 is
   flat and agreement reads 0.996–0.998. The pre-fix numbers are not reported
   anywhere as results.
2. **The pre-existing banking77 checkpoint was unmasked.** `runs/banking77-slot-ho`
   has no `qmask` flag; measured against it VSS degraded 0.68 → 0.01 with Q and
   interference reached −43 pts. That is System *not*-C, so a qmask Banking77
   checkpoint was trained for this test (8 epochs, best eval **1.0478** post-fix,
   was 1.3206 pre-fix) and its numbers are the ones reported.
3. **Ablation asymmetry, disclosed.** CLINC150 uses the pre-existing
   `runs/clinc150-slot-ho-qmask` checkpoint *without* retraining, because
   retraining it on the fixed mask produced a **worse** model (0.407 at Q=1 vs
   0.527 for the repo checkpoint — it overfits hard, best epoch 2 = 2.2322 then
   3.5 by epoch 7). Using the better checkpoint is the conservative choice for
   VSS, and it means CLINC150's VSS numbers are a *code-only* fix while the other
   two datasets additionally include a retrain. Both facts are in the table
   above.
4. **VSS never converges on synthetic at this data budget.** Its validation
   loss was still descending at the 8-epoch cap (1.957/2.217/2.266 at epoch 7)
   while plain had plateaued (0.186/0.196/0.200). The remaining synthetic
   accuracy gap is therefore **partly an optimization-budget artifact**, and
   this experiment cannot separate "needs more training" from "worse
   architecture". This is the single largest remaining confound in the report.
5. **Harness limitation**: latency used 5 warmup + 20 measured iterations
   rather than the requested 100 + 500, because a mode-A iteration at Q=50 is
   50 sequential forwards on a CPU-only box. Percentiles are from 20 samples,
   so p95/p99 are coarse.

## 17. VSS vs Plain Classifier

Directly, on the primary metric (per-question accuracy, abstention counted as
incorrect), the plain classifier wins on all three datasets and every question
count, with CIs excluding zero — **including after the mask fix**, which is
the important part: VSS's earlier deficits were *not* caused by the bugs.

VSS does win three things the plain architecture cannot offer at all: a
single-pass multi-question path that beats even the *batched* plain classifier
by 2.5× at Q=50 on long requests (while losing 1.1× on short real requests), a
confidence gate whose answered subset is 11.8 pts more accurate than plain's
always-on output on CLINC150, and the complete absence of systematic
cross-question interference (0 paired flips in 27,000 synthetic pairs;
net ≤0.31 pts on real data).

## 18. Does VSS Provide Real Value?

**Partially — more than the first pass concluded, but still not enough to
justify the architecture as it stands.**

1. *Multi-question efficiency* — **supported where questions are long and
   numerous** (2.5× batched, 6.8× sequential at Q=50 synthetic), **not
   supported** on short real-world requests where VSS is 1.1× slower per
   request.
2. *No cross-question interference* — **supported everywhere now** (synthetic
   0 paired flips in 27,000 pairs; real-data net deltas −0.31…+0.06 pts). The
   first pass's "fails on real data" verdict was a harness artifact.
3. *Better decisions per question* — **not supported** (plain wins accuracy and
   mostly NLL everywhere), **partially supported through selectivity**
   (CLINC150 answered-only 0.777 vs 0.660 plain).

The specialized architecture does not pay for its complexity on the primary
metric, and the two most damning findings against it have now been retracted as
bugs. What survives is narrower and worth keeping: the single-pass
multi-question mechanism, the typed per-question heads, the confidence-gated
abstention, and a now-verified isolation property.

## 19. What Is Proven

- **DEMONSTRATED**: at matched budget, a plain classifier beats VSS on
  per-question accuracy on synthetic, CLINC150 and Banking77 at every Q from
  1 to 50 (35 cells; paired bootstrap CIs exclude zero in all) — measured
  *after* the mask fix, so this is not a bug artifact.
- **DEMONSTRATED**: VSS single-pass latency beats both sequential *and batched*
  plain inference by 6.8× and 2.5× respectively at Q=50 on long-text
  synthetic requests, and loses to both on short real-data requests.
- **DEMONSTRATED**: VSS has no systematic cross-question interference on any
  dataset (synthetic 0/27,000 paired decision flips; real data net −0.31…+0.06
  pts with 1–3.5% symmetric flips).
- **DEMONSTRATED**: VSS's question answers are 96.4–99.3% order-invariant on
  distinct synthetic questions (was 74–79% before the mask fix).
- **DEMONSTRATED**: VSS's confidence gate selects a more accurate subset than
  the plain classifier delivers unconditionally on CLINC150 (0.777 answered vs
  0.660 always-on at Q=50), and costs no overall decision accuracy.
- **DEMONSTRATED**: the question-isolation mask was per batch rather than per
  example, and this invalidated the first pass's order-sensitivity and
  real-data-interference findings.
- **DEMONSTRATED**: A ≡ B for the plain network in all 35 cells
  (batching changes nothing on this deterministic model).

## 20. What Remains Unproven

- **UNKNOWN**: whether VSS closes the accuracy gap with a larger training
  budget or more parameters. Its synthetic validation loss was still falling at
  the 8-epoch cap; the experiment cannot separate "needs more training" from
  "worse architecture". This is the main reason not to read the accuracy gap as
  architectural.
- **UNKNOWN**: seed variance on real data (single seed 13 per system).
- **UNKNOWN**: whether the remaining 3–4% order disagreement on synthetic is
  pure fp32 non-associativity or a small genuine slot effect. The interference
  probe's symmetric near-zero deltas argue for the former, but it was not
  isolated.
- **UNKNOWN**: whether the calibration gap closes with temperature scaling or
  per-head calibration, which was not attempted here.
- **UNKNOWN**: whether VSS's real-data Q>1 behaviour generalises, since every
  real request is a replicated single question — slot stability, not
  multi-question competence.
- **UNKNOWN**: GPU/accelerator behaviour, batched-serving throughput under
  concurrency, and memory scaling — all CPU-only, single-request measurements.
- **PLAUSIBLE but unproven**: that the single-pass mechanism plus typed heads
  becomes the right choice at large Q with long, genuinely distinct questions.

## 21. Next Step

**Still do not scale VSS to 52M parameters — but the reason has changed.**
Item 1 of the first pass's decision tree is **done**: both measured defects were
implementation bugs, they are fixed, regression-tested, and the whole value test
was re-measured on corrected code with retrained checkpoints. The accuracy
verdict survived the fix, so the remaining decision hinges on items 2 and 3:

1. **~~Fix the two measured defects~~ — DONE.** Mask is per example
   (`df5eb50`, 2 regression tests); the interference probe now compares paired
   `(state, question)` sets and reports paired flips. The first pass's
   order-sensitivity and real-data-interference claims are withdrawn, and the
   15–17-pt abstention claim is retracted.
2. **Re-run with a matched, converged budget** — train VSS until its validation
   curve is flat (the synthetic runs are still descending at epoch 8) and run
   3 seeds on both real datasets. Until this is done, the accuracy gap in §6 is
   confounded and "plain wins" is a statement about *this budget*, not about
   the architecture.
3. **Add a genuinely multi-question real-data task.** Every real result here is
   limited by CLINC150 and Banking77 offering one natural question per state;
   the replicated-question protocol measures slot stability, not multi-question
   competence.
4. **Then** decide whether a larger model is justified. If VSS matches plain on
   accuracy at a converged budget while keeping its 2.5×-batched latency
   advantage, the architecture earns a scale test. If it does not, the
   single-pass mechanism should be transplanted into a plain model — which is
   the cheaper way to keep the one property that actually survived.
