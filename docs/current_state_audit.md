# VSS Current-State Audit

Written as part of the validation pass. Every number below was re-measured
from code and checkpoints, not copied from earlier claims. Dates: 2026-09,
single machine, CPU-only training.

## 1. Architecture (measured)

Two trained configurations exist:

| | synthetic prototype | CLINC150 slot-ho |
|---|---|---|
| total parameters | 13,655,364 | 11,164,483 |
| encoder share | 12,586,240 (92.2%) | 10,489,088 (94.0%) |
| heads share | 1,069,124 (7.8%) | 675,395 (6.0%) |
| layers / hidden / heads | 8 / 256 / 8 | 6 / 256 / 8 |
| kv_heads (GQA) | 8 (= heads, no grouping) | 8 |
| intermediate (SwiGLU) | 1024 | 1024 |
| context | 4096 | 4096 |
| positional | RoPE theta 10000 | RoPE theta 10000 |
| normalization | RMSNorm (pre-norm) | RMSNorm (pre-norm) |
| vocab | 16384 word/hash buckets | 16384 word/hash buckets |

Notes:
- The tokenizer is a whitespace/regex word tokenizer + FNV-1a hash bucket
  for OOV. No subwords: rare words, typos, and morphology are unhandled
  by construction. This is a known limitation (see §5).
- The embedding table (16384 x 256 = 4.2M) is 92% unused: only ~750
  reserved tokens plus up to 8192 hash buckets are ever indexed. Roughly
  6.1M parameters are dead weight in the prototype config.
- Question representation: mean-pool of the contextual tokens inside each
  `<QUESTION>` block, then a per-question MLP adapter conditioned on the
  head type. Bidirectional attention over [STATE blocks][QUESTION blocks].
- Decision heads: shared 1024-slot choice projection (options hashed to
  slots; `use_refine=false` = pure slot logits), sigmoid noul head,
  64-bin score head (expected value over bin centers), and an auxiliary
  calibration head (P(correct) BCE).
- `header_only_choice` (new): choice question blocks serialize WITHOUT the
  option list. The head enforces schema membership through its slot mask.
  Added after the dilution experiment (§4); see validation report.

## 2. Training (measured)

Synthetic prototype run (`runs/prototype`):
- Data: generated synthetic support-desk examples, 2400 train / 480 eval.
- Loss: joint softmax over options+abstain (CE) + noul BCE + score Huber
  + ordinal CE + calibration BCE; equal weights.
- Optimizer AdamW, lr 3e-4 cosine, 5 epochs, seed 13, batch 32, fp32, CPU.
- Checkpoint selection: best eval LOSS on the same generator distribution.
- Calibration: temperature fitted on the eval split (LEAKED; see §5).

CLINC150 runs (`runs/clinc150*`):
- Data: official CLINC150 (clinc/clinc_oos, imbalanced) splits; 10,625
  train / 3,100 validation / 5,500 test. Training examples declare a
  subset schema (gold + 14 random distractors); validation/test use the
  full 151-option schema.
- Three training regimes tested (identical data, identical backbone):
  1. refine-CE (subset schemas, refine head): subset-val CE 0.894,
     full-schema test accuracy 6.2%.
  2. slot-CE (pure slot projection over all 151 intents): best subset
     CE 0.732 (ep 2); full-151 test accuracy 16.8%; subset protocol 81.7%.
  3. slot-CE + header-only blocks: best full-schema val CE 2.568 (ep 3);
     full-151 test accuracy 74.0% (best) / 73.9% (final epoch).
- Optimizer AdamW, lr 3e-4 cosine, 8 epochs, seed 13, batch 32, CPU.

## 3. Evaluation infrastructure

- `scripts/evaluate.py`: synthetic accuracy/F1/ECE/Brier/NLL/latency.
- `scripts/eval_realdata.py`: batched full-schema accuracy/F1/ECE/Brier/
  NLL + throughput + forward-pass count.
- `scripts/eval_ood.py`: OOS AUROC, validation-selected abstain threshold,
  coverage/selective-accuracy (risk-coverage), abstention recall/precision.
- `scripts/baseline_tfidf.py`: TF-IDF(1-2 ngram)+LogisticRegression, same
  split, same metrics.
- `benchmarks/leakage/audit_leakage.py`: duplicate/near-duplicate/generator
  similarity audit producing JSON reports.

## 4. Results summary (honest)

| Protocol | VSS (best config) | TF-IDF+LR |
|---|---|---|
| Synthetic, seen templates (leaky) | acc 1.000, ECE 0.004 | — |
| Synthetic, held-out templates | acc 0.3125, ECE 0.65 | not run |
| CLINC150 full-151 schema | acc 0.740, ECE 0.040 | acc 0.888, ECE 0.297 |
| CLINC150 OOS AUROC | 0.800 | 0.889 |
| CLINC150 risk-coverage | monotone, 99.6% @50% cov | not run |

The synthetic 1.000/ECE 0.004 numbers are template memorization artifacts
(75/480 test examples were exact duplicates of training examples; 93% had
Jaccard >= 0.7 to a train example). They must not be cited as evidence of
generalization.

## 5. Known problems (verified)

1. Generator/template leakage (fixed in data pipeline, historical results
   invalid): shared template banks produced 92.9% near-duplicate test
   examples. Held-out-template accuracy collapsed to 31.25%.
2. Calibration/temperature fitting on the eval split (fixed: calibration
   must use a dedicated calibration split; the OOD script selects thresholds
   on validation only).
3. Checkpoint selection on eval loss of a mismatched protocol (subset
   schemas during training vs full schema at deployment) hid a ranking
   inconsistency; fixed by matched-protocol validation (full-schema val
   subset) once header-only serialization made it tractable.
4. Choice dilution (root-caused, fixed): option text inside the question
   block diluted the mean-pooled question vector; full-schema accuracy 8.6%
   vs 52.4% for the same weights with a minimal block.
5. Question-text-size brittleness remains: accuracy depends on the token
   length of the question block relative to the state; no length
   normalization in pooling (candidate fix: attention pooling or a [CLS]-
   style readout).
6. Tokenizer: word+hash has no subword fallback; real-data typos and
   rare intents rely on hash collisions. Comparisons vs BPE pending.
7. Class imbalance: CLINC 'oos' dominates the option prior at eval when
   intent discrimination is weak (seen as 'oos' winning logits in the
   refine-CE run).
8. Multi-seed evidence: all results are single-seed (seed 13); CIs are
   not yet established (the bootstrap over 4500 test examples suggests
   roughly +/-1.5-2% on CLINC accuracy, but this is not a seed CI).
9. Latency numbers for the prototype were single-run medians without cold
   vs warm separation; a proper scaling benchmark is still pending.

## 6. What the current evidence supports

- Demonstrated: single-pass multi-question inference; schema-constrained
  outputs; training-objective and serialization changes causally move
  real-data accuracy by tens of points; risk-coverage monotonicity.
- Not demonstrated: superiority over classical baselines (VSS trails
  TF-IDF+LR by ~15 accuracy points and ~0.09 AUROC); generalization
  across domains; calibration under shift; multi-seed stability.
