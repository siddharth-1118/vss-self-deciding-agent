# VSS architecture

VSS is a **System-One decision model**: it receives a state and a set of
typed questions and answers all of them in a single transformer forward
pass. It is not a text generator and never needs an LLM at inference time.

```
                ┌─────────────────────┐
State ─────────→│  canonical          │
Questions ─────→│  serialization      │
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │ TransformerEncoder  │   RoPE, RMSNorm, SwiGLU, GQA
                │ (bidirectional)     │   padding-masked attention
                └──────────┬──────────┘
                           │  mean-pool each <QUESTION> block span
                           ▼
                ┌─────────────────────┐
                │ QuestionAdapter     │   type-conditioned refinement
                └──────────┬──────────┘
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ChoiceHead        NoulHead         ScoreHead
    (hashed option    (sigmoid)        (64 ordinal bins,
     slots + learned                    expected value)
     abstain logit)                      │
          └────────────────┼──────────────┘
                           ▼
                  CalibrationHead
                  P(answer correct | q, a)
                           ▼
              typed JSON answers (schema-checked)
```

## Provenance

Every design element is tagged:

1. **[public]** — publicly documented System-One concepts: typed questions
   (Choice / Noul / Score), multiple questions per state, parallel outputs,
   probability distributions, confidence, schema-constrained answers,
   transformer-based architecture, calibration emphasis, speed emphasis.
2. **[std]** — standard ML techniques: RoPE, RMSNorm, SwiGLU, grouped-query
   attention, cross-entropy/BCE/Huber losses, temperature scaling, ECE.
3. **[vss]** — VSS-original engineering decisions (made because the
   proprietary details are unknown; measured against baselines before
   being kept):

   - canonical `<STATE>`/`<QUESTION>` serialization with atomic tag tokens
   - word tokenizer + FNV-1a OOV hashing into extra embedding buckets
   - question vectors = mean-pooled `<QUESTION>` block spans (questions are
     siblings of the state, not repeated text)
   - choice options mapped into a shared slot space by hashing, so ANY
     option set works with the trained weights (dynamic candidate masking)
   - learned per-choice-head abstain logit trained against `ABSTAIN`
     targets; threshold abstention at the API layer
   - score head = ordinal bin distribution; expected value = score;
     confidence = calibration head's P(correct)
   - confidence modes: `max_prob`, `calibrated_head`, `blend`

## Why questions are not repeated per state token

The mission forbids naive repeated concatenation of question text. Here the
state is serialized once; each question adds one small block. The encoder is
bidirectional, so each question block attends to the whole state and vice
versa. One forward pass serves N questions; measured latency scales ~10x
(not 50x) from 1 -> 50 questions.

## Calibration

Confidence is not "max softmax and call it calibrated":

- choice/noul: class probability is the base signal
- score: the auxiliary `CalibrationHead` outputs P(answer correct | inputs),
  trained on empirical correctness every training step
- post-hoc temperature scaling is fitted on held-out data
  (`scripts/calibrate.py`); ECE/Brier/NLL and reliability buckets are
  reported by `scripts/evaluate.py`

## Scaling

| preset | params | hidden | layers | heads | kv_heads | ffn |
|---|---|---|---|---|---|---|
| vss-prototype | ~13.7M | 256 | 6 | 8 | 8 | 1024 |
| vss-small | ~52M | 512 | 10 | 8 | 4 | 2048 |
| vss-base | ~250M | 768 | 16 | 12 | 4 | 3072 |
| vss-large | ~600M | 1024 | 20 | 16 | 4 | 4096 |

All values are config; nothing is hard-coded. Train Small/Base/Large only
after the previous tier's experiments justify it.
