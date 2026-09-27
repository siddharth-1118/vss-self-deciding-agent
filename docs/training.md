# Training VSS

## Data

Training examples are JSONL:

```json
{"state": {"message": "My card was charged twice."}, "questions": [
  {"id": "department", "type": "choice", "options": ["billing", "technical", "sales", "other"], "answer": "billing"},
  {"id": "refund", "type": "noul", "answer": 1}
]}
```

Every example is validated against `vss/data/schema.py` before training.
Invalid examples fail loudly, before a single GPU-second is spent.

## Curriculum

`src/vss/training/curriculum.py` builds stage-ordered datasets:

| stage | trains |
|---|---|
| basic | state -> single choice |
| noul | state -> probability (with negation traps) |
| score | state -> score on a declared scale |
| multi | 2-4 mixed questions per example |
| dynamic | varying option sets per example |
| hard | OOD, ambiguity, contradiction, long context, ABSTAIN targets |

## Loss

```
L = w_choice * CE(choice logits, target)
  + w_noul   * BCE(noul prob, target)
  + w_score  * Huber(expected score, target)
  + w_ordinal* soft-CE(bin distribution, triangular kernel around target)
  + w_calib  * BCE(P(correct), empirical correctness)
```

All weights live under `training.loss_weights` in the YAML config.
`ABSTAIN` choice targets select the learned abstain logit.

## Running

```bash
python scripts/prepare_data.py --out data/generated
python scripts/train.py --config configs/vss-prototype.yaml \
    --train data/generated/train.jsonl --eval data/generated/eval.jsonl
```

or the CLI: `vss train --config ... --train ... [--resume runs/.../last.pt]`

## Checkpoints and resume

Each epoch writes `last.pt`; improvements write `best.pt`; every epoch also
refreshes `final/` (safetensors + config + tokenizer) for direct loading.
Training resumes exactly (model, optimizer, epoch counter, best loss) from
`--resume runs/<run>/last.pt`. Seeds are recorded in the checkpoint; data
order is a seeded function of (seed, epoch) for reproducibility.

## RLCD (experimental)

`src/vss/training/rlcd.py` implements an independent, RLCD-inspired research
track: reward-shaped fine-tuning that reinforces correct decisions, honest
uncertainty, and useful abstention, penalizing overconfidence. Compare
against SFT and SFT+temperature on the same split and keep the simpler
method on ties. This is NOT a reproduction of any proprietary RLCD.
