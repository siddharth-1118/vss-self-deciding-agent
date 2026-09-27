# VSS data directory

Layout:

```
data/
  generated/     # synthetic data produced by scripts/prepare_data.py (gitignored)
  raw/           # downloaded public datasets (gitignored)
```

## Synthetic data (default)

```bash
python scripts/prepare_data.py --out data/generated --train-per-stage 400 --eval-per-stage 80
```

Produces `train.jsonl` / `eval.jsonl` with per-stage counts across the
curriculum: `basic, noul, score, multi, dynamic, hard`. Labels are produced
by deterministic programmatic rules — no LLM in the labeling path.

## Real public datasets

Per the dataset strategy, real datasets are converted through a normalizer
into the VSS question schema, never blindly merged. Candidate sources with
permissive licenses: CLINC150 (intent routing -> choice), banking77
(-> choice), SNIPS (-> choice/noul), GoEmotions (-> multi-label noul),
TREC (-> choice), AG News (-> choice), Jigsaw (-> noul severity scores).

Conversion pipeline:

```
source dataset -> normalizer -> VSS question schema -> JSONL
```

Validate every converted file before training:

```bash
vss validate --data data/train.jsonl
```

Only examples that pass `vss/data/schema.py` validation enter training.
