# Training and Reproduction

Everything needed to rebuild the reported numbers from a clean checkout.
Commands are copy-pasteable and were executed as written on the machine
described below.

---

## 1. Environment

| | |
|---|---|
| OS | Windows 10 (10.0.26200), x86-64 |
| Python | 3.11.9 |
| torch | 2.14.0+cpu (CPU only; no CUDA used or required) |
| Threads | `torch.set_num_threads(6)` — the box has 6 physical cores |
| Precision | fp32 throughout |

```bash
pip install -e ".[dev]"        # core + tests
pip install -e ".[serve]"      # REST API (fastapi, uvicorn)
pip install -e ".[onnx]"       # ONNX export (optional)
pip install -e ".[plots]"      # matplotlib (optional; NOT needed for any number here)
```

No matplotlib is required for training, evaluation, or any reported metric.

## 2. Tests

```bash
python -m pytest -q
```

The suite covers model behaviour, masks, training, the inference engine/API,
the convergence-audit regression tests, run isolation, and API robustness.

## 3. Data

### Synthetic (generated, ~10 s)

```bash
python scripts/prepare_data.py --out data/generated
```

Writes `train.jsonl` (2400) and `eval.jsonl` (480). This is the quick-start path
and needs no downloads.

### Real datasets (CLINC150, Banking77)

Both are prepared from the public HF datasets into VSS JSONL:

```bash
python -m vss.data.clinc150  --raw-dir data/raw --out data/clinc150
python -m vss.data.banking77 --raw-dir data/raw --out data/banking77
```

Resulting splits (verified disjoint — `train ∩ test = 0` utterances for both):

| dataset | train | validation | test | classes | options in train | options in val/test |
|---|---:|---:|---:|---:|---:|---:|
| clinc150 | 10,625 | 3,100 | 5,500 | 151 | 15 | 151 |
| banking77 | 9,079 | 924 | 3,080 | 77 | 15 | 77 |

Two properties of these splits matter when reading any result:

1. **Training uses 15 options; evaluation uses the full label set.** This is a
   deliberate dynamic-schema generalisation test, not an accident. A model is
   trained to pick a gold answer from 14 distractors and is then asked to rank
   all 151 (or 77) labels.
2. **clinc150 train and validation share 3 duplicate utterances.** Small, but
   real; treat clinc150 validation as very slightly optimistic.

## 4. Training

Every training run must own its own output directory. `scripts/train.py`
refuses to start if another live process owns the directory, and writes a
manifest recording the config hash, git commit, dirty state, split sizes, and an
explicit outcome (`done` / `failed`). Only `status: done` counts as a
successful experiment.

```bash
# quick-start / smoke model
python scripts/train.py \
    --config configs/vss-prototype.yaml \
    --train  data/generated/train.jsonl \
    --eval   data/generated/eval.jsonl \
    --out    runs/my-run
```

Outputs land in `<out>/`: `last.pt`, `best.pt`, `final/`, `manifest.json`,
`result.json`.

### Checkpoint selection

`best.pt` is selected by **validation** loss only. The test split is never read
during training, tuning, or threshold selection. `scripts/calibrate.py` fits the
abstention threshold on the calibration split.

## 5. Evaluation

```bash
python scripts/evaluate.py \
    --model runs/my-run/final \
    --data  data/generated/eval.jsonl \
    --latency --sweep-thresholds
```

Real-data evaluation:

```bash
python scripts/eval_realdata.py --help     # CLINC150 / Banking77 harnesses
python scripts/eval_ood.py --help           # OOD / abstention behaviour
```

## 6. Inference (library, CLI, REST)

```bash
python examples/basic.py                     # uses runs/prototype/final
python examples/basic.py runs/my-run/final   # or an explicit checkpoint
vss decide --model runs/my-run/final --state '{"message":"card charged twice"}' \
           --questions '[{"id":"d","type":"choice","options":["billing","sales"]}]'
vss serve --model runs/my-run/final --port 8000
```

Trained checkpoints are build artifacts and are **not committed** (`.gitignore`
excludes `runs/`), so a fresh clone must train one first — see the quickstart in
`README.md`. `examples/basic.py` exits with that exact instruction rather than
failing with a stack trace.

## 7. Convergence sweep (the tuning harness)

`benchmarks/convergence/` contains the harness that produced
`benchmarks/convergence/tables.md` and every number in
`docs/convergence_report.md`.

```bash
# per-dataset LR screen, both architectures, identical budget each
python benchmarks/convergence/sweep.py --plan lr_screen_real --list
python benchmarks/convergence/sweep.py --plan lr_screen_real --only clinc150

# render the screen from the run JSONs (accuracy at the selected checkpoint)
python benchmarks/convergence/lr_table.py

# convergence-matched final runs at the per-dataset screened LRs
python benchmarks/convergence/sweep.py --plan real_final --seeds 13 --steps 2000
python benchmarks/convergence/sweep.py --plan real_final --seeds 7 21 --steps 2000

# regenerate the tables from the run JSONs (never hand-edit them)
python benchmarks/convergence/analyze.py --out benchmarks/convergence/tables.md
```

`--plan real` still exists but applies **one LR per system to every dataset** —
the policy this audit identified as a defect. Use `--plan real_final`, which
reads each `(dataset, system)` LR back out of the screen run files, so the value
a run uses is always traceable to the trial that justified it.

### Per-dataset learning rates — required, not optional

A single global LR is a known defect: 3e-4 was selected on synthetic and applied
to CLINC150, where the plain baseline collapsed to 0.000 accuracy. The grid must
also **bracket** the optimum — the first screen put plain's best point on the top
grid edge, which meant the optimum was unknown rather than found. The current
grid (3e-5 … 3e-3) brackets all four optima as interior points.

Current selections (seed 13, validation loss only):

| dataset | plain | VSS |
|---|---|---|
| Banking77 | 3e-4 | 3e-4 |
| CLINC150 | 1e-3 | 3e-4 |

### Out-of-distribution probe

```bash
python benchmarks/convergence/ood_probe.py --model runs/prototype/final \
    --out benchmarks/convergence/results/ood_probe_synthetic.json
```

Measures whether the abstain head detects unfamiliar input or merely reports a
fixed prior. It currently shows the latter (0.103 in-distribution vs 0.106
word-scrambled vs 0.000 foreign-topic) — see `docs/model_card.md` limitation 5.

### Run isolation

`benchmarks/convergence/manifest.py` provides `claim_run_dir`, which writes an
`OWNER.json` and raises `RunCollision` if another **live** process on the same
host owns the directory. Stale owners (dead PID, different host, or explicitly
released) are reclaimed. Manifests and results are written atomically
(tmp + `os.replace`).

## 8. Long-running jobs on a contended box

Background Python processes on the development box intermittently wedge: CPU
time stops advancing while the process still exists. **A wedged process is not a
crash.** Killing it and restarting is safe and equivalent to an uninterrupted
run, because:

* `last.pt` is written atomically every `ckpt_every` (default 100) steps;
* the LR schedule shape is pinned into the checkpoint, so a resumed run follows
  the identical curve;
* the accumulated epoch history is carried forward.

`benchmarks/convergence/supervise.sh <plan> [only]` runs a sweep plan with
automatic wedge detection and resume. Detect a wedge by sampling
`Get-Process python | Select Id,CPU` twice ~60 s apart; identical values with no
output is the signature.

## 9. Reproducing the reported release numbers

```bash
# 1. tests
python -m pytest -q

# 2. per-dataset LR screen on validation only (20 runs)
python benchmarks/convergence/sweep.py --plan lr_screen_real
python benchmarks/convergence/lr_table.py

# 3. convergence-matched runs at the selected LRs, 2000 steps
python benchmarks/convergence/sweep.py --plan real_final --seeds 13 7 21 --steps 2000

# 4. regenerate every table from run JSONs
python benchmarks/convergence/analyze.py --out benchmarks/convergence/tables.md

# 5. OOD abstention probe
python benchmarks/convergence/ood_probe.py --model runs/prototype/final

# 6. confirm each run's provenance
cat runs/convergence/<run>/manifest.json
```

Each run's `manifest.json` carries `config_hash`, `git.commit`, `git.dirty`,
split sizes, `params`, `best_checkpoint`, `elapsed_seconds`, and an explicit
`status` of `done` or `failed` — an interrupted run can never read as completed.

Runs are resumable and independently restartable: re-invoke the same command and
completed runs are left alone while an interrupted one resumes from `last.pt`.