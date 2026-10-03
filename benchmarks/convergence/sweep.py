"""Convergence sweep: train VSS and the plain classifier to a defensible stop.

Every run is selected on the VALIDATION split only. The calibration split is
reserved for threshold/calibration fitting and the test split is never touched
here. Run logs are written incrementally so the sweep is resumable and a
killed process (600 s shell cap) loses at most the current epoch.

Usage:
    python benchmarks/convergence/sweep.py --plan screen
    python benchmarks/convergence/sweep.py --plan final
    python benchmarks/convergence/sweep.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "multi_question_value"))

import torch  # noqa: E402

import manifest  # noqa: E402  (run identity + collision isolation)

torch.set_num_threads(6)

LOG_DIR = HERE / "runs"
CKPT_ROOT = REPO / "runs" / "convergence"

# ---------------------------------------------------------------- criteria
# Documented BEFORE looking at any new result (docs/convergence_report.md §2).
PATIENCE = 3            # epochs without eval-loss improvement > MIN_DELTA
MIN_DELTA = 5e-3        # material-improvement threshold on validation loss
MIN_EPOCHS = 4          # never stop before the LR schedule has had time to work
VAL_SLICE = 200         # validation examples scored per epoch (selection only)

# Optimizer-step budgets, not epoch budgets. The two systems have very
# different cost per step (VSS: one multi-question state per step; plain: one
# (state, question) row per step) and different splits, so "same number of
# epochs" is not a like-for-like comparison. Every system in a given budget
# gets the SAME number of optimizer steps and the same cosine schedule shape.
STEP_BUDGET_SYNTH = 400
STEP_BUDGET_REAL = 1200
# epochs are only an outer safety cap; the step budget binds first
EPOCH_CAP_SYNTH = 40
EPOCH_CAP_REAL = 16
SEEDS_FINAL = (1, 2, 3)
LR_SCREEN = (3e-4, 1e-3)   # baseline and 3x baseline


@dataclass(frozen=True)
class Run:
    system: str          # vss | plain
    dataset: str         # synthetic | clinc150 | banking77
    seed: int
    lr: float
    epochs: int
    max_steps: int | None = None
    tag: str = ""

    @property
    def name(self) -> str:
        return (f"{self.system}-{self.dataset}-s{self.seed}"
                f"-lr{self.lr:g}-st{self.max_steps}{('-' + self.tag) if self.tag else ''}")


def load_splits(dataset: str) -> tuple[list, list, list]:
    """train / validation (selection) / calibration — never the test split."""
    from vss.data.schema import TrainingExample, load_jsonl
    import dataset as mqv_dataset

    if dataset == "synthetic":
        tr = mqv_dataset.load_synthetic("train")
        va = mqv_dataset.load_synthetic("validation")
        ca = mqv_dataset.load_synthetic("calibration")
        conv = lambda exs: [TrainingExample.model_validate(e.to_training_dict())
                            for e in exs]
        return conv(tr), conv(va), conv(ca)
    d = REPO / "data" / dataset
    tr = load_jsonl(str(d / "train.jsonl"))
    va = load_jsonl(str(d / "validation.jsonl"))
    if (d / "calibration.jsonl").exists():
        ca = load_jsonl(str(d / "calibration.jsonl"))
    else:
        # reserve the tail of validation for calibration; selection uses the head
        cut = int(len(va) * 0.75)
        ca, va = va[cut:], va[:cut]
    return tr, va, ca


def build_run(run: Run):
    from vss.model.config import VSSConfig

    train_ex, val_ex, cal_ex = load_splits(run.dataset)
    if run.system == "vss":
        cfg = VSSConfig.load(str(REPO / "configs" /
                                 "vss-prototype-clinc-slot-ho-qmask.yaml"))
        from vss.model.vss_model import VSSModel
        from vss.training.trainer import Trainer
        t = cfg.training
        t.seed = run.seed
        t.lr = run.lr
        t.epochs = run.epochs
        t.max_steps = run.max_steps
        t.early_stop_patience = PATIENCE
        t.early_stop_min_delta = MIN_DELTA
        t.min_epochs = MIN_EPOCHS
        t.log_every = 10_000
        t.checkpoint_dir = str(CKPT_ROOT / run.name)
        model = VSSModel(cfg.model)
        trainer = Trainer(model, cfg, train_ex, val_ex[:VAL_SLICE])
        return cfg, model, trainer, train_ex, val_ex, cal_ex, model.num_parameters()
    # plain baseline: its own training loop, driven through the shared schedule
    import plain_classifier as pc
    from config import PlainConfig

    cfg = PlainConfig()
    cfg.seed = run.seed
    cfg.lr = run.lr
    cfg.epochs = run.epochs
    cfg.max_steps = run.max_steps
    cfg.early_stop_patience = PATIENCE
    cfg.early_stop_min_delta = MIN_DELTA
    cfg.min_epochs = MIN_EPOCHS
    # PARITY: VSS takes `header_only_choice` from its model config (true in
    # vss-prototype-clinc-slot-ho-qmask.yaml). The plain baseline used to take
    # it from the question object, which the real-data loader never sets -- so
    # the sweep silently compared VSS (header-only + full-inventory CE) against
    # plain (full option text + masked CE). That is not the same task, and it is
    # what made plain appear to collapse on CLINC150. Match it here.
    cfg.header_only_choice = True
    labels = pc.build_label_inventory(train_ex, with_abstain=True)
    labels = sorted(set(labels) | set(pc.build_label_inventory(val_ex[:VAL_SLICE],
                                                              with_abstain=False)))
    model = pc.PlainClassifier(cfg, n_labels=len(labels), with_abstain=True)
    model.attach_labels(labels)
    pc.init_scratch_(model, run.seed)
    return (cfg, model, None, train_ex, val_ex, cal_ex,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            labels)


def run_once(run: Run) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run.name}.json"
    if log_path.exists():
        d = json.loads(log_path.read_text(encoding="utf-8"))
        if d.get("done"):
            return d

    run_dir = CKPT_ROOT / run.name
    # Fail loudly rather than corrupting a checkpoint another process is writing.
    manifest.claim_run_dir(run_dir)
    t0 = time.time()
    built = build_run(run)
    cfg, model, trainer, train_ex, val_ex, cal_ex, params = built[:7]
    print(f"[{run.name}] train={len(train_ex)} val={len(val_ex)} "
          f"cal={len(cal_ex)} params={params:,}", flush=True)
    ckpt_dir = str(run_dir)
    if run.system == "vss":
        cfg.training.checkpoint_dir = ckpt_dir
    man = manifest.start_manifest(
        run_dir, run_id=run.name, system=run.system, dataset=run.dataset,
        splits={"train": len(train_ex), "validation_selection": len(val_ex[:VAL_SLICE]),
                "calibration": len(cal_ex),
                "test_used": False},
        model_cfg=(cfg.model if run.system == "vss" else
                   {"kind": "PlainClassifier", "n_labels": len(built[7])
                    if run.system == "plain" else None,
                    "hidden_size": getattr(cfg, "hidden_size", None),
                    "layers": getattr(cfg, "layers", None)}),
        training_cfg=(cfg.training if run.system == "vss" else cfg),
        params=params, repo=REPO,
        extra={"criteria": {"patience": PATIENCE, "min_delta": MIN_DELTA,
                            "min_epochs": MIN_EPOCHS, "val_slice": VAL_SLICE},
               "checkpoint_dir": ckpt_dir})
    try:
        if run.system == "vss":
            resume = str(Path(ckpt_dir) / "last.pt")
            resume = resume if Path(resume).exists() else None
            if resume:
                print(f"[{run.name}] resuming from {resume}", flush=True)
            res = trainer.fit(resume_from=resume)
            best = res["best_loss"]
        else:
            import plain_classifier as pc
            res = pc.train_plain(model, train_ex, val_ex[:VAL_SLICE], cfg, run.seed,
                                 ckpt_dir)
            best = res["best_val"]
    except Exception as exc:
        manifest.finish_manifest(run_dir, man, status="failed",
                                 error=f"{type(exc).__name__}: {exc}")
        manifest.release_run_dir(run_dir)
        raise
    rec = {
        **asdict(run),
        "params": params,
        "n_train_examples": len(train_ex),
        "n_val_examples": len(val_ex[:VAL_SLICE]),
        "n_cal_examples": len(cal_ex),
        "best_loss": best,
        "steps": res["steps"],
        "epochs_run": res.get("epochs_run", len(res["history"])),
        "stopped_early": res["stopped_early"],
        "schedule_total_steps": res["schedule_total_steps"],
        "schedule_warmup": res["schedule_warmup"],
        "wall_seconds": round(time.time() - t0, 1),
        "history": res["history"],
        "criteria": {"patience": PATIENCE, "min_delta": MIN_DELTA,
                     "min_epochs": MIN_EPOCHS, "val_slice": VAL_SLICE},
        "checkpoint_dir": ckpt_dir,
        "manifest": str(run_dir / manifest.MANIFEST_FILE),
        "config_hash": man["config_hash"],
        "git_commit": man["git"]["commit"],
        "done": True,
    }
    manifest.write_json_atomic(log_path, rec)
    manifest.finish_manifest(
        run_dir, man, status="done", result=rec,
        best_checkpoint=str(Path(ckpt_dir) / "best.pt"))
    manifest.release_run_dir(run_dir)
    print(f"[{run.name}] done best={best:.4f} epochs={rec['epochs_run']} "
          f"early={rec['stopped_early']} steps={rec['steps']} "
          f"{rec['wall_seconds']}s", flush=True)
    return rec


# ------------------------------------------------------------------- plans
def plan_screen() -> list[Run]:
    """1-seed LR screen, step-matched. Selection on validation only."""
    return [Run(system, "synthetic", 1, lr, EPOCH_CAP_SYNTH, STEP_BUDGET_SYNTH)
            for system in ("vss", "plain") for lr in LR_SCREEN]


def plan_final() -> list[Run]:
    lrs = json.loads((HERE / "selected_lrs.json").read_text(encoding="utf-8"))
    return [Run(system, "synthetic", seed, float(lrs[system]), EPOCH_CAP_SYNTH,
                STEP_BUDGET_SYNTH)
            for system in ("vss", "plain") for seed in SEEDS_FINAL]


def plan_real() -> list[Run]:
    """Real data, step-matched, single seed (CPU budget)."""
    lrs = json.loads((HERE / "selected_lrs.json").read_text(encoding="utf-8"))
    return [Run(system, ds, 13, float(lrs[system]), EPOCH_CAP_REAL, STEP_BUDGET_REAL,
                tag="s13")
            for ds in ("clinc150", "banking77") for system in ("vss", "plain")]


# Per-dataset LR screen. The one-LR-for-everything policy was a real defect:
# 3e-4 was selected on synthetic (small label space) and then applied to CLINC150,
# where the plain baseline collapsed to 0.000 accuracy. Each (system, dataset)
# pair is screened on its own validation split, with an identical budget for both
# architectures so neither is advantaged.
#
# The grid must BRACKET the optimum. The first CLINC150 screen showed plain
# improving monotonically all the way to the top of the grid
# (0.000 / 0.000 / 0.120 / 0.195 accuracy at 3e-5 / 1e-4 / 3e-4 / 1e-3), so 1e-3
# was an edge, not a peak; 3e-3 was added to close the bracket.
LR_GRID_REAL = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
STEP_BUDGET_SCREEN_REAL = 600


def plan_lr_screen_real(dataset: str) -> list[Run]:
    """LR screen for one real dataset, both architectures, same budget each."""
    return [Run(system, dataset, 13, lr, EPOCH_CAP_REAL, STEP_BUDGET_SCREEN_REAL,
                tag="screen")
            for system in ("vss", "plain") for lr in LR_GRID_REAL]


def selected_lr_per_dataset() -> dict[tuple[str, str], float]:
    """Derive each (dataset, system) LR from the screen run files.

    Read back from the result files rather than hand-written into a JSON file, so
    the number a run uses can always be traced to the trial that justified it. If
    a screen run is missing or has no history the pair is skipped rather than
    silently defaulting to a synthetic-derived LR.
    """
    import lr_table

    chosen: dict[tuple[str, str], float] = {}
    for dataset in ("banking77", "clinc150"):
        for system in ("plain", "vss"):
            best = None
            for lr in LR_GRID_REAL:
                path = lr_table.find_run(system, dataset, lr)
                if path is None:
                    continue
                sel = lr_table.selected_epoch(json.load(open(path)))
                if sel is None:
                    continue
                loss = lr_table.val_loss_of(sel)
                if best is None or loss < best[0]:
                    best = (loss, lr)
            if best is not None:
                chosen[(dataset, system)] = best[1]
    return chosen


def plan_real_final(seeds=(13,), max_steps: int = 2000, tag: str = "final"):
    """Convergence-matched real-data runs at the per-dataset screened LR.

    2000 steps rather than the screen's 600: VSS is the slower converger on
    both real datasets, so a 600-step budget under-serves it and would bias any
    comparison toward plain. The screen exists to *pick the LR*; the ranking has
    to come from a budget both systems can reach.
    """
    chosen = selected_lr_per_dataset()
    if not chosen:
        raise SystemExit("no completed screen runs; run --plan lr_screen_real first")
    return [Run(system, dataset, seed, lr, EPOCH_CAP_REAL, max_steps, tag=tag)
            for dataset in ("banking77", "clinc150")
            for system in ("plain", "vss")
            for seed in seeds
            if (dataset, system) in chosen
            for lr in (chosen[(dataset, system)],)]


PLANS = {"screen": plan_screen, "final": plan_final, "real": plan_real,
         "lr_screen_real": lambda: plan_lr_screen_real("banking77")
                               + plan_lr_screen_real("clinc150"),
         "real_final": lambda: plan_real_final()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=sorted(PLANS), default="screen")
    ap.add_argument("--deadline-min", type=float, default=None,
                    help="stop launching new runs after this many minutes")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on run name")
    ap.add_argument("--seeds", nargs="*", type=int, default=[13],
                    help="seeds for the real_final plan")
    ap.add_argument("--steps", type=int, default=2000,
                    help="step budget for the real_final plan")
    a = ap.parse_args()
    runs = (plan_real_final(seeds=tuple(a.seeds), max_steps=a.steps)
            if a.plan == "real_final" else PLANS[a.plan]())
    if a.only:
        runs = [r for r in runs if a.only in r.name]
    if a.list:
        for r in runs:
            print(r.name)
        return 0
    t_start = time.time()
    for r in runs:
        if a.deadline_min is not None and (time.time() - t_start) / 60 > a.deadline_min:
            print(f"deadline reached after {(time.time()-t_start)/60:.1f} min; "
                  f"stopping before {r.name}", flush=True)
            break
        try:
            run_once(r)
        except Exception as exc:  # keep the sweep alive; preserve the failure
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            (LOG_DIR / f"{r.name}.FAILED.txt").write_text(
                f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            print(f"[{r.name}] FAILED {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
