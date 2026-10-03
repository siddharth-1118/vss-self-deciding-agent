"""Regression tests for the four defects found in the convergence audit.

Each test pins a defect that was live in the codebase and is now fixed, so a
regression is caught immediately:

  1. synthetic train/validation/test splits were nested prefixes of one another
     (validation == train[:300], test[:800] == train);
  2. a fixed 150-step warmup consumed 75% of a 200-step schedule, and the LR
     was exactly 0 on the first optimizer step;
  3. stable-RoPE positions were built from example 0's state length for the
     whole batch, making outputs batch-composition dependent whenever states
     have different lengths (true for both real datasets);
  4. validation loss averaged per-BATCH means, so it was not comparable
     across batch compositions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from vss.model.config import ModelConfig, TrainingConfig
from vss.model.tokenizer import VSSTokenizer
from vss.model.vss_model import VSSModel
from vss.training.trainer import Trainer, effective_warmup, lr_lambda

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "multi_question_value"))


# ------------------------------------------------------------------ defect 1
class TestSyntheticSplitsAreDisjoint:
    def test_generated_splits_do_not_share_states(self, tmp_path):
        import dataset as mqv_dataset

        sizes = {"train": 40, "validation": 20, "calibration": 20, "test": 40}
        splits = {
            name: mqv_dataset.generate_synthetic(n, seed=99, split=name, force=True, out_dir=tmp_path)
            for name, n in sizes.items()
        }
        mqv_dataset.verify_splits_disjoint(splits)  # must not raise
        fps = {k: {mqv_dataset.state_fingerprint(e) for e in v}
               for k, v in splits.items()}
        assert not (fps["train"] & fps["test"])
        assert not (fps["train"] & fps["validation"])
        assert not (fps["validation"] & fps["test"])

    def test_verify_splits_disjoint_raises_on_overlap(self, tmp_path):
        import dataset as mqv_dataset

        ex = mqv_dataset.generate_synthetic(2, seed=5, split="probe_a", force=True, out_dir=tmp_path)
        with pytest.raises(ValueError, match="leakage"):
            mqv_dataset.verify_splits_disjoint({"train": ex, "test": ex})

    def test_shipped_synthetic_split_files_are_disjoint(self):
        import dataset as mqv_dataset

        splits = {}
        for name in mqv_dataset.ALL_SPLITS:
            p = mqv_dataset.synth_dataset_path(name)
            if not p.exists():
                pytest.skip(f"{p} not generated on this machine")
            splits[name] = mqv_dataset.load_synthetic(name)
        mqv_dataset.verify_splits_disjoint(splits)


# ------------------------------------------------------------------ defect 2
class TestWarmupSchedule:
    def test_warmup_cannot_dominate_a_short_run(self):
        # 800 examples x 8 epochs / batch 32 = 200 steps
        w = effective_warmup(150, 0.10, 200)
        assert w <= 0.10 * 200 + 1
        assert w < 150

    def test_warmup_fraction_does_not_change_long_runs(self):
        # real-data schedules are ~2.3k-2.7k steps: 150 is under the 10% cap,
        # so the fix cannot be confused with a real-data hyperparameter change
        assert effective_warmup(150, 0.10, 2656) == 150
        assert effective_warmup(150, 0.10, 2269) == 150
        # plain synthetic (1600 steps) is likewise untouched
        assert effective_warmup(150, 0.10, 1600) == 150

    def test_warmup_frac_none_restores_legacy_behaviour(self):
        assert effective_warmup(150, None, 200) == 150

    def test_first_optimizer_step_has_nonzero_lr(self):
        assert lr_lambda(0, 10, 200, "cosine") > 0.0

    def test_schedule_decays_monotonically_after_warmup(self):
        w = effective_warmup(150, 0.10, 200)
        vals = [lr_lambda(s, w, 200, "cosine") for s in range(w, 200)]
        assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))

    def test_resume_pins_the_schedule_shape(self, tmp_path):
        """Extending epochs on resume must not reshape the LR curve."""
        from vss.model.config import VSSConfig

        tcfg = TrainingConfig(seed=1, epochs=1, batch_size=2,
                              checkpoint_dir=str(tmp_path), log_every=10_000)
        cfg = ModelConfig(hidden_size=32, layers=1, heads=2, kv_heads=2,
                          intermediate_size=64, vocab_size=256, hash_buckets=64,
                          option_slots=32, score_bins=8, question_masked=True)
        ex = _toy_example()
        model = VSSModel(cfg)
        tr = Trainer(model, VSSConfig(model=cfg, training=tcfg), [ex, ex, ex, ex], [ex])
        out = tr.fit()
        assert out["schedule_total_steps"] == (4 * 1) // 2
        payload = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
        assert payload["schedule_total_steps"] == out["schedule_total_steps"]
        assert payload["schedule_warmup"] == out["schedule_warmup"]


# ------------------------------------------------------------------ defect 3
class TestStablePositionsArePerExample:
    def test_positions_use_each_examples_own_state_length(self):
        cfg = ModelConfig(hidden_size=32, layers=1, heads=2, kv_heads=2,
                          intermediate_size=64, vocab_size=512, hash_buckets=128,
                          option_slots=32, score_bins=8, question_masked=True)
        model = VSSModel(cfg)
        tok = VSSTokenizer(vocab_size=cfg.vocab_size,
                           hash_buckets=cfg.hash_buckets).fit(
            ["<STATE> alpha beta gamma <QUESTION> pick one from a b c"])
        model.tokenizer = tok
        model.vss_encoder.tokenizer = tok
        short = {"message": "hi"}
        long = {"message": "hi " + " ".join(f"w{i}" for i in range(40))}
        q = [{"id": "a", "type": "choice", "options": ["x", "y"]}]
        enc = model.vss_encoder.encode_batch([short, long], [q, q])
        pos = enc["positions"]
        for b in (0, 1):
            (s, e) = enc["spans"][b][0]
            own_state_len = s
            # question block starts exactly at its own state length
            assert int(pos[b, s]) == own_state_len
            # and increases by one per token inside the block
            assert [int(v) for v in pos[b, s:e]] == list(
                range(own_state_len, own_state_len + (e - s))
            )
        # the two examples genuinely differ in state length, else the test is vacuous
        assert enc["spans"][0][0][0] != enc["spans"][1][0][0]

    def test_outputs_do_not_depend_on_batch_composition(self):
        """Same example, different batch neighbours -> same logits."""
        cfg = ModelConfig(hidden_size=32, layers=1, heads=2, kv_heads=2,
                          intermediate_size=64, vocab_size=512, hash_buckets=128,
                          option_slots=32, score_bins=8, question_masked=True,
                          dropout=0.0)
        model = VSSModel(cfg)
        model.eval()
        tok = VSSTokenizer(vocab_size=cfg.vocab_size,
                           hash_buckets=cfg.hash_buckets).fit(
            ["<STATE> alpha beta gamma <QUESTION> pick one from a b c"])
        model.tokenizer = tok
        model.vss_encoder.tokenizer = tok
        target = {"message": "short one"}
        long_state = {"message": "a much much longer state " * 6}
        q = [{"id": "a", "type": "choice", "options": ["x", "y"]}]

        def run(states):
            with torch.no_grad():
                out = model(states, [q] * len(states))
            r = out["per_example_rows"][states.index(target)][0]
            return torch.cat([r["logits"], r["abstain_logit"].reshape(1)])

        alone = run([target])
        with_long_first = run([long_state, target])
        with_long_last = run([target, long_state])
        assert torch.allclose(alone, with_long_first, atol=1e-5)
        assert torch.allclose(alone, with_long_last, atol=1e-5)


# ------------------------------------------------------------------ defect 4
class TestEvaluationIsRowWeighted:
    def test_eval_loss_is_independent_of_batch_composition(self):
        from vss.model.config import VSSConfig

        cfg = ModelConfig(hidden_size=32, layers=1, heads=2, kv_heads=2,
                          intermediate_size=64, vocab_size=512, hash_buckets=128,
                          option_slots=32, score_bins=8, question_masked=True,
                          dropout=0.0)
        tcfg = TrainingConfig(seed=1, batch_size=32, checkpoint_dir="runs/_tmp")
        model = VSSModel(cfg)
        exs = [_toy_example(i) for i in range(7)]
        tr = Trainer(model, VSSConfig(model=cfg, training=tcfg), exs, exs)
        one_batch = tr.evaluate_detailed()
        tr.tcfg.batch_size = 2
        many_batches = tr.evaluate_detailed()
        assert one_batch["loss"] == pytest.approx(many_batches["loss"], rel=1e-6)
        assert one_batch["n_rows"] == many_batches["n_rows"] == 21  # 7 examples x 3 questions

    def test_early_stopping_fires_and_is_reported(self, tmp_path):
        from vss.model.config import VSSConfig

        cfg = ModelConfig(hidden_size=32, layers=1, heads=2, kv_heads=2,
                          intermediate_size=64, vocab_size=256, hash_buckets=64,
                          option_slots=32, score_bins=8, question_masked=True)
        tcfg = TrainingConfig(seed=1, epochs=8, batch_size=2, lr=0.0,
                              warmup_steps=1, checkpoint_dir=str(tmp_path),
                              log_every=10_000, early_stop_patience=2,
                              early_stop_min_delta=1e-9, min_epochs=1)
        full = VSSConfig(model=cfg, training=tcfg)
        ex = _toy_example()
        model = VSSModel(cfg)
        tr = Trainer(model, full, [ex] * 4, [ex])
        out = tr.fit()
        # lr=0 -> loss is flat -> patience must trip well before 8 epochs
        assert out["stopped_early"] is True
        assert out["epochs_run"] < 8


def _toy_example(i: int = 0):
    from vss.data.schema import TrainingExample

    return TrainingExample.model_validate({
        "state": {"text": f"hello there number {i} " + "pad " * (i * 3)},
        "questions": [
            {"id": "a", "type": "choice", "options": ["yes", "no"],
             "answer": "yes" if i % 2 == 0 else "no"},
            {"id": "b", "type": "noul", "answer": i % 2},
            {"id": "c", "type": "score", "answer": float(i % 5),
             "min": 0.0, "max": 10.0},
        ],
    })


class TestGlobalStepBudgetTerminatesEpochLoop:
    """Audit finding 5: the epoch loop ignored the GLOBAL step budget.

    On a dataset where `max_steps` < steps-per-epoch x epochs, the budget runs
    out mid-epoch. The loop used to keep spinning, running one lr==0 batch per
    remaining epoch and evaluating after each. That wasted wall-clock and
    inflated `epochs_without_improvement`, so early stopping reported a
    "convergence" that was really just the budget expiring. Observed on
    banking77: 4 reported epochs for a 400-step budget, 3 of them no-ops.
    """

    def _cfg(self, tmp_path, **kw):
        from vss.model.config import VSSConfig

        m = ModelConfig(hidden_size=32, layers=1, heads=2, kv_heads=2,
                        intermediate_size=64, vocab_size=256, hash_buckets=64,
                        option_slots=32, score_bins=8, question_masked=True,
                        dropout=0.0)
        t = TrainingConfig(seed=1, epochs=8, batch_size=2, warmup_steps=1,
                           checkpoint_dir=str(tmp_path), log_every=10_000,
                           **kw)
        return m, VSSConfig(model=m, training=t)

    def test_epochs_run_never_exceeds_steps_taken(self, tmp_path):
        m, full = self._cfg(tmp_path, max_steps=5)
        ex = _toy_example()
        model = VSSModel(m)
        # 4 examples / batch 2 = 2 steps per epoch, budget 5 -> at most 3 epochs
        tr = Trainer(model, full, [ex] * 4, [ex])
        out = tr.fit()
        assert out["steps"] <= 5 + 1
        assert out["epochs_run"] <= 3, out["history"]

    def test_budget_exhaustion_is_not_reported_as_early_stop(self, tmp_path):
        m, full = self._cfg(tmp_path, max_steps=5, early_stop_patience=2,
                            early_stop_min_delta=1e-9, min_epochs=1)
        ex = _toy_example()
        model = VSSModel(m)
        tr = Trainer(model, full, [ex] * 4, [ex])
        out = tr.fit()
        # The budget ended the run, so this is NOT an early stop on a plateau.
        assert out["stopped_early"] is False
        assert out["steps"] >= 5

    def test_no_epoch_runs_at_zero_lr_after_budget(self, tmp_path):
        m, full = self._cfg(tmp_path, max_steps=4)
        ex = _toy_example()
        model = VSSModel(m)
        tr = Trainer(model, full, [ex] * 4, [ex])
        out = tr.fit()
        # Every reported epoch must contain real optimizer work.
        assert all(e.get("n_updates", 0) > 0 for e in out["history"]), out["history"]

    def test_resume_after_budget_exhaustion_does_not_spin(self, tmp_path):
        """A resumed run whose budget is already spent must exit, not re-epoch."""
        m, full = self._cfg(tmp_path, max_steps=4)
        ex = _toy_example()
        tr = Trainer(VSSModel(m), full, [ex] * 4, [ex])
        first = tr.fit()
        assert first["steps"] >= 4
        # simulate the harness resuming from the finished checkpoint
        tr2 = Trainer(VSSModel(m), full, [ex] * 4, [ex])
        out = tr2.fit(resume_from=str(tmp_path / "last.pt"))
        assert out["steps"] == first["steps"]
        assert out["epochs_run"] == first["epochs_run"]
        assert out["stopped_early"] is False


class TestMidEpochCheckpointPreservesHistory:
    """Audit finding 6: partial checkpoints wiped the per-epoch history.

    The mid-epoch step hook called _checkpoint() without `history=`, which
    defaults to []. Any run killed mid-epoch (the normal case on this box) and
    resumed therefore lost every previously completed epoch from its run JSON.
    Observed on banking77: the recorded history started at epoch 1 and epoch 0
    existed only in the log.
    """

    def test_partial_checkpoints_carry_completed_epochs(self, tmp_path, monkeypatch):
        from vss.model.config import VSSConfig

        m = ModelConfig(hidden_size=32, layers=1, heads=2, kv_heads=2,
                        intermediate_size=64, vocab_size=256, hash_buckets=64,
                        option_slots=32, score_bins=8, question_masked=True,
                        dropout=0.0)
        t = TrainingConfig(seed=1, epochs=3, batch_size=2, warmup_steps=1,
                           checkpoint_dir=str(tmp_path), log_every=10_000,
                           ckpt_every=1)
        full = VSSConfig(model=m, training=t)
        ex = _toy_example()
        tr = Trainer(VSSModel(m), full, [ex] * 4, [ex])

        # intercept every checkpoint write so a mid-epoch (partial) one is
        # observable -- a completed epoch immediately overwrites it on disk
        seen: list[tuple[int, bool, list]] = []
        orig = tr._checkpoint

        def spy(path, epoch, global_step, best_loss, partial=False,
                batch_index=0, history=None, epochs_without_improvement=0,
                best_accuracy=None):
            seen.append((epoch, partial, list(history or [])))
            return orig(path, epoch, global_step, best_loss, partial,
                        batch_index, history, epochs_without_improvement,
                        best_accuracy)

        monkeypatch.setattr(tr, "_checkpoint", spy)
        tr.fit()

        partials = [s for s in seen if s[1]]
        assert partials, "no mid-epoch checkpoint was written"
        # every partial checkpoint taken after epoch 0 has completed must carry
        # the epochs that finished before it
        lossy = [s for s in partials if s[0] >= 1 and not s[2]]
        assert not lossy, f"partial checkpoints dropped history: {lossy}"
