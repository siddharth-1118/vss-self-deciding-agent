"""Run isolation / manifest guarantees (docs/training_and_reproduction.md).

A duplicate-supervisor collision corrupted a checkpoint during the convergence
study: two live processes wrote the same `last.pt`, and one result was later
reported from a checkpoint that the other process had overwritten. These tests
pin the behaviour that makes that failure impossible to repeat silently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "convergence"))

import manifest  # noqa: E402


class TestConfigHash:
    def test_hash_is_order_independent_for_dicts(self):
        a = manifest.config_hash({"lr": 3e-4, "seed": 1, "model": {"a": 1, "b": 2}})
        b = manifest.config_hash({"model": {"b": 2, "a": 1}, "seed": 1, "lr": 3e-4})
        assert a == b

    def test_hash_changes_with_any_field(self):
        base = {"lr": 3e-4, "seed": 1}
        assert manifest.config_hash(base) != manifest.config_hash({**base, "lr": 1e-3})
        assert manifest.config_hash(base) != manifest.config_hash({**base, "seed": 2})

    def test_hash_handles_dataclasses(self):
        from vss.model.config import ModelConfig

        m = ModelConfig(hidden_size=32, layers=1)
        assert isinstance(manifest.config_hash(m), str)
        assert manifest.config_hash(m) != manifest.config_hash(ModelConfig(hidden_size=64, layers=1))


class TestAtomicWrite:
    def test_write_is_atomic_and_leaves_no_tmp(self, tmp_path):
        p = tmp_path / "x.json"
        manifest.write_json_atomic(p, {"a": 1})
        assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
        assert not list(tmp_path.glob("*.tmp"))

    def test_overwrite_never_leaves_truncated_file(self, tmp_path):
        p = tmp_path / "x.json"
        manifest.write_json_atomic(p, {"v": "first"})
        manifest.write_json_atomic(p, {"v": "second", "big": [1] * 1000})
        assert json.loads(p.read_text(encoding="utf-8"))["v"] == "second"


class TestRunDirCollision:
    def test_live_owner_blocks_a_second_claim(self, tmp_path):
        """The exact collision that corrupted a checkpoint."""
        run_dir = tmp_path / "run"
        manifest.claim_run_dir(run_dir)  # this process is alive
        with pytest.raises(manifest.RunCollision):
            manifest.claim_run_dir(run_dir)

    def test_released_dir_can_be_reclaimed(self, tmp_path):
        run_dir = tmp_path / "run"
        manifest.claim_run_dir(run_dir)
        manifest.release_run_dir(run_dir)
        # after release the owner is no longer "live" for our purposes
        manifest.claim_run_dir(run_dir)  # must not raise

    def test_stale_owner_from_dead_pid_is_reclaimed(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True)
        (run_dir / manifest.OWNER_FILE).write_text(
            json.dumps({"pid": 999_999_999, "host": manifest.socket.gethostname()}),
            encoding="utf-8")
        manifest.claim_run_dir(run_dir)  # dead pid -> reclaimed, no raise

    def test_owner_records_pid_and_host(self, tmp_path):
        import os
        run_dir = tmp_path / "run"
        owner = manifest.claim_run_dir(run_dir)
        assert owner["pid"] == os.getpid()
        assert owner["host"]


class TestManifestLifecycle:
    def _start(self, tmp_path):
        return manifest.start_manifest(
            tmp_path / "run", run_id="r1", system="vss", dataset="banking77",
            splits={"train": 9079, "test_used": False}, model_cfg={"hidden_size": 256},
            training_cfg={"lr": 3e-4}, params=11164483, repo=REPO)

    def test_running_manifest_carries_provenance(self, tmp_path):
        man = self._start(tmp_path)
        assert man["status"] == "running"
        assert man["config_hash"]
        assert man["params"] == 11164483
        assert "git" in man and "commit" in man["git"]
        assert man["splits"]["test_used"] is not True
        saved = json.loads((tmp_path / "run" / manifest.MANIFEST_FILE).read_text())
        assert saved["run_id"] == "r1"

    def test_only_done_status_is_treated_as_success(self, tmp_path):
        man = self._start(tmp_path)
        manifest.finish_manifest(tmp_path / "run", man, status="failed", error="boom")
        assert manifest.load_result(tmp_path / "run") is None
        manifest.finish_manifest(tmp_path / "run", man, status="done", result={"ok": 1})
        res = manifest.load_result(tmp_path / "run")
        assert res is not None and res["result"]["ok"] == 1

    def test_interrupted_run_leaves_no_successful_result(self, tmp_path):
        """A killed run must never look like a completed experiment."""
        self._start(tmp_path)
        assert manifest.load_result(tmp_path / "run") is None

    def test_elapsed_time_recorded(self, tmp_path):
        man = self._start(tmp_path)
        done = manifest.finish_manifest(tmp_path / "run", man, status="done", result={})
        assert done["elapsed_seconds"] >= 0
        assert done["ended_at"] > done["started_at"]

    def test_corrupt_manifest_json_is_not_accepted(self, tmp_path):
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / manifest.RESULT_FILE).write_text("{not json", encoding="utf-8")
        assert manifest.load_result(tmp_path / "run") is None


class TestGitInfo:
    def test_git_info_never_raises(self, tmp_path):
        info = manifest.git_info(tmp_path)  # not a repo -> all None/None
        assert set(info) == {"commit", "dirty", "branch"}

    def test_git_info_reports_real_repo(self):
        info = manifest.git_info(REPO)
        assert info["commit"] and len(info["commit"]) == 40