"""Run identity, provenance and isolation for convergence experiments.

Every experiment must be attributable to (a) a unique directory, (b) a config
hash, (c) a git revision and dirty state, (d) the dataset/split, and (e) the
outcome. Two processes writing to the same run directory silently corrupt each
other's checkpoints -- that happened during this study (two supervisors writing
`vss-banking77-.../last.pt`), so isolation is enforced here rather than left to
the caller's discipline.

Design rules:
  * A run directory is claimed by writing ``OWNER.json`` containing the creating
    PID and host. A second process that finds an owner whose PID is alive on the
    same host **fails loudly** instead of proceeding.
  * Manifests and results are written atomically (tmp + replace) so a killed
    process cannot leave a truncated file that later parses as valid.
  * ``status`` is explicit: only ``"done"`` records count as successful. A run
    interrupted mid-training leaves ``status: "running"`` and is never mistaken
    for a completed experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

OWNER_FILE = "OWNER.json"
MANIFEST_FILE = "manifest.json"
RESULT_FILE = "result.json"


# ----------------------------------------------------------------- hashing
def config_hash(obj: Any) -> str:
    """Stable hash of any JSON-serialisable config (dataclasses included)."""
    payload = _jsonable(obj)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


# ------------------------------------------------------------------- git
def git_info(repo: Path) -> dict[str, Any]:
    """Commit + dirty state. Never raises: a missing git just means unknown."""
    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=str(repo),
                                 capture_output=True, text=True, timeout=20)
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    status = _run("status", "--porcelain")
    return {
        "commit": _run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


# ------------------------------------------------------------- atomic I/O
def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via tmp+replace. A partial write can never be observed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- isolation
def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                                 capture_output=True, text=True, timeout=20)
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except Exception:
        return False


class RunCollision(RuntimeError):
    """Raised when another live process already owns this run directory."""


def claim_run_dir(run_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Claim `run_dir` exclusively. Raises `RunCollision` if it is taken.

    A stale owner (dead PID, or a different host) is reclaimed; a live owner on
    this host is a hard error.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    owner_path = run_dir / OWNER_FILE
    owner = {"pid": os.getpid(), "host": socket.gethostname(),
             "claimed_at": time.time()}

    if owner_path.exists() and not force:
        try:
            prev = json.loads(owner_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
        same_host = prev.get("host") == owner["host"]
        # An explicitly released owner is reclaimable even while that pid is
        # still alive: otherwise a supervisor that finishes a run and re-enters
        # cannot reclaim its own directory.
        alive = (same_host and not prev.get("released")
                 and _pid_alive(int(prev.get("pid", -1))))
        if alive:
            raise RunCollision(
                f"run directory {run_dir} is owned by live pid {prev.get('pid')} "
                f"on {prev.get('host')}; refusing to write concurrently"
            )
    write_json_atomic(owner_path, owner)
    return owner


def release_run_dir(run_dir: Path) -> None:
    """Mark the owner as finished so a later run may reclaim the directory."""
    path = run_dir / OWNER_FILE
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    data["released_at"] = time.time()
    data["released"] = True
    write_json_atomic(path, data)


# --------------------------------------------------------------- manifest
def start_manifest(run_dir: Path, *, run_id: str, system: str, dataset: str,
                   splits: dict[str, Any], model_cfg: Any, training_cfg: Any,
                   params: int, repo: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write manifest.json describing a run that is about to start."""
    man = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": "running",
        "started_at": time.time(),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "system": system,
        "dataset": dataset,
        "splits": splits,
        "model_config": _jsonable(model_cfg),
        "training_config": _jsonable(training_cfg),
        "config_hash": config_hash({"m": _jsonable(model_cfg), "t": _jsonable(training_cfg),
                                    "sys": system, "ds": dataset, "sp": _jsonable(splits)}),
        "params": params,
        "git": git_info(repo),
        **(extra or {}),
    }
    write_json_atomic(run_dir / MANIFEST_FILE, man)
    return man


def finish_manifest(run_dir: Path, man: dict[str, Any], *, status: str,
                    result: Any = None, best_checkpoint: str | None = None,
                    error: str | None = None) -> dict[str, Any]:
    """Mark a run finished. `status` must be an explicit outcome word.

    Only ``status == "done"`` marks a successful experiment; anything else
    (``failed``, ``aborted``, ``collided``) is recorded as such so an interrupted
    run can never be read as a completed one.
    """
    man = dict(man)
    man["status"] = status
    man["ended_at"] = time.time()
    man["ended_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    man["elapsed_seconds"] = round(man["ended_at"] - man["started_at"], 2)
    if result is not None:
        man["result"] = _jsonable(result)
    if best_checkpoint:
        man["best_checkpoint"] = best_checkpoint
    if error:
        man["error"] = error
    write_json_atomic(run_dir / MANIFEST_FILE, man)
    if result is not None:
        write_json_atomic(run_dir / RESULT_FILE,
                          {"status": status, "run_id": man.get("run_id"),
                           "config_hash": man.get("config_hash"), "result": _jsonable(result)})
    return man


def load_result(run_dir: Path) -> dict[str, Any] | None:
    """Load a completed result, or None if absent / not `done`."""
    p = run_dir / RESULT_FILE
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    return d if d.get("status") == "done" else None