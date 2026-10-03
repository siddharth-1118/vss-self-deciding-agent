#!/bin/bash
# Single-instance supervisor for the real-data convergence runs.
#
# Two facts about this box drive the design:
#   1. background python processes WEDGE -- CPU time stops advancing while the
#      process still exists. This happens roughly every 1500 s of CPU.
#   2. a wedged run is not a crash. sweep.py resumes from runs/convergence/<name>/
#      last.pt, and the schedule shape is pinned in the checkpoint, so a
#      kill+resume is equivalent to an uninterrupted run.
#
# So: run one run at a time, watch its CPU, and relaunch on a freeze. A lockfile
# prevents the duplicate-run bug that corrupted the previous attempt (two
# copies writing to the same checkpoint dir).
set -u
cd /e/projects/vss || exit 1
LOCK=/tmp/real_sweep.pid
LOG=benchmarks/convergence/real_rerun.log
SAMPLE=40
DEADLINE=$(( $(date +%s) + ${MAX_SECONDS:-20000} ))

# `flock` does not exist in Git Bash, so use the classic PID file: create it
# exclusively, and refuse to start if a live supervisor already owns it.
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null | tr -d '\r ')
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "supervisor $old already running -- refusing to start a duplicate" >&2
    exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cpu_of() {
  powershell -NoProfile -Command \
    "(Get-Process -Id $1 -ErrorAction SilentlyContinue).CPU" 2>/dev/null \
    | tr -d '\r' | grep -E '^[0-9]+(\.[0-9]+)?$' | head -1
}

run_one() {   # $1=system $2=dataset
  local sys="$1" ds="$2" tries=0 last=-1 frozen=0
  local json="benchmarks/convergence/runs/${sys}-${ds}-s13-lr0.0003-st1200-s13.json"
  while [ "$tries" -lt 40 ]; do
    [ -f "$json" ] && { echo "[sup] $sys/$ds complete"; return 0; }
    [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[sup] deadline"; return 1; }
    tries=$((tries + 1))
    python -c "
import sys; sys.path.insert(0,'benchmarks/convergence'); sys.path.insert(0,'src')
import sweep
sweep.run_once(sweep.Run('$sys','$ds',13,3e-4,16,1200,tag='s13'))
" >> "$LOG" 2>&1 &
    local pid=$!
    echo "[sup] $sys/$ds attempt $tries pid=$pid"
    # watch this pid until it exits or wedges
    last=-1; frozen=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$SAMPLE"
      local c; c=$(cpu_of "$pid")
      if [ -z "$c" ]; then continue; fi
      if [ "$c" = "$last" ]; then
        frozen=$((frozen + 1))
        [ "$frozen" -ge 2 ] && { echo "[sup] $pid WEDGED at CPU $c -- killing, will resume"
                                taskkill //F //PID "$pid" >/dev/null 2>&1; sleep 5; break; }
      else
        frozen=0
      fi
      last="$c"
    done
    wait "$pid" 2>/dev/null
    [ -f "$json" ] && { echo "[sup] $sys/$ds complete"; return 0; }
  done
  echo "[sup] $sys/$ds gave up after $tries attempts"; return 1
}

run_one vss   banking77 && \
run_one plain banking77 && \
run_one vss   clinc150  && \
run_one plain clinc150  && \
echo ALL_REAL_DONE | tee -a "$LOG"