#!/bin/bash
# Generic supervisor for convergence sweep plans.
#
#   usage: supervise.sh <plan> [only-substring]
#
# This box wedges background python processes: CPU time stops advancing while
# the process still exists, roughly every ~1500 s of CPU. A wedged run is not a
# crash -- sweep.py resumes from runs/convergence/<run>/last.pt and the schedule
# shape is pinned in the checkpoint, so kill+resume is equivalent to an
# uninterrupted run. This script therefore watches CPU and relaunches on a
# freeze, running ONE run at a time.
#
# Run-directory collisions are independently prevented by
# benchmarks/convergence/manifest.py (claim_run_dir raises on a live owner).
set -u
cd /e/projects/vss || exit 1
PLAN="${1:?usage: supervise.sh <plan> [only]}"
ONLY="${2:-}"
LOCK="/tmp/sweep_${PLAN}.pid"
LOG="benchmarks/convergence/${PLAN}.log"
SAMPLE=60
# A real wedge freezes CPU permanently. Writing a ~133 MB mid-epoch checkpoint
# accrues no CPU, so a short freeze is NOT proof of a wedge -- killing on it
# needlessly interrupts a healthy run. Require 4 consecutive frozen samples
# (~4 min) before intervening.
FROZEN_NEEDED=4
DEADLINE=$(( $(date +%s) + ${MAX_SECONDS:-40000} ))

if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null | tr -d '\r ')
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "supervisor $old already running -- refusing duplicate" >&2; exit 1
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cpu_of() {
  powershell -NoProfile -Command \
    "(Get-Process -Id $1 -ErrorAction SilentlyContinue).CPU" 2>/dev/null \
    | tr -d '\r' | grep -E '^[0-9]+(\.[0-9]+)?$' | head -1
}

# each Run in the plan, in order
mapfile -t RUNS < <(python benchmarks/convergence/sweep.py --plan "$PLAN" --list ${ONLY:+--only "$ONLY"} 2>/dev/null)
echo "[sup] plan=$PLAN only='${ONLY}' runs=${#RUNS[@]}"

for name in "${RUNS[@]}"; do
  json="benchmarks/convergence/runs/${name}.json"
  [ -f "$json" ] && { echo "[sup] $name already done"; continue; }
  [ "$(date +%s)" -ge "$DEADLINE" ] && { echo "[sup] deadline reached"; exit 0; }
  tries=0
  while [ $tries -lt 30 ]; do
    [ -f "$json" ] && break
    tries=$((tries + 1))
    python benchmarks/convergence/sweep.py --plan "$PLAN" --only "$name" >> "$LOG" 2>&1 &
    pid=$!
    echo "[sup] $name attempt $tries pid=$pid"
    last=-1; frozen=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$SAMPLE"
      c=$(cpu_of "$pid")
      [ -z "$c" ] && continue
      if [ "$c" = "$last" ]; then
        frozen=$((frozen + 1))
        [ "$frozen" -ge "$FROZEN_NEEDED" ] && {
          echo "[sup] $pid WEDGED at CPU $c -- killing (will resume from last.pt)"
          taskkill //F //PID "$pid" >/dev/null 2>&1; sleep 5; break; }
      else
        frozen=0
      fi
      last="$c"
    done
    wait "$pid" 2>/dev/null
  done
  [ -f "$json" ] && echo "[sup] $name COMPLETE" || echo "[sup] $name INCOMPLETE"
done
echo "SUPERVISOR_DONE_$PLAN" | tee -a "$LOG"