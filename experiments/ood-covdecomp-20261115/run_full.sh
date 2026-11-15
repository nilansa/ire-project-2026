#!/bin/bash
set -euo pipefail
set -C
ROOT="/Users/nilan/Documents/ire-p5-minimal-evidence/branches/OOD_COVDECOMP_20261115"
PY="/Users/nilan/Documents/ire-p5-minimal-evidence/stage2_musique/.venv/bin/python"
RUN_START_EPOCH=""
status() {
  local phase="$1"
  local when
  when="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '# P5 run status\n\n- phase: %s\n- timestamp_utc: %s\n- launcher_pid: %s\n' "$phase" "$when" "$$" >| "$ROOT/STATUS.md"
}
if [[ -e "$ROOT/run.log" || -e "$ROOT/DONE" || -e "$ROOT/FAILED" ]]; then
  printf 'refusing to overwrite existing run artifacts\n' >&2
  exit 3
fi
exec > "$ROOT/run.log" 2>&1
on_exit() {
  rc=$?
  if [[ $rc -ne 0 ]]; then
    when="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    status "FAILED (exit $rc)"
    if [[ ! -e "$ROOT/FAILED" ]]; then printf 'exit=%s\ntimestamp_utc=%s\n' "$rc" "$when" > "$ROOT/FAILED"; fi
  fi
}
trap on_exit EXIT
RUN_START_EPOCH="$(date +%s)"
run_phase() {
  local now left
  now="$(date +%s)"
  left=$((9000 - now + RUN_START_EPOCH))
  if (( left <= 0 )); then return 124; fi
  "$PY" "$ROOT/run_phase.py" "$left" "$@"
}
status "RETRIEVE_RUNNING"
echo "run_pid=$$ phase=retrieve start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_phase "$PY" "$ROOT/retrieve.py" --main
"$PY" - <<'PY'
from pathlib import Path
import shutil
root = Path('/Users/nilan/Documents/ire-p5-minimal-evidence/branches/OOD_COVDECOMP_20261115')
with (root/'retrieval'/'decompositions.jsonl').open('rb') as src, (root/'decompositions.jsonl').open('xb') as dst:
    shutil.copyfileobj(src, dst)
PY
status "SCORE_RUNNING"
echo "run_pid=$$ phase=score start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_phase "$PY" "$ROOT/score.py"
status "CONTROLS_FINAL_RUNNING"
run_phase "$PY" "$ROOT/controls.py" --final
when="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
status "DONE"
printf 'timestamp_utc=%s\nlauncher_pid=%s\n' "$when" "$$" > "$ROOT/DONE"
echo "run_pid=$$ phase=done timestamp_utc=$when"
