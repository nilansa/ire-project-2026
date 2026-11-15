#!/bin/bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export P=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-ann-learning-pilot-20260907/.venv/bin/python
export WORK=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/research-worktree/experiments/inference-time-ceous-20261115
export OUT=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/baseline-work
export ALIGN="$OUT/alignment-C9224B25-7E96-422D-BB87-00F18E665639.json"
export REVIEW="$OUT/ceous-review-certificate-20261115-b62f385.json"
export RUN="/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/baseline-work/ceous-approved-study-20261115T175011Z-c8f4e559-9a3a-4d49-9652-f700eec75c19"
export BUDGET="${RUN}-budget/budget.json"
printf 'START %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'STAGE start\n'
"$P" "$WORK/harness.py" start --review "$REVIEW" --alignment "$ALIGN" --out "${RUN}-budget"
printf 'STAGE dev-predict %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$P" "$WORK/harness.py" predict --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --split dev --out "${RUN}-dev-predict"
printf 'STAGE dev-score %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$P" "$WORK/harness.py" score --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --predictions "${RUN}-dev-predict" --out "${RUN}-dev-score"
printf 'STAGE selection %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$P" "$WORK/harness.py" select --budget "$BUDGET" --scored "${RUN}-dev-score" --out "${RUN}-selection"
printf 'STAGE inner-predict %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$P" "$WORK/harness.py" predict --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --split inner --selection "${RUN}-selection/selection.json" --out "${RUN}-inner-predict"
printf 'STAGE inner-score %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$P" "$WORK/harness.py" score --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --predictions "${RUN}-inner-predict" --out "${RUN}-inner-score"
printf 'COMPLETE %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
