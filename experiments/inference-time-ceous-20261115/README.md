# Review and reproduction

This directory is a proposed, uncommitted implementation. **No real-method predictions, scores, tuning or evaluation were run.** Root must commit the protocol, secure a non-implementer scientific review and reserve execution before resuming. The review gate is a provenance check, not a substitute for independent judgment; the implementer must never fabricate its approval file. The complete review must address the research harness and project adapter, data authority, duplicate split, metrics, controls, timing, and decision rule.

The existing alignment artifact is:

`/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/baseline-work/alignment-C9224B25-7E96-422D-BB87-00F18E665639.json`

It contains SHA-256 input hashes, all 809 candidate checks, all 505 teacher joins, exact duplicate texts and split IDs. A passing report establishes that the cached exact top 50 agrees with a raw float32 dot-product full-corpus ranking on this machine. It does not establish new retrieval quality. A fresh audit can be written to a unique path using the command below; doing so changes the report hash and requires the reviewer to approve that new hash.

```sh
export PYTHONDONTWRITEBYTECODE=1
export P=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-ann-learning-pilot-20260907/.venv/bin/python
export WORK=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/research-worktree/experiments/inference-time-ceous-20261115
export OUT=/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/baseline-work
"$P" -m unittest discover -s "$WORK" -p 'test_harness.py' -v
"$P" "$WORK/alignment.py" > "$OUT/alignment-$(uuidgen).json"
```

The following command displays the exact review bindings, including the SHA-256 of the qrels file without loading labels into the predictor. It creates no approval and must be run again if any source changes:

```sh
export ALIGN="$OUT/alignment-C9224B25-7E96-422D-BB87-00F18E665639.json"
PYTHONPATH="$WORK" "$P" -c 'import json,sys; from pathlib import Path; import alignment as a, harness as h; print(json.dumps({"code_hashes":h.code_hashes(),"alignment_sha256":a.sha(Path(sys.argv[1])),"qrels_sha256":a.sha(a.DATA / "qrels/train.tsv")},indent=2))' "$ALIGN"
```

After committing and independent approval, the reviewer or root places a distinct JSON certificate at a private path. Its exact keys are `decision` (`APPROVED` only when genuinely approved), `independent_reviewer`, `independence_statement`, `reservation_id`, `code_hashes` (the object printed above), `alignment_sha256`, and `qrels_sha256`. Include the committed protocol revision and actual review artifact in the independent review record; this program does not verify Git history. Approval must cover the actual hashes and complete protocol. Root must also determine whether all project authorization and data-use requirements are satisfied. Do not create or modify a certificate merely to bypass the gate.

Only after final independent review, protocol commit, reservation and root's explicit resume, a complete run can use these commands. Set `REVIEW` to the actual approved certificate. The new corrections-specific prefix preserves the first handoff. The start command sets a single 1,800-second wall-clock budget shared across all subsequent stages; do not restart it to evade that cap. Output directories and files are created exclusively, and a failed or incomplete one must be preserved. Prediction and score directories are complete only when their final manifest exists; JSON completion markers are created atomically.

```sh
export REVIEW="$OUT/REPLACE_WITH_ACTUAL_REVIEW_CERTIFICATE.json"
export RUN="$OUT/ceous-harness-review-corrections-20261115-v1-$(date -u +%Y%m%dT%H%M%SZ)-$(uuidgen)"
"$P" "$WORK/harness.py" start --review "$REVIEW" --alignment "$ALIGN" --out "${RUN}-budget"
export BUDGET="${RUN}-budget/budget.json"
"$P" "$WORK/harness.py" predict --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --split dev --out "${RUN}-dev-predict"
"$P" "$WORK/harness.py" score --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --predictions "${RUN}-dev-predict" --out "${RUN}-dev-score"
"$P" "$WORK/harness.py" select --budget "$BUDGET" --scored "${RUN}-dev-score" --out "${RUN}-selection"
"$P" "$WORK/harness.py" predict --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --split inner --selection "${RUN}-selection/selection.json" --out "${RUN}-inner-predict"
"$P" "$WORK/harness.py" score --review "$REVIEW" --alignment "$ALIGN" --budget "$BUDGET" --predictions "${RUN}-inner-predict" --out "${RUN}-inner-score"
```

Predictions contain row IDs, initial and final ranked lists, matched complete-step diagnostics and distinct logical and physical cost counters; only `score` opens qrels. Each score directory has JSON and CSV outputs, including the three relevance regions. Timing is warm-cache CPU update plus exact search, not teacher or end-to-end latency. Each development family winner goes to INNER even when dense is recorded as its deployment fallback. Do not interpret old exposure as a fresh holdout, or bootstrap intervals as multiplicity-adjusted claims. If any command fails, stop and preserve its evidence.
