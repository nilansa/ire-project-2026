# OOD covariance-decomposition P5 pilot — source snapshot (2026-11-15)

Source snapshot from the ongoing local IRE P5 research at `branches/OOD_COVDECOMP_20261115/`, published on its own GitHub branch. This is a **first pilot**, not a finished result or an independently verified performance claim.

## What the source contains
- `PREREG.md` and SHA-256 record: frozen hypotheses, datasets, retrieval arms A1–A8 and failure criteria.
- `prepare.py`, `retrieve.py`, `score.py`: data preparation, retrieval variants and relevance scoring.
- `common.py`, `controls.py`, `run_phase.py`, `run_full.sh`: shared helpers, controls and phased execution.
- `preflight_controls.json`, `smoke/run_summary.json`, `DEVIATIONS.md`: small preflight/smoke and deviation evidence.

The full local input corpora, temporary retrieval outputs, 250+ MB Hugging Face cache, embeddings, checkpoints, and hidden labels are **not** in Git. Do not claim any experiment passed beyond the recorded evidence without rerunning and checking the phase-specific output.

## Relation to other branches
- Based on the current remote `main`, which already contains the 2026-11-15 inference-time Ceous protocol, alignment and harness tests.
- Kept separate from the older selector/teacher code-preservation branch.
- Research source and local runs may continue evolving after this snapshot; compare hashes with the original working directory before using it for a final report.
