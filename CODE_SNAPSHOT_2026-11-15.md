# IRE code preservation — 2026-11-15

This branch is a non-destructive snapshot of the older IRE checkpoint working tree, based on `ire-mid-20261022-report`. It is intentionally **separate** from the current `main` branch and is not a claim that these methods outperform baselines.

## Preserved code
- `experiments/selector-rank-falsification/`: rank/identity invariance and insertion controls, seeded and exact probability analyses, overlap and budget bounds.
- `experiments/teacher-feedback-audit/`: exact teacher controls, cached feedback audit, ANCE model I/O and training/evaluation recipes, held-out validation guards, frozen positive controls and JSON reports.
- `experiments/evaluation-metric-audit/`: MRR rank realizability audit.

Also includes related research notes, October checkpoint report sources, and ANN/retrieval prior-work notes. Downloaded research-paper PDFs and local agent/configuration/transfer-index files were deliberately excluded.

## Checks and limitations
- The 7 selector-rank tests passed with system Python on 2026-11-15.
- Source was syntax-parsed and scanned for high-confidence credential patterns before publication.
- Do not treat these scripts or cached JSON files as independent held-out improvement evidence. The checkpoint research context is preserved, not reinterpreted.
- The original working directory at `/Users/nilan/Documents/ire-project-2026` was left on its existing branch with its working changes intact.
