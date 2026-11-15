# Earlier IRE P5 experimental code — code-only snapshot, 2026-11-15

This branch preserves locally authored experiment scripts that were not previously in the main IRE GitHub repository. It is intentionally separate from `main` and from the newer `OOD_COVDECOMP_20261115` pilot. These files were produced during prior P5 experiments; their inclusion here does not mean they were newly written today or establish any improvement claim.

## Directories
- `pilot/`: first-stage pilot and a metric test.
- `branches/B1_setsel/`, `branches/B2_suff/`, `branches/B3_alloc/`, `branches/B3R_alloc/`: alternative selection, sufficiency and allocation experiments.
- `stage1_musique/` and `stage2_musique/`: MuSiQue retrieval, frozen selection, adapter, metrics, runtime, tests and scoring.
- `stage2/`: original stage-2 H1/H2/H3/H5 research scripts and tests.
- `sufficiency/` and `wave2-20261031-0815/`: sufficiency and additional stopping, teacher-allocation, reranking experiments.

This is a **source-only preservation snapshot**, retaining the original relative paths under `experiments/p5-legacy/`. Third-party DISCo, XTR-WARP, installed dependencies, model weights, datasets, caches, generated outputs and private files were not included. Some scripts rely on external local data and are not standalone runnable without their original environment. All included Python files passed static syntax parsing; no downstream score or runtime claim is made by this archival commit.
