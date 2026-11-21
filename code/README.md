# Code organization and reproducibility

The source files here come from **completed Ceous experimental efforts**; they have been grouped rather than rewritten. The historical tracked main branch at `archive/pre-final-cleanup-20261121` preserves the original code provenance, output manifests and scientific reviews. These source-only copies do not include full model or dataset caches.

- `fiqa/`: query-wise feedback objectives, selector arms, evaluation metrics and tests. `run_fiqa_5fold_crossfit.py` is a **separate task-wise** experiment and must not be pooled with FiQA-200 query-wise results. The `modal_*.py` files require a configured Modal deployment. Start by inspecting `fit_querywise.py`, `selection_arms.py`, `metrics.py` and `evaluate_run.py`.
- `hotpotqa/local` and `hotpotqa/modal`: H8 HotpotQA-HardNegatives controls, recorded split and server-side GPU stages. Read the archived H8 preregistration and results record before running anything; **do not** treat post-decision POST600 as an untouched holdout.
- `scifact/`: original source for the teacher-guided fixed-index evaluation harness. `replay_metrics.py` requires the large result inputs retained in the archive branch; the source-only branch cannot run it without these files. The frozen benchmark and metric protocol are archived.
- `scidocs/`: SCIDOCS ReFIT adaptation and baseline diagnostics; some scripts reference local checkpoint/data paths and require relinking to your environment. This study was not an exact reproduction of the originally published ReFIT improvement.
- `ann-pilot/`: minimal candidate exposure diagnostic.

## Lightweight checks

```bash
python3 -m compileall -q code
python3 -m unittest discover -s code/ann-pilot -p 'test_*.py'
```

Code includes GPU workflows and data-dependent scripts. Do not launch those just to verify the checkout. Model version pins, split IDs, raw scores and full teacher-score acquisition logs are documented in the historical archive and final two-page report. Keep any credentials and local model caches out of Git.
