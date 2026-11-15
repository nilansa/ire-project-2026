# Team Ceous — TA codebase and reproducibility walkthrough

**For the November 15 IRE codebase assessment · 25 October 2026**
**Task:** `ire-mid-checkpoint-20261022-001` · **Local branch:** `ire-mid-20261022-report`

This is a **read-only walkthrough of evidence and implementation**. It is not a claim that the proposed exact-retrieval teacher-positive-control experiment has run. The October 31 one-page checkpoint is in `reports/IRE_MID_CHECKPOINT_2026-10-31_ONE_PAGE.pdf` (not yet submitted). SciFact and its corrected nDCG audit are a **separate** experimental track.

## 1. Explain the research question (one minute)

Pipeline: **frozen ANCE embedding → exact/HNSW/IVF top-200 ordered candidate IDs → pre-teacher selector (20 negatives) → optional teacher scores (plus one known positive) → retriever loss/update → re-encode all 50k passages → held-out qrel evaluation**.

The question is not whether ANN search changes the top-200 IDs, nor whether it changes selected IDs. It is whether a selector **restricted to information available before expensive teacher scoring** improves **final held-out retrieval** for a fixed acquisition budget and controlled student initialization, teacher, training and evaluation. ANN Recall@200 measures agreement with exact neighbors, **not** relevance recall. Candidate overlap, selected overlap, teacher-label value and post-training MRR are different measurements.

## 2. Show the historical evidence (three minutes; do not rerun D1)

Read the authenticated private `IRE_Audit_Reproducibility_2026-10-12.zip` in the existing task scratch; its SHA-256 is `b51dea1c6dfe9c61fa7d245eeed004d9bfb347b3cf7e1a38ba1d665924426115`. The archive contains the original experiment evidence under `inputs/ire-audit-evidence.zip` and `inputs/identity-and-search-evidence.zip`; its `diagnostic/selector_diagnostic.py` is the **archived October 12** rank-selector diagnostic. The accompanying `IRE_Adversarial_Research_Review_2026-10-12.pdf` distinguishes historical experiments from diagnostics and proposals. Do **not** reimplement or rerun the archived selector falsification.

The original 300-update **no-teacher** MS MARCO/ANCE 50k diagnostic gives untouched/exact/HNSW/IVF MRR@10 of **0.890777 / 0.826063 / 0.833257 / 0.850488**. The separate 100-update **MiniLM-teacher** pilot gives untouched/HNSW/IVF **0.890777 / 0.866675 / 0.855858**; **exact teacher training was not run**. Both trained pilots underperformed untouched ANCE. Original model inference was not rerun during the October audits.

Archived D1 proves a **rank-sensitive exposure mechanism**, not improved learning: original HNSW–IVF selected-20 ID overlap **43.44%**; same-ID position perturbation **9.415%**; ID-keyed selector control **93.495%**. In the archived selected-ID disagreement, **92.39%** involved IDs present in both candidate pools. These are results from original saved candidate IDs; do not present them as new experiments.

## 3. Follow the implemented exact-positive-control contracts (four minutes)

| File | What the TA should inspect | Verified boundary |
| --- | --- | --- |
| `experiments/teacher-feedback-audit/exact_positive_control_freeze.json` | Immutable original **800-query ordered schedule**, **16,800** exact (query, passage) pairs, 500 held-out qrels and source/model identities. | Protocol frozen; **not trained**. |
| `experiments/teacher-feedback-audit/exact_execution_recipe.json` | Original 100 batches × 8 queries, token limits, AdamW, clipping and source revision. | Recipe only; not an optimizer implementation. |
| `experiments/teacher-feedback-audit/exact_control_objectives.py` | `query_loss`: positive index 0 of 21; hard-positive CE versus teacher-soft CE, with float64 teacher softmax then float32 target. | Tested on eight real cached slates and constant student test tensors; no student forward pass or update. |
| `experiments/teacher-feedback-audit/guarded_exact_control_runner.py` | `build_plan`, `batches`, `loss_for_logits`, `evaluate_exact_rankings`, `train`, `acquire_teacher_scores`. | The last two explicitly raise `PermissionError`. **Not a complete trainer.** |
| `experiments/teacher-feedback-audit/source_locked_model_io.py` | Original query/passage text and fixed order, 32/128-token inputs, unnormalized float32 full-50k inner-product evaluation. | Text/contract adapter; `reencode_or_rank` is disabled. |
| `experiments/teacher-feedback-audit/validate_heldout_exact_evaluator.py` | `evaluate_ranked_ids` validates exactly 500 frozen query rankings of 1,000 in-corpus IDs and scores all qrels. | Independently reproduced **saved historical** 500-query ranking metrics; no new rankings. |
| `experiments/teacher-feedback-audit/source_locked_train_and_evaluate.py` | Authorization-gated 100-update student trainer, fresh final-model full-50k encoding, exact ranking and qrel evaluation; compare the implementation against original source. | **Implemented, but never executed end-to-end or validated on model outputs**; CLI exposes only `--check`. |
| `experiments/teacher-feedback-audit/validated_teacher_gap_extension.py` | Read-only, create-separate-score-artifact contract for exactly the 6,518 frozen missing rows; rejects changed identities, budget, archive/revision declarations and existing cache replacement. | Fail-closed path checked; **genuine future score-file ingestion and claimed inference provenance remain unverified**. |

The **same 21 document IDs** and same schedule support two proposed controls: hard-positive CE with **zero acquired teacher labels**, and teacher-soft CE charged **16,800 labels**. Hard-positive is an informative **no-teacher baseline, not a matched-16,800-label selector policy**. Of the 16,800 exact-arm teacher pairs, **10,282 raw scores are cached** and **6,518 are unscored**. Reusing cached identical scores reduces physical inference, **not the per-policy information charge**. Teacher scoring is blocked and missing rows must not be replaced or filtered away.

**Important acquisition-budget distinction:** The two **historical** teacher-scored HNSW/IVF arms acquired **21,000 labels per arm** across 1,000 query slates, but only **16,800** labels on 800 slates entered training (4,200 acquired labels/arm were unused). Across those two arms the actual stored cache contains 32,312 unique pairs. The **proposed** frozen exact arm would acquire **16,800** and consume all 16,800; 10,282 scores are cached and 6,518 remain missing. Therefore the exact-vs-historical comparison would match **consumed training pairs, not prospective teacher-label acquisition budgets**, and cannot establish strict pre-teacher feedback efficiency. The earlier archived-data audit also records a **different, not frozen** 1,000-query exact acquisition option (21,000 pairs; 12,859 cached, 8,141 missing). Do not silently substitute that 8,141-pair gap into the frozen 6,518-row extension. An equal-acquisition selector comparison needs a separately fixed common acquisition scope for every arm.

The original recovered consumed-schedule audit (see `experiments/teacher-feedback-audit/consumed_teacher_signal_results.json`) found that 30/800 query slates per backend account for approximately **82–83% of the summed non-positive target mass**. This is about **teacher target concentration**, not proven gradient importance or held-out utility. Do not use these *post-teacher* scores to choose a supposedly *pre-teacher* selector.

## 4. Safe local demonstration (two minutes)

From `/Users/nilan/Documents/ire-project-2026`, using the original user's home directory:

```bash
HOME=/Users/nilan python3 -B experiments/teacher-feedback-audit/guarded_exact_control_runner.py --check
HOME=/Users/nilan python3 -B experiments/teacher-feedback-audit/source_locked_model_io.py --check
HOME=/Users/nilan python3 -B experiments/teacher-feedback-audit/validate_heldout_exact_evaluator.py --check
HOME=/Users/nilan python3 -B experiments/teacher-feedback-audit/source_locked_train_and_evaluate.py --check
```

Previously verified messages include `EXACT_CONTROL_RUNNER_PREFLIGHT PASS` and `SOURCE_LOCKED_MODEL_IO_PREFLIGHT PASS`; the saved-ranking evaluator check previously exited 0. These are **CPU-only provenance/contract checks on already cached evidence**, not learning experiments. Their execution on a TA machine depends on the original private archives, old pilot runtime sources and compatible Python environment; do not distribute private scratch or download models merely to make a demo succeed. The pure loss-parity unit tests require the archived Python/PyTorch environment and previously passed **4/4**; do not silently substitute a different environment.

**Never run** training, `acquire_teacher_scores`, model encoding/re-encoding, package installation or GPU/model inference as a demonstration without new authorization. The October 12 archive and frozen data/recipe have create-once semantics. The transfer index is **Codex-owned**; this walkthrough does not revise it or transfer artifacts.

## 5. Show the held-out validity limit and decision (two minutes)

The 500-query MS MARCO evaluation uses a **positive-enriched 50,000-passage diagnostic corpus**, not official full MS MARCO retrieval. Evaluation query IDs and normalized query strings were checked against the training/calibration groups; this does **not** rule out semantic or model-pretraining leakage. Actual future model comparisons require an identically initialized ANCE student, original 800-query schedule, equivalent optimizer/update budget, genuine final checkpoint re-encoding of all 50k passage texts, exhaustive inner-product top-1000 rankings, and unchanged held-out qrels. Never evaluate only the selected training slates or reuse untouched passage embeddings as though they came from the trained model.

**Experimental GO (conditional on explicit permission):** complete exactly the missing **6,518** teacher pair scores, then compare identical-slate hard-positive/no-teacher versus exact-teacher-soft on the frozen cohort and report their final held-out delta versus **untouched ANCE**. Recovered original initialization and teacher revisions are pinned; a dormant train/evaluation implementation **exists but has not been executed or validated end-to-end**. Independently attest any new teacher-logit provenance. Only after a functioning teacher positive control may one run a **separately acquisition-budget-matched** pre-teacher selector comparison, preferably with paired seeds and stronger generalization evidence.

**Current NO-GO:** new scoring, inference, GPU training, optimizer updates, extra ANN grids, or claims of superior feedback efficiency. If the exact-teacher control cannot improve on untouched ANCE, report the negative result and stop the expansion. The November 15 report/video and TA explanation must clearly separate **completed historical results**, **CPU-validated components**, and **unexecuted proposed experiments**.

## Questions a TA should be able to ask

**What exactly is the causal D1 finding?** A selector depending on absolute candidate rank can choose different document IDs even when the candidate identity set is unchanged. It does not establish that those differences help training.

**Why are the two pilots not a clean backend comparison?** The 300-update no-teacher negative-mining experiment and the separate 100-update teacher-soft experiment differ in supervision, loss and schedule; all training arms also lost to the same untouched diagnostic baseline.

**Why is the hard-positive baseline not budget-matched?** It uses the same 21 IDs and updates but zero teacher scores. A cost-matched selector comparison requires **two teacher-scored policies**, each charged for its own allowed 16,800 acquired labels.

**What makes an evaluator valid here?** Qrel-locked held-out queries, all known positives in the 50k corpus, a full-corpus exact ranking generated from the *final* trained checkpoint, and source-matched MRR/recall definitions; the diagnostic's enriched construction still limits generalization.

**Why isn't the exact control already a result?** The exposed CLI intentionally permits only static `--check`; 6,518 required logits are absent, the model-execution path has no end-to-end validation, no authorized trainer has generated updated weights, and no resulting held-out ranking has been measured.

**Do historical and exact teacher arms have equal feedback cost?** They would consume 16,800 labels each during the 100 updates, but the historical HNSW/IVF policies acquired 21,000 labels per arm before training while the frozen exact control would acquire 16,800. Equal consumption is **not** equal pre-teacher acquisition; do not present it as a feedback-efficiency comparison.
