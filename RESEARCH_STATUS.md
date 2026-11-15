# Ceous inference research — coordination status

**Task:** `ire-final-submission-20261115-1943`
**Scope:** initial coordination only; no new experiments have been run for this tracker. The user prepares the report and video.
**Last remote read:** `b59a5bac89867d3213036ad0ac6b0ad3d171725f`, 2026-11-15T17:10:39Z (no tracker existed). Later readers: record the commit and UTC time actually read in the inbox/read log below.
**Live ownership reconciliation:** pending root's independent audit of active Claude Science work. No proposed external assignment below implies acceptance.

## Coordination and user inbox

The root Astra orchestrates and designs H5. Codex is the **single writer** for this central status, integrating user notes and confirmed worker evidence; workers append run records in existing `experiments/` paths and do not overwrite others' files. Before any launch and after each meaningful run, agents fetch and read the latest remote `main` and this file, record commit/time, inspect reservations and prior runs, and report changes to the writer. Preserve all freeform user notes verbatim, including notes without an ID; do not replace or silently delete them.

Inbox entries have stable IDs, status `NEW`, `ACKNOWLEDGED`, or `DONE`, author, UTC timestamp, and freeform text. The writer changes status only upon actual acknowledgement or completion and preserves history. User notes can be appended directly; the writer must reconcile concurrent edits from `main` before publishing.

| ID | Status | Author | UTC | Note |
|---|---|---|---|---|
| INBOX-001 | NEW | Codex | 2026-11-15T17:10:39Z | Root: reconcile active Claude Science ownership and existing runs before assigning or launching H4/H6; assess the next independent review and bounded experiments. |

**Freeform user notes:**

<!-- Add notes here; preserve this section and its contents. -->

**Read log:** initial writer read `b59a5bac89867d3213036ad0ac6b0ad3d171725f` at 2026-11-15T17:10:39Z. Append reader, commit, UTC timestamp and relevant reservation/inbox IDs on subsequent reads.

## Question and branches

Can per-query inference-time adaptation improve retrieval quality at a fair total retrieval and teacher-scoring cost while keeping encoder and index weights frozen? Per-query state is optimized: call this **no per-domain retraining**, not zero optimization. Branch statuses are exactly `proposed`, `running`, `completed`, or `rejected`; a completed analysis is not a completed experiment. No branch is running under this coordination task.

| ID | Status | Mechanism and prediction | Required comparison / owner |
|---|---|---|---|
| H1 | proposed | Budgeted teacher-guided query movement may discover relevant documents beyond the initial pool. | TOUR-style soft-label query optimization is a required baseline; Codex implementation candidate, ownership/run not yet reserved. |
| H2 | proposed | Shrinkage/local-covariance preconditioned movement may use feedback gradients more efficiently than isotropic movement. | Match gradients and budgets to H1; `qᵀ A_q d = (A_qᵀ q)ᵀ d`, so this bilinear form is not more expressive than a transformed query. Terra mathematics review: completed analysis only; Codex implementation candidate. |
| H3 | proposed | Nonlinear residual scoring may promote relevant documents within a fixed candidate pool. | Same-pool reranking and matched cost; it cannot discover outside-pool documents without a search mechanism. Sol reviewer proposed, not acknowledged. |
| H4 | proposed | Adaptive allocation may spend retrieval/teacher effort on queries with greater potential benefit. | Static schedule and equal-total-cost controls; Claude Science ownership proposed, pending explicit reconciliation and ACK. |
| H5 | proposed | Disagreement, leverage, or exploration can select more informative feedback than simply scoring the highest-ranked documents. | Top-score and random controls, independent qrels, and explicit selection-bias analysis. Root Astra designs; independent review required. |
| H6 | proposed | Multiple retrieval directions may cover relevant documents missed by a single query. | Deep single-query and random-direction controls; count duplicate retrieved slots and every probe. Claude Science ownership proposed, pending explicit reconciliation and ACK. |

For each branch, preregister the counterhypothesis (matched extra retrieval or reranking alone explains any apparent gain), causal prediction, decisive development screen, comparator, fixed cost cap, metrics, threshold and rejection rule before running. Record distinct mechanisms and an independent scientific referee for every new question and material result. The user has authorized this documentation and coordination, **not experiments**; separate execution authority and available compute are required.

## Reservation and ownership gate

Before launch, the root reconciles live ownership and existing work; a proposed worker must explicitly ACK the exact scope. The single writer then publishes a unique reservation after checking the latest remote tracker and existing append-only ledgers. No ACK, reservation, independent review, preregistration or execution authority means no launch. An ACK alone is not human authorization. Concurrent claims for overlapping data, hypothesis, method, seeds and budgets are resolved before either run starts; record supersession rather than silently deleting a reservation. Do not rerun old failed synthetic controls.

Reservation record schema: `reservation_id`, `branch`, `status` (`proposed`/`running`/`completed`/`rejected`), `owner`, `owner_ack_evidence`, `root_reconciliation_evidence`, `purpose`, `baseline_and_candidate`, `dataset/split`, `seeds`, `budget_cap`, `preregistration_hash`, `independent_review`, `execution_authority`, `code/config/data/checkpoint hashes`, `planned_start_utc`, `last_remote_commit_read`, `last_read_utc`, `run_ids`, `ledger_path`, `outcome_or_blocker`, and `updated_utc`. No reservations yet.

Fresh work belongs in unique scoped subfolders under existing `experiments/` and data locations, with unique run IDs and append-only manifests: exact commands, software versions, code/config/data/split/checkpoint hashes, seeds, hardware, raw per-query outputs, controls and decisions. Preserve negative and invalidated results; corrected runs get new IDs. Never overwrite another agent's artifacts.

## Budget, baselines and evaluation contract

- Freeze encoder and index weights. Fix final `K` and initial pool `B` in each comparison. Account for **all** retrieved slots including duplicates, unique documents, ANN probes/search work, and every teacher query-document score including final reranking; measure CPU, GPU and wall cost. Equalize or report the full budget vector, rather than matching just one component.
- Baseline suite: untouched retriever, dense Rocchio PRF, teacher-weighted Rocchio, TOUR-style soft-label query optimization, direct same-pool reranking, and deeper retrieval references. After the audit select **2–3 executable baselines** appropriate to each screen; do not claim an unaudited implementation is ready.
- Use independent qrels exclusively for evaluation, never query movement, hyperparameter selection, or stopping. Keep development and blind holdout groups separated; declare data authority, splits, seed list and hashes before confirmatory runs. Teacher scores on an extra diagnostic pool are analysis overhead, separate from the deployed budget and disclosed in full.
- Report Recall@K, nDCG@10 and MRR@10, judged coverage, relevant documents recovered outside the original pool, in-pool promotion, paired per-query differences and seeded paired bootstrap intervals. Report measured end-to-end latency p50/p95 and component timings; include hardware and full costs. Missing qrels do not prove a document irrelevant.
- With nested corpus `D ⊇ U ⊇ C ⊇ L`, the same judged objective `J`, and `J*(S)` the best attainable result from set `S`, decompose `J*(D) − J(L) = [J*(D) − J*(U)] + [J*(U) − J*(C)] + [J*(C) − J(L)]`: outside-pool, candidate-selection and ranking gaps. These are judgment-dependent diagnostic bounds, not proof about unjudged documents.
- Predeclare a development-screen resource cap and meaningful gain threshold against the strongest cost-matched baseline. Stop or park weak branches when the screen budget is exhausted without gain; keep negative results. Any leakage or failed required control invalidates claims until corrected and independently reevaluated. Preregister planted-truth positive and at least two distinct negative controls, an independent referee, and explicit CONTINUE / PIVOT / PARK / STOP decisions with re-entry criteria.

## Historical evidence and limits

- The [older ANCE teacher-training pilot](experiments/ann-micro-pilot-20260907/RESULTS.md) reports untouched ANCE MRR@10 **0.890777**, HNSW + teacher **0.866675**, and IVF + teacher **0.855858**, one seed. Both trained branches were worse than the untouched checkpoint; these are historical results, not new verified improvements. The earlier harness adapter separately says source-locked same-path untouched inference still needed reconciliation; don't conflate that open audit with the historical reported number.
- Earlier missing-artifact blockers were later superseded by recovered archives. Cached frozen ANCE vectors `de.npy`, `ee.npy`, `ce.npy`, `te.npy` are in `/Users/nilan/.codex/scratch/cross-surface/IRE/ann-micro-pilot-20260907/shared/runtime/cache/emb_50000_20260907/`; `slice_50000_20260907.json` is in the parent cache directory. The historical `teacher_scores.csv` has 32,312 training query-document pairs. The 50k positive-enriched slice is diagnostic, not full MS MARCO. Audits are still checking which recovered artifacts are reusable; historical recovery does not establish reproducibility of a new experiment.
- The later HotpotQA pilot record at `/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/results.md` already tested dense, BM25, MMR, DISCo approximations, MDR-style, random and oracle. The holdout has 324 bridge and 76 comparison questions. At bridge `k=2`, G2 failed: +0.028, paired 95% CI [−0.049, +0.083] versus +0.08 target; G3 also failed. The approximation is not official DISCo. Check that record before proposing related work; do not duplicate it.
- Historical Nov 3 P0 Terra referee: `INSUFFICIENT_EVIDENCE` because the MedCPT N1 control failed; the P0-prime design referee returned `PARK / INSUFFICIENT_EVIDENCE` and rejected the frozen design. These are historical reports, not a statement of live Claude Science ownership or current activity. Root's live audit is pending.

## Prior work and claim boundaries

- [TOUR](https://aclanthology.org/2023.findings-acl.354/) and [Reddy et al., SIGIR 2025](https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference) are essential inference-time feedback baselines and context.
- [DISCo](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ded182245f8e4db3963d2027174133d-Abstract-Conference.html) addresses collective coverage of contextual query vectors by contextual item vectors, with dense index probes; it is not n-gram coverage. The supplied `2409.17424` PDF is Big ANN, not DISCo.
- Close preprints [arXiv:2605.12487](https://arxiv.org/abs/2605.12487) and [arXiv:2608.12569](https://arxiv.org/abs/2608.12569) require detailed verification before specific comparative claims. The root's literature review is completed; no novelty claim follows from this tracker.
