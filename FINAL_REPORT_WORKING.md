# Ceous — Final Report Working Document

**Status:** WORKING DRAFT — revised research narrative; a Moodle final-submission receipt exists, but this working file is not verified as identical to the submitted PDF.
**Course:** IIIT Hyderabad · IRE M26CS4.406 · 15 November 2026
**Author:** Nilanjan Sarkar
**Document for joint review by:** project team, ChatGPT, and the existing Claude Science Ceous project. Do not represent this revision as already approved by Claude Science.

## Submission links

- **Canonical editable working document:** https://docs.google.com/document/d/1zJC22a8WGSWu6td3N_p48TvS_5qTCW9Q0c1QvxENwbU/edit
- **Verified two-page PDF draft:** https://drive.google.com/file/d/1VHIo-DbdMh2sZ1Z2BIkWw-8uSwwYx2Ym/view
- **Existing code and research repository:** https://github.com/nilansa/ire-project-2026
- **Drive final submission folder:** https://drive.google.com/drive/folders/1XZlJ2FYkFk1r5HtXGoKSGwQd4aVAi5C0
- **Presentation video:** [VIDEO LINK TO BE ADDED — NOT YET UPLOADED]
- **Official Moodle report/video submission:** unverified; no submission action is implied by these links

## Report text — revise this, then re-export the PDF

### Abstract

Modern two-stage retrieval uses a relatively cheap retriever to collect candidate documents, followed by an expensive reranker that can inspect only a limited shortlist. Retraining the retriever with costly reward-model feedback is one way to improve this pipeline, but every new training iteration requires selecting informative query–document pairs and updating model parameters. Motivated by post-training research and inference-time feedback literature, we study a narrower **minimal-additional-training** alternative: keep retriever and index weights frozen, selectively use expensive teacher feedback at query time, and adapt query-specific state or shortlist selection under a measured retrieval and feedback budget. We evaluate relevance quality against direct reranking and other cost-aware controls, and distinguish candidate-pool coverage from within-pool promotion. These are inference-time experiments, not evidence of completed retriever post-training.

### 1. Problem statement

Given a query q, corpus D, inexpensive retriever scores s_R(q,d), and costly teacher/reranker scores s_T(q,d), let C_B(q) be a broader retriever candidate pool (e.g., B=200) and S_K(q) be the K candidates sent to the reranker, with K << B. We seek a budget-aware selection or retrieval policy that increases the representation of high-quality candidates in S_K, particularly when teacher-preferred documents lie in C_B but below the original top-K cutoff. Teacher scores are imperfect proxy feedback, not ground-truth relevance. Documents outside C_B present a separate candidate-discovery problem: reranking or reordering only C_B cannot recover them.

### 2. Research questions and outcomes

RQ1: How often and for which queries do retriever top-K rankings disagree with stronger reranker preferences and independently annotated relevance? RQ2: Can limited teacher feedback identify and promote useful documents from ranks K+1 through B, instead of simply widening the expensive reranking pool? RQ3: When is feedback most useful for online score/query adaptation, selective teacher allocation, or longer-term retriever updates? RQ4: What effectiveness gains remain after accounting for teacher calls, latency, and index/search cost?

### 3. Evaluation design

Hold query sets, corpus, first-stage baseline, teacher, candidate depth, and label definitions fixed. Compare (i) original retriever top-K, (ii) larger reranking-depth and cost-matched baselines, and (iii) tested feedback-informed shortlist policies. Within C_B, measure teacher-preference coverage: the fraction of the teacher's top-m documents among C_B included in S_K. Report relevance Recall@K using independent qrels, MRR@10 or nDCG@K as appropriate, and measured teacher-call counts and latency. Separate in-pool promotion from out-of-pool discovery; use held-out queries and controlled ablations. The teacher should not be evaluated solely on its own pseudo-labels.

### 4. Controlled inference-time experiments

**Audited exploratory SciFact study (15 November 2026).** The encoder and index weights remained frozen; the run used existing aligned cached dense vectors and teacher scores, with no fresh teacher or encoder inference. The team used a 405-query DEV split for family selection and a 100-query INNER partition for descriptive consistency. **All 505 queries were previously exposed**; INNER is not an untouched test set.

| Method | DEV nDCG@10 | INNER nDCG@10 | Teacher pairs | Retrieval work |
|:--|--:|--:|--:|:--|
| Dense retrieval | 0.650374 | 0.663048 | 0 | One scan |
| Direct rerank20 | 0.744338 | 0.753822 | 20 | One scan |
| TOUR-style per-query movement | 0.747601 | 0.760260 | 20 | Two scans |
| H2 local-covariance movement | 0.745127 | 0.757902 | 20 | Two scans |
| Direct rerank50 | 0.753785 | 0.791469 | 50 | Higher-feedback diagnostic |

A TOUR-style transfer exceeded rerank20 by **+0.003264 DEV nDCG@10** but required additional retrieval work and fell below the **predeclared +0.01** practical development screen. The tested H2 mechanism lost to the exact norm-matched TOUR comparator by **−0.002475 DEV** and **−0.002358 INNER**; its descriptive paired INNER bootstrap interval was approximately **[−0.01434, +0.00959]**. Neither result establishes a generalizable advantage. Across the 505 queries, 504/568 judged positive query-document pairs were within initial top50, while 57 queries had no judged positive in that pool. These observations motivate separating within-pool ranking from candidate discovery.

Teacher-cache inference cost was not incurred in this run; reported warm CPU timings exclude actual encoder and teacher inference, and cannot be called end-to-end deployment latency. For complete method budgets, preregistration, per-query outputs, and independent metrics audit, see [RESULTS.md](experiments/inference-time-ceous-20261115/RESULTS.md), [AUDIT_RESULTS.md](experiments/inference-time-ceous-20261115/AUDIT_RESULTS.md) and [PROTOCOL.md](experiments/inference-time-ceous-20261115/PROTOCOL.md).

**Separate exploratory ideas, not the primary evidence:** DISCo-inspired coverage and multi-query decomposition have been investigated in a distinct P5 effort. Small preliminary feasibility checks do not establish performance on an independently audited holdout, and must not be mixed into the table above. Earlier ANN/ANCE domain-exploration records remain archived but are intentionally excluded from this project's main explanatory storyline.

### 5. Relation to prior work

The Microsoft Research challenge on efficient post-training motivates data selection and expensive reward-model feedback; we focus instead on **per-query feedback with frozen model weights** as a more tightly scoped inference-time question. TOUR and ReFIT are relevant prior inference-time feedback methods. Reddy et al. (SIGIR 2025) directly study reranker relevance feedback at inference using query-representation updates and teacher distillation. Huo et al. (2026) study query-specific optimal reranking depth. DISCo (Nair et al., ICLR 2026) optimizes collective query coverage through successive dense index probes; this is conceptually relevant, but distinct from aligning individual retriever top-K candidates with a reranker's relevance preferences. We claim no new algorithm or performance improvement yet.

### Submission and reproducibility status

This Markdown file is a **working report narrative**. A Moodle receipt confirms a final-submission PDF was received on 15 November 2026, but this revision has **not** been checked against or substituted for that submitted PDF. Maximum permitted final report length is two readable pages. Repository: https://github.com/nilansa/ire-project-2026. Audited result files and metric-replay scripts are present; a complete fresh-device end-to-end replay is not verified. Recorded presentation and mentor sign-off have not been independently confirmed here.

### Selected references

Microsoft Research, Efficient and accurate post-training of retrieval models: https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/

Reddy et al., A Large-Scale Study of Reranker Relevance Feedback at Inference, SIGIR 2025: https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference

Huo et al., Optimal Re-Ranking Depth, 2026: https://arxiv.org/abs/2611.05908

Nair et al., A Dense Subset Index for Collective Query Coverage (DISCo), ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ded182245f8e4db3963d2027174133d-Abstract-Conference.html

## Editing and evidence rules

- Do **not** commit to a training-free or trained approach in the problem definition; compare alternatives against the same measured target.
- Use independent relevance judgments in addition to teacher pseudo-relevance.
- Distinguish reranking within top-200 from recovering material missing entirely from top-200.
- Do not claim practical improvement over cost-matched direct reranking based on the small DEV gain or treat earlier exploratory domain trials as motivation for the final problem.
- When adding a result, record the dataset split, configuration, exact command/log, run timestamp, and evaluated metric.
- Keep the submitted report at or below **two pages**; the present linked PDF was checked to be exactly two A4 pages.
- Link only audited result packages and metric-replay scripts; distinguish those from independently verified complete end-to-end inference.
- Do not insert personal credentials or private data. The submission Drive folder is private.

**Shared task ID:** `ire-final-submission-20261115-1943` (existing IRE submission task; this cloud-native document edit did not transfer files to local Codex and does not assert a new scratch-index revision).
