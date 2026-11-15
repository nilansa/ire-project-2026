# Ceous — Final Report Working Document

**Status:** WORKING DRAFT — final submission and Claude Science co-review are not yet confirmed.
**Course:** IIIT Hyderabad · IRE M26CS4.406 · 15 November 2026
**Team:** Nilanjan Sarkar; Raman Rayaprolu
**Document for joint review by:** project team, ChatGPT, and the existing Claude Science Ceous project. Do not represent this revision as already approved by Claude Science.

## Submission links

- **Canonical editable working document:** https://docs.google.com/document/d/1zJC22a8WGSWu6td3N_p48TvS_5qTCW9Q0c1QvxENwbU/edit
- **Verified two-page PDF draft:** https://drive.google.com/file/d/15vxhxV_CFGdcdWa0wmw6d1S95q0XfvMX/view
- **Existing code and research repository:** https://github.com/nilansa/ire-project-2026
- **Drive final submission folder:** https://drive.google.com/drive/folders/1XZlJ2FYkFk1r5HtXGoKSGwQd4aVAi5C0
- **Presentation video:** [VIDEO LINK TO BE ADDED — NOT YET UPLOADED]
- **Official Moodle report/video submission:** unverified; no submission action is implied by these links

## Report text — revise this, then re-export the PDF

### Abstract

Two-stage retrieval relies on a fast first-stage retriever to shortlist documents for a slower, stronger reranker. A document ranked within a broad first-stage pool (e.g., top 200), but below the actual reranking cutoff (e.g., top 20), cannot benefit from the reranker. We study how expensive reranker feedback, used selectively, can improve coverage of highly relevant or reranker-preferred documents in a fixed-size top-K shortlist, subject to feedback and inference-cost constraints. The solution space remains open: inference-time adaptation, selective reranking, and retriever post-training may be evaluated rather than assumed.

### 1. Problem statement

Given a query q, corpus D, inexpensive retriever scores s_R(q,d), and costly teacher/reranker scores s_T(q,d), let C_B(q) be a broader retriever candidate pool (e.g., B=200) and S_K(q) be the K candidates sent to the reranker, with K << B. We seek a budget-aware selection or retrieval policy that increases the representation of high-quality candidates in S_K, particularly when teacher-preferred documents lie in C_B but below the original top-K cutoff. Teacher scores are imperfect proxy feedback, not ground-truth relevance. Documents outside C_B present a separate candidate-discovery problem: reranking or reordering only C_B cannot recover them.

### 2. Research questions and outcomes

RQ1: How often and for which queries do retriever top-K rankings disagree with stronger reranker preferences and independently annotated relevance? RQ2: Can limited teacher feedback identify and promote useful documents from ranks K+1 through B, instead of simply widening the expensive reranking pool? RQ3: When is feedback most useful for online score/query adaptation, selective teacher allocation, or longer-term retriever updates? RQ4: What effectiveness gains remain after accounting for teacher calls, latency, and index/search cost?

### 3. Evaluation design

Hold query sets, corpus, first-stage baseline, teacher, candidate depth, and label definitions fixed. Compare (i) original retriever top-K, (ii) larger reranking-depth and cost-matched baselines, and (iii) tested feedback-informed shortlist policies. Within C_B, measure teacher-preference coverage: the fraction of the teacher's top-m documents among C_B included in S_K. Report relevance Recall@K using independent qrels, MRR@10 or nDCG@K as appropriate, and measured teacher-call counts and latency. Separate in-pool promotion from out-of-pool discovery; use held-out queries and controlled ablations. The teacher should not be evaluated solely on its own pseudo-labels.

### 4. Existing evidence and limitations

An earlier Ceous ANN micro-pilot compared the untouched ANCE checkpoint with teacher-trained variants on a 50,000-passage MS MARCO subset, using 500 evaluation queries. The recorded exact-search MRR@10 was 0.890777 (untouched ANCE), 0.866675 (HNSW + teacher) and 0.855858 (IVF + teacher). The two candidate-generation procedures changed which query–passage pairs received feedback, but neither trained branch improved the untouched baseline. The experiment used one seed; it does not establish a general improvement, statistical significance, or the effectiveness of the new top-K objective. Its repository record is a preliminary baseline, not a validated final result.

### 5. Relation to prior work

The Microsoft Research challenge on efficient post-training motivates data selection and expensive reward-model feedback. Reddy et al. (SIGIR 2025) directly study reranker relevance feedback at inference using query-representation updates and teacher distillation. Huo et al. (2026) study query-specific optimal reranking depth. DISCo (Nair et al., ICLR 2026) optimizes collective query coverage through successive dense index probes; this is conceptually relevant, but distinct from aligning individual retriever top-K candidates with a reranker's relevance preferences. We claim no new algorithm or performance improvement yet.

### Submission and reproducibility status

Working editable report: this Google Doc. Submission PDF: to be exported from this draft after evidence review (maximum two pages). Code/research repository: https://github.com/nilansa/ire-project-2026 (currently contains the documented research plan, pilot record and survey; a runnable full codebase has not been verified in this review). Recorded presentation: [VIDEO LINK TO BE ADDED]. Final upload / mentor sign-off: not verified. Claude Science: previous project direction reviewed from existing records, but live co-review of this revision has not been confirmed.

### Selected references

Microsoft Research, Efficient and accurate post-training of retrieval models: https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/

Reddy et al., A Large-Scale Study of Reranker Relevance Feedback at Inference, SIGIR 2025: https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference

Huo et al., Optimal Re-Ranking Depth, 2026: https://arxiv.org/abs/2611.05908

Nair et al., A Dense Subset Index for Collective Query Coverage (DISCo), ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ded182245f8e4db3963d2027174133d-Abstract-Conference.html

Existing Ceous pilot: https://github.com/nilansa/ire-project-2026/blob/main/experiments/ann-micro-pilot-20260907/RESULTS.md

## Editing and evidence rules

- Do **not** commit to a training-free or trained approach in the problem definition; compare alternatives against the same measured target.
- Use independent relevance judgments in addition to teacher pseudo-relevance.
- Distinguish reranking within top-200 from recovering material missing entirely from top-200.
- Do not claim an improvement based on the existing pilot: untouched ANCE remained stronger; one seed only.
- When adding a result, record the dataset split, configuration, exact command/log, run timestamp, and evaluated metric.
- Keep the submitted report at or below **two pages**; the present linked PDF was checked to be exactly two A4 pages.
- The GitHub default branch contains research/survey/pilot documentation; do not claim a runnable final codebase until it has been verified.
- Do not insert personal credentials or private data. The submission Drive folder is private.

**Shared task ID:** `ire-final-submission-20261115-1943` (existing IRE submission task; this cloud-native document edit did not transfer files to local Codex and does not assert a new scratch-index revision).
