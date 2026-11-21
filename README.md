# Ceous — Complementary Feedback Selection for Test-Time Dense Retrieval

**Information Retrieval and Extraction (M26CS4.406), IIIT Hyderabad**

**Project:** Ceous · **Authors:** Nilanjan Sarkar and Raman Rayaprolu

Ceous studies how a frozen dense retriever can use a small budget of reranker feedback at inference time. A query-specific correction is fitted from ten black-box teacher scores, and the corrected query searches the full corpus again. We compare dense-top-ten feedback with complementary selection inspired by DISCo, as well as MMR, BM25 and other controls.

## Submission

- **[Ceous Final Report (PDF)](report/Ceous-Final-Report.pdf)** — version included in this GitHub repository
- **[Final report folder in Nilanjan Sarkar's Drive](https://drive.google.com/drive/folders/1qpqFbFM1dtip521oe314WEknWLM1lyKo)**
- **[Recorded presentation video](https://drive.google.com/file/d/1kf9SSlwgiJVrJ_ITRr9Xv2ngINr7PYrC/view)** and [video folder](https://drive.google.com/drive/folders/1w5bleV1Xb0ELg6PQO1xmRNsaCEnuXnHD)
- **[Original experiment code and evidence archive](https://github.com/nilansa/ire-project-2026/tree/archive/pre-final-cleanup-20261121)** — prior tracked research, complete logs, review and evaluation evidence

## Results at a glance

| Dataset / comparison | Dense-feedback TTT | Coverage-feedback TTT | Difference in nDCG@10 |
|---|---:|---:|---:|
| FiQA-200 (200 exploratory queries) | 0.5060 | 0.4807 | −0.0253 |
| HotpotQA-HN DEV400 (preregistered) | 0.7195 | 0.7298 | +0.0103 |
| HotpotQA-HN POST600 (post-decision analysis) | 0.7447 | 0.7628 | +0.0181 |

The preregistered HotpotQA decision was **PARK**: the confidence interval for the development-set difference included zero. The FiQA sample had been previously exposed. See the report and archived result records for split definitions and paired uncertainty intervals. The DISCo-inspired policy selects from an existing dense top-50, rather than reproducing DISCo's original token ANN index.

## Code layout

| Directory | Contents |
|---|---|
| [`code/fiqa`](code/fiqa) | Frozen Qwen/TTT feedback selection, FiQA query-wise evaluation, baselines, tests and task-wise cross-fit scripts |
| [`code/hotpotqa`](code/hotpotqa) | HotpotQA-HardNegatives DISCo-style selection and TTT evaluation, plus Modal GPU orchestration |
| [`code/scifact`](code/scifact) | Frozen-feedback baseline harness, alignment checks, independent metric replay and tests |
| [`code/scidocs`](code/scidocs) | SCIDOCS ReFIT reproduction scripts and CPU/MPS diagnostics |
| [`code/ann-pilot`](code/ann-pilot) | Early ANN/candidate headroom diagnostic and tests |
| [`code/README.md`](code/README.md) | Execution, dependencies, provenance and limitations |

**Scope:** This branch keeps the final source code and primary report easy to navigate. It intentionally excludes cached embeddings, model weights, dense retrieval indexes, temporary output files and the extensive older experiment directories; historical material is retained on the linked `archive/pre-final-cleanup-20261121` branch. Not every experiment can be rerun using only this source checkout: several require explicitly pinned model snapshots, source datasets and cached teacher scores. No new experiment has been executed by this cleanup.
