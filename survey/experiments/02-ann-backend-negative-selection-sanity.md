# ANN backend → selected-negative supervision sanity experiment

**Status:** completed pilot; one seed; motivation/sanity evidence only.

## Question

If two ANN/search backends have similar agreement with exact top-200 retrieval, do they still expose different passage identities to a fixed training-data selection rule, and can those different pools lead to different trained retrievers?

This pilot is motivated by the Microsoft Research challenge **Efficient and accurate post-training of retrieval models**, especially its **Data Selection** question: which query-document pairs should receive expensive reward/relevance feedback during post-training?

## Setup

- **Dataset:** 50,000-passage MS MARCO subset.
- **Queries:** 1,000 train, 200 ANN-calibration, 500 disjoint evaluation queries.
- **Retriever checkpoint:** `sentence-transformers/msmarco-roberta-base-ance-firstp`.
- **Similarity:** raw inner product; no embedding normalization.
- **Candidate depth:** top-200.
- **Search conditions:** exact, HNSW, IVF-Flat.
- **Selection budget:** 20 passages/query using the same deterministic query-specific permutation of candidate rank positions; all known positive passage IDs excluded.
- **Training:** fixed-snapshot ANCE-style pairwise training using one known qrel positive and one selected passage at a time; 300 optimizer updates/branch, batch size 8, same seed/schedule/optimizer and identical initial model state.
- **Important:** this completed pilot did **not** use a cross-encoder/reward model. It therefore tests ANN-induced candidate/supervision identity differences before the intended teacher-scoring stage.

## ANN agreement with exact top-200

`Recall@200` here means overlap with the **exact raw-inner-product top-200 passage IDs**, not recall of human relevance labels.

| Backend | Calibration Recall@200 | Train-query Recall@200 | Operating point |
|---|---:|---:|---|
| HNSW | 0.955975 | 0.958155 | M=32, efConstruction=100, efSearch=256 |
| IVF-Flat | 0.962250 | 0.967460 | nlist=256, nprobe=96 |

The recalls are similar, though not identical.

## Candidate and selected-pool differences

- HNSW vs IVF top-200 candidate **Jaccard**: **0.9173598779**.
- HNSW vs IVF selected-20 **shared fraction**: **0.4344**.
- This selected-20 number is **not Jaccard and not recall**. It is the average intersection size divided by 20.
- Across 1,000 training queries, the two selected pools had 8,688 shared query-passage entries in total: **8.688 common passages/query out of 20**.
- Exact/HNSW selected-20 shared fraction: **0.44885**.
- Exact/IVF selected-20 shared fraction: **0.48055**.

Thus, under this fixed rank-sensitive selector, relatively small differences in the top-200 candidate lists translated into substantially larger differences in which passage identities entered the selected supervision pool.

## Initial supervision difficulty

Despite the identity differences, coarse initial difficulty statistics were nearly the same:

| Pool | Mean pairwise loss | Mean negative score | Mean positive − negative margin |
|---|---:|---:|---:|
| Exact | 0.03579266 | 706.54907 | 6.48467 |
| HNSW | 0.03562985 | 706.53122 | 6.50251 |
| IVF | 0.03541438 | 706.53128 | 6.50246 |

This does **not** establish that the different passage IDs are different semantic kinds of negatives.

## Final exact-search evaluation after 300 updates

All final models were re-evaluated with the same exact raw-inner-product search over the same 50k corpus.

| Model | MRR@10 | Recall@100 | Recall@1000 |
|---|---:|---:|---:|
| Untouched initial checkpoint | 0.8907769841 | 0.9930 | 1.0000 |
| Exact-mined training | 0.8260634921 | 0.9860 | 0.9955 |
| HNSW-mined training | 0.8332571429 | 0.9870 | 0.9960 |
| IVF-mined training | 0.8504880952 | 0.9885 | 0.9975 |

All three trained branches degraded relative to the already-strong untouched checkpoint. Therefore this pilot is **not evidence that any ANN backend is better**, nor that post-training improved retrieval.

## What this pilot supports

A defensible motivation statement is:

> In this 50k-passage pilot, HNSW and IVF-Flat had similar approximately 96% agreement with exact top-200 retrieval, yet under the same 20-passage rank-based selection rule their selected pools shared only 43.44% of passage identities. Aggregate ANN recall therefore did not characterize which query-passage pairs entered the training supervision.

This motivates studying the intended pipeline:

`ANN candidate generation → fixed limited-budget selection → relevance/reward teacher scoring → retriever post-training`

The next scientific question is whether different selected identities receive meaningfully different teacher feedback and whether that difference affects the final retriever.

## Limits / non-claims

- One seed only; no repeat-run variance estimate.
- MPS deterministic algorithms were not enforced.
- The 50k corpus was constructed to include known positives for train/calibration/evaluation queries and is much easier than full MS MARCO; the untouched baseline already had Recall@100 = 0.993.
- The 20-passage result depends on the fixed rank-sensitive selection rule; it is not a universal property of HNSW versus IVF.
- The pilot used qrel positives plus selected unlabelled passages, not a cross-encoder teacher.
- Different passage IDs can still be semantically similar; no semantic negative taxonomy was measured.
- A tiny NumPy/FAISS near-tie discrepancy remained at exhaustive IVF (`nprobe=256` gave 0.99995 agreement rather than exactly 1.0), so do not overclaim pure ANN-family causality.
- These results do not establish statistical significance, novelty, or general IVF superiority.
