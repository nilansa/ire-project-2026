# ANN Micro-Pilot (2026-09-07)

## Question
Does changing the ANN candidate generator alter the retriever training signal and final retriever?

## Pipeline
ANN candidates -> fixed 20 passage selection -> cross-encoder scoring -> soft-label retriever training -> exact evaluation.

## Setup
- Corpus: 50k MS MARCO passages
- Train/calibration/eval queries: 1000/200/500
- ANN: HNSW vs IVF-Flat
- Teacher: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Student: ANCE checkpoint
- Teacher scores: 32,312 unique query-passage pairs
- Training: 100 updates, batch 8 queries, soft-label cross entropy, LR 2e-5
- Evaluation: exact raw inner-product search

## What "Untouched ANCE" means

`Untouched ANCE` is the original ANCE retriever checkpoint **before any training in this experiment**. It is the control baseline.

It was **not trained again** and did **not** receive HNSW candidates, IVF candidates, or cross-encoder teacher supervision. We simply evaluated that original checkpoint with the same exact-search evaluation used for the two trained branches.

The comparison is therefore:

- **Untouched ANCE** = original starting checkpoint -> no additional training -> exact evaluation.
- **HNSW + teacher** = same original starting checkpoint -> HNSW-selected passages -> cross-encoder soft supervision -> 100 training updates -> exact evaluation.
- **IVF + teacher** = same original starting checkpoint -> IVF-selected passages -> cross-encoder soft supervision -> 100 training updates -> exact evaluation.

The HNSW and IVF students were verified to start from identical copies of the same original ANCE parameters.

## ANN Recall@200 (not relevance recall)

ANN Recall@200 here means the fraction of the **exact raw-inner-product top-200 passage IDs** recovered by the approximate ANN search. It is not recall against human/qrel relevance labels.

Training-query ANN Recall@200:
- HNSW: 95.8155%
- IVF: 96.746%

IVF therefore had slightly higher agreement with exact nearest-neighbor search on this metric.

## Final Retrieval Metrics

These metrics are computed after **exact search** on the same 500 evaluation queries against the same 50k-passage corpus. `Recall@100` and `Recall@1000` below are relevance recall against known relevant passages, not ANN Recall@200.

| Retriever | MRR@10 | Relevance Recall@100 | Relevance Recall@1000 |
|---|---:|---:|---:|
| Untouched ANCE | 0.890777 | 0.993 | 1.000 |
| HNSW + teacher | 0.866675 | 0.993 | 1.000 |
| IVF + teacher | 0.855858 | 0.991 | 1.000 |

So, in this one run:
- IVF had higher ANN Recall@200 than HNSW.
- HNSW had higher final MRR@10 than IVF.
- The untouched original ANCE checkpoint had the highest MRR@10 of all three.
- Untouched ANCE tied HNSW on relevance Recall@100 and tied both trained branches on Recall@1000.

Therefore, higher ANN Recall@200 did not translate into better downstream retriever performance in this pilot, and the short teacher-training procedure did not improve over the original ANCE starting checkpoint.

## Teacher diagnostics

Selected passage ID overlap between HNSW and IVF pools:
- 43.44% shared fraction over 1000 queries.

Teacher agreement:
- Top teacher passage matched in 993/1000 queries.
- Mean positive teacher probability mass: HNSW 0.963693, IVF 0.964083.
- Mean JSD between teacher distributions aligned by passage ID: 0.00523051.

Interpretation:
ANN changes which passage identities are presented to the teacher, but the teacher mostly concentrates probability on the same known positive in this pilot. Different passage IDs were not shown to be semantically different.

## Claim boundary

Supported:
- ANN backend + fixed selection changes training candidate identities.
- ANN candidate differences can propagate through a teacher-based training pipeline.
- In this run, IVF had higher ANN Recall@200 while HNSW produced a higher-MRR trained retriever.
- Both teacher-trained branches performed below the untouched ANCE control on MRR@10.

Not established:
- General ANN family superiority.
- That lower or higher ANN recall causally improves training.
- Large teacher-supervision differences in general.
- Semantic differences between the candidate pools.
- Statistical significance (one seed only; no repeat-run variance control).
