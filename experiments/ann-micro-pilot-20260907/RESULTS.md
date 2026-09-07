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

## ANN Recall@200 (not relevance recall)

- HNSW: 95.8155%
- IVF: 96.746%

IVF had higher ANN nearest-neighbor agreement.

## Final Retrieval Metrics

| Retriever | MRR@10 | Recall@100 | Recall@1000 |
|---|---:|---:|---:|
| Untouched ANCE | 0.890777 | 0.993 | 1.000 |
| HNSW + teacher | 0.866675 | 0.993 | 1.000 |
| IVF + teacher | 0.855858 | 0.991 | 1.000 |

Higher ANN Recall@200 did not translate into better final retriever performance in this pilot.

## Teacher diagnostics

Selected passage ID overlap between HNSW and IVF pools:
- 43.44% shared fraction over 1000 queries.

Teacher agreement:
- Top teacher passage matched in 993/1000 queries.
- Mean positive teacher probability mass: HNSW 0.963693, IVF 0.964083.
- Mean JSD between teacher distributions aligned by passage ID: 0.00523051.

Interpretation:
ANN changes which passage identities are presented to the teacher, but the teacher mostly concentrates probability on the same passages in this pilot.

## Claim boundary

Supported:
- ANN backend + fixed selection changes training candidate identities.
- ANN candidate differences can propagate through a teacher-based training pipeline.

Not established:
- General ANN family superiority.
- Higher ANN recall improves training.
- Large teacher-supervision differences.
- Statistical significance (one seed only).
