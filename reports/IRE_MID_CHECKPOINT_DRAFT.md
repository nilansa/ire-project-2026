# IRE Mid-Checkpoint Progress Report — Draft

**Team Ceous · 22 October 2026 · Draft for the 31 October checkpoint**

## Question and completed evidence

We ask whether approximate-nearest-neighbor (ANN) candidate generation changes which query–passage pairs receive limited relevance feedback and the retriever after post-training. Recall@200 is the fraction of exact-search top-200 passage IDs recovered; candidate Jaccard is intersection divided by union; selected shared fraction is shared IDs divided by 20, not Jaccard or relevance recall.

The **300-update, 50k-passage MS MARCO ANCE sanity pilot** used known qrel positives and unlabelled negatives, with no cross-encoder teacher. HNSW/IVF calibration Recall@200 was 0.955975/0.962250; top-200 candidate Jaccard was 0.91736; selected-20 shared fraction was 0.4344. MRR@10 (mean reciprocal rank of the first relevant result within ten) was 0.826063/0.833257/0.850488 for exact/HNSW/IVF training, versus 0.890777 for untouched ANCE. Every trained branch was lower; this was one seed.

A **separate 100-update MS MARCO pilot** used a cross-encoder (a model that scores query and passage text together) on 32,312 pairs for soft-label training. HNSW/IVF MRR@10 was 0.866675/0.855858, below the same untouched baseline. The teacher's top passage agreed in 993/1,000 queries; mean aligned-distribution Jensen–Shannon divergence (a measure of distribution difference) was 0.00523. This shows neither a teacher benefit nor a backend effect. **SciFact is separate:** its two-seed teacher pilot has duplicate query texts across train/test IDs; repair and audit that split before interpreting its results.

## Synthetic selector control

The CPU control reorders the same 200 synthetic IDs. At full reorder, rank-position selection shared 0.060 of its 20 IDs on average across ten seeds; a selector choosing IDs independently of rank stayed at 1.000. Independent random 20-of-200 pairs averaged 0.09865 (expected 0.10). This demonstrates a possible rank-sensitivity mechanism only. It does **not** validate the historical 0.4344: the original selector, query-to-seed mapping, and per-query lists are unavailable.

## Prior work, next steps, and schedule

ANCE already uses ANN-mined negatives; RocketQA uses cross-encoder filtering; SimANS most directly establishes that negative rank selection matters. The narrow open test is whether backend identity changes supervision when the remaining pipeline is fixed. No “first” or superiority claim is supported. See the [11-paper survey](../survey/docs/IRE_MID_2026_NEAREST_PRIOR_WORK.md).

Next, recover and replay the historical selector; establish a clean split and a working positive-control training setup; then compare exact/HNSW/IVF with a fixed teacher and budget across at least three paired seeds. Stop if historical inputs remain unavailable, split leakage remains, or the positive control fails to beat untouched ANCE across three seeds. The **31 October** deliverable is the one-page checkpoint; the **15 November** deliverable is the final written report and video.

**Sources:** [research plan](../RESEARCH_PLAN.md); [300-update no-teacher record](../survey/experiments/02-ann-backend-negative-selection-sanity.md); [100-update teacher record](../experiments/ann-micro-pilot-20260907/RESULTS.md); [SciFact split caveat](../docs/index.html); [synthetic control](../experiments/selector-rank-falsification/RESULTS.md). The original research plan is unchanged.
