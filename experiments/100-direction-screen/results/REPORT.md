# 100-direction empirical screen for retrieval post-training

## Bottom line

This is a broad **sanity screen**, not a publication claim. It compares 100 practical feedback-selection configurations under a fixed label budget. The 100 configurations are 25 substantially different pair-selection primitives crossed with four broadly useful query-budget allocation policies.

- Datasets: cranfield, scifact
- Split seeds: 0, 1, 2
- Configurations: 100
- Dataset-seed-config runs: 600
- Feedback budget: 12 pairs/query on average; one known positive/query is fixed outside the feedback budget.
- Student: one global logistic retrieval scorer over lexical, semantic, character, title, coverage, rank and disagreement signals; interpolation with the untouched retriever is tuned on validation queries.
- Feedback oracle: relevance judgments are revealed **only after** a pair is selected. No selector sees the hidden label.
- Wall-clock runtime: 2.4 minutes.

## Evaluation of the submitted plan

The submitted ANN-backend question is valid as a diagnostic, but too narrow as the final project thesis. Its own pilot already showed the key limitation: candidate identities changed sharply, yet coarse difficulty and teacher distributions were almost unchanged, and both post-trained branches degraded. A stronger, general direction is therefore **budgeted feedback allocation for retrieval post-training**: candidate generation is one source of variation, but the research object is the policy that decides which query–document pairs receive expensive labels.

The screen below treats ANN choice as one component inside a larger data-selection problem and directly compares exploitation, uncertainty, disagreement, semantic coverage, experimental design, query allocation and policy portfolios.

## Best 5 directions

| Rank | Direction | Mean NDCG@10 | Δ vs standard top-candidate labeling | Worst-dataset Δ | Win rate | Positive yield | Semantic diversity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Disagreement-adaptive budget + Uniform rank coverage | 0.3469 | +0.0047 | +0.0036 | 50% | 0.3% | 0.666 |
| 2 | Topic-balanced budget + Uncertainty–diversity MMR | 0.3439 | +0.0016 | -0.0017 | 83% | 0.2% | 0.807 |
| 3 | Uniform per-query budget + Uncertainty–diversity MMR | 0.3435 | +0.0012 | -0.0022 | 83% | 0.2% | 0.805 |
| 4 | Uniform per-query budget + Relevance–diversity MMR | 0.3446 | +0.0024 | +0.0004 | 50% | 0.5% | 0.745 |
| 5 | Topic-balanced budget + Lexical-first selection | 0.3441 | +0.0019 | +0.0014 | 50% | 0.8% | 0.580 |

### How to interpret the top five

Promote a direction only when it is not merely high on the mean: it should be non-negative on the worse dataset, beat the standard top-candidate policy repeatedly, and retain a plausible mechanism such as finding hidden positives, covering distinct candidate regions, or allocating more budget to difficult queries.

## Baselines and controls

| Dataset | Untouched NDCG@10 | Untouched MRR@10 | Untouched R@100 | Cheap prior NDCG@10 | Candidate relevance recall |
|---|---:|---:|---:|---:|---:|
| cranfield | 0.0162 | 0.0263 | 0.1087 | 0.0007 | 0.0999 |
| scifact | 0.6719 | 0.6250 | 0.9322 | 0.1559 | 0.9390 |

## All 100 directions

| Rank | ID | Direction | Mean NDCG@10 | Δ top-candidate control | Worst-dataset Δ | Win rate | Positive yield | Diversity |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `disagreement__uniform_rank` | Disagreement-adaptive budget + Uniform rank coverage | 0.3469 | +0.0047 | +0.0036 | 50% | 0.3% | 0.666 |
| 2 | `topic_balanced__mmr_uncertainty` | Topic-balanced budget + Uncertainty–diversity MMR | 0.3439 | +0.0016 | -0.0017 | 83% | 0.2% | 0.807 |
| 3 | `uniform__mmr_uncertainty` | Uniform per-query budget + Uncertainty–diversity MMR | 0.3435 | +0.0012 | -0.0022 | 83% | 0.2% | 0.805 |
| 4 | `uniform__mmr_relevance` | Uniform per-query budget + Relevance–diversity MMR | 0.3446 | +0.0024 | +0.0004 | 50% | 0.5% | 0.745 |
| 5 | `topic_balanced__top_bm25` | Topic-balanced budget + Lexical-first selection | 0.3441 | +0.0019 | +0.0014 | 50% | 0.8% | 0.580 |
| 6 | `difficulty__false_positive` | Difficulty-adaptive budget + Likely false-positive hunting | 0.3466 | +0.0044 | -0.0021 | 33% | 0.3% | 0.613 |
| 7 | `disagreement__mmr_relevance` | Disagreement-adaptive budget + Relevance–diversity MMR | 0.3453 | +0.0030 | -0.0011 | 33% | 0.4% | 0.742 |
| 8 | `topic_balanced__retriever_disagreement` | Topic-balanced budget + Retriever disagreement | 0.3421 | -0.0001 | -0.0004 | 50% | 0.5% | 0.506 |
| 9 | `topic_balanced__top_base` | Topic-balanced budget + Top current-retriever candidates | 0.3422 | -0.0001 | -0.0010 | 50% | 0.8% | 0.524 |
| 10 | `uniform__leverage` | Uniform per-query budget + Feature-space leverage | 0.3423 | +0.0001 | -0.0033 | 50% | 0.4% | 0.757 |
| 11 | `uniform__uniform_rank` | Uniform per-query budget + Uniform rank coverage | 0.3433 | +0.0010 | -0.0026 | 33% | 0.4% | 0.674 |
| 12 | `disagreement__leverage` | Disagreement-adaptive budget + Feature-space leverage | 0.3409 | -0.0014 | -0.0019 | 50% | 0.3% | 0.760 |
| 13 | `disagreement__top_bm25` | Disagreement-adaptive budget + Lexical-first selection | 0.3424 | +0.0001 | -0.0015 | 33% | 0.7% | 0.580 |
| 14 | `uniform__retriever_disagreement` | Uniform per-query budget + Retriever disagreement | 0.3410 | -0.0013 | -0.0026 | 50% | 0.5% | 0.505 |
| 15 | `disagreement__retriever_disagreement` | Disagreement-adaptive budget + Retriever disagreement | 0.3408 | -0.0015 | -0.0031 | 50% | 0.4% | 0.503 |
| 16 | `uniform__top_bm25` | Uniform per-query budget + Lexical-first selection | 0.3426 | +0.0004 | -0.0035 | 33% | 0.8% | 0.586 |
| 17 | `difficulty__d_optimal` | Difficulty-adaptive budget + D-optimal experimental design | 0.3405 | -0.0018 | -0.0023 | 50% | 0.4% | 0.746 |
| 18 | `topic_balanced__coverage` | Topic-balanced budget + Query-term coverage | 0.3426 | +0.0003 | -0.0036 | 33% | 0.7% | 0.577 |
| 19 | `difficulty__leverage` | Difficulty-adaptive budget + Feature-space leverage | 0.3406 | -0.0017 | -0.0032 | 50% | 0.3% | 0.757 |
| 20 | `difficulty__top_base` | Difficulty-adaptive budget + Top current-retriever candidates | 0.3421 | -0.0002 | -0.0028 | 33% | 0.8% | 0.528 |
| 21 | `topic_balanced__leverage` | Topic-balanced budget + Feature-space leverage | 0.3401 | -0.0021 | -0.0022 | 50% | 0.4% | 0.754 |
| 22 | `difficulty__retriever_disagreement` | Difficulty-adaptive budget + Retriever disagreement | 0.3405 | -0.0017 | -0.0035 | 50% | 0.4% | 0.506 |
| 23 | `disagreement__top_base` | Disagreement-adaptive budget + Top current-retriever candidates | 0.3418 | -0.0004 | -0.0026 | 33% | 0.8% | 0.525 |
| 24 | `disagreement__hard_negative` | Disagreement-adaptive budget + Hard-negative proxy | 0.3412 | -0.0011 | -0.0013 | 33% | 0.3% | 0.584 |
| 25 | `topic_balanced__mmr_relevance` | Topic-balanced budget + Relevance–diversity MMR | 0.3427 | +0.0005 | -0.0011 | 17% | 0.5% | 0.744 |
| 26 | `topic_balanced__uniform_rank` | Topic-balanced budget + Uniform rank coverage | 0.3409 | -0.0013 | -0.0025 | 33% | 0.4% | 0.674 |
| 27 | `difficulty__rrf_consensus` | Difficulty-adaptive budget + Multi-retriever consensus | 0.3405 | -0.0018 | -0.0033 | 33% | 0.8% | 0.519 |
| 28 | `difficulty__mmr_relevance` | Difficulty-adaptive budget + Relevance–diversity MMR | 0.3404 | -0.0019 | -0.0031 | 33% | 0.5% | 0.744 |
| 29 | `topic_balanced__top_char` | Topic-balanced budget + Character-robust selection | 0.3402 | -0.0021 | -0.0029 | 33% | 0.8% | 0.572 |
| 30 | `uniform__top_base` | Uniform per-query budget + Top current-retriever candidates | 0.3423 | +0.0000 | +0.0000 | 0% | 0.8% | 0.528 |
| 31 | `uniform__hard_negative` | Uniform per-query budget + Hard-negative proxy | 0.3413 | -0.0009 | -0.0029 | 17% | 0.3% | 0.588 |
| 32 | `topic_balanced__false_negative` | Topic-balanced budget + Likely false-negative hunting | 0.3394 | -0.0028 | -0.0035 | 33% | 0.5% | 0.699 |
| 33 | `difficulty__cluster_representatives` | Difficulty-adaptive budget + Candidate-cluster representatives | 0.3396 | -0.0027 | -0.0042 | 33% | 0.2% | 0.728 |
| 34 | `disagreement__top_char` | Disagreement-adaptive budget + Character-robust selection | 0.3395 | -0.0027 | -0.0042 | 33% | 0.8% | 0.571 |
| 35 | `difficulty__uniform_rank` | Difficulty-adaptive budget + Uniform rank coverage | 0.3380 | -0.0042 | -0.0049 | 50% | 0.3% | 0.673 |
| 36 | `disagreement__rrf_consensus` | Disagreement-adaptive budget + Multi-retriever consensus | 0.3392 | -0.0030 | -0.0055 | 33% | 0.7% | 0.515 |
| 37 | `uniform__cluster_representatives` | Uniform per-query budget + Candidate-cluster representatives | 0.3385 | -0.0038 | -0.0042 | 33% | 0.3% | 0.727 |
| 38 | `topic_balanced__rank_strata` | Topic-balanced budget + Rank-stratified sampling | 0.3386 | -0.0037 | -0.0050 | 33% | 0.3% | 0.662 |
| 39 | `topic_balanced__rrf_consensus` | Topic-balanced budget + Multi-retriever consensus | 0.3385 | -0.0038 | -0.0050 | 33% | 0.8% | 0.513 |
| 40 | `disagreement__d_optimal` | Disagreement-adaptive budget + D-optimal experimental design | 0.3385 | -0.0037 | -0.0052 | 33% | 0.3% | 0.746 |
| 41 | `disagreement__head_tail` | Disagreement-adaptive budget + Head–tail mixture | 0.3385 | -0.0038 | -0.0056 | 33% | 0.4% | 0.694 |
| 42 | `uniform__d_optimal` | Uniform per-query budget + D-optimal experimental design | 0.3384 | -0.0039 | -0.0055 | 33% | 0.4% | 0.745 |
| 43 | `uniform__rank_strata` | Uniform per-query budget + Rank-stratified sampling | 0.3383 | -0.0040 | -0.0055 | 33% | 0.3% | 0.660 |
| 44 | `disagreement__mmr_uncertainty` | Disagreement-adaptive budget + Uncertainty–diversity MMR | 0.3384 | -0.0038 | -0.0124 | 50% | 0.2% | 0.804 |
| 45 | `topic_balanced__hard_negative` | Topic-balanced budget + Hard-negative proxy | 0.3396 | -0.0026 | -0.0030 | 0% | 0.3% | 0.586 |
| 46 | `disagreement__top_title` | Disagreement-adaptive budget + Title-focused selection | 0.3378 | -0.0045 | -0.0076 | 33% | 0.5% | 0.628 |
| 47 | `uniform__farthest_first` | Uniform per-query budget + Semantic k-center coverage | 0.3368 | -0.0055 | -0.0098 | 50% | 0.1% | 0.896 |
| 48 | `disagreement__center_outlier_mix` | Disagreement-adaptive budget + Typical–outlier mixture | 0.3383 | -0.0040 | -0.0046 | 17% | 0.4% | 0.750 |
| 49 | `topic_balanced__farthest_first` | Topic-balanced budget + Semantic k-center coverage | 0.3366 | -0.0056 | -0.0095 | 50% | 0.1% | 0.896 |
| 50 | `uniform__portfolio` | Uniform per-query budget + Diversified policy portfolio | 0.3381 | -0.0042 | -0.0042 | 17% | 0.5% | 0.688 |
| 51 | `topic_balanced__top_title` | Topic-balanced budget + Title-focused selection | 0.3376 | -0.0047 | -0.0075 | 33% | 0.5% | 0.632 |
| 52 | `disagreement__false_positive` | Disagreement-adaptive budget + Likely false-positive hunting | 0.3395 | -0.0027 | -0.0040 | 0% | 0.3% | 0.609 |
| 53 | `difficulty__farthest_first` | Difficulty-adaptive budget + Semantic k-center coverage | 0.3365 | -0.0058 | -0.0099 | 50% | 0.1% | 0.896 |
| 54 | `difficulty__random` | Difficulty-adaptive budget + Random exploration | 0.3368 | -0.0055 | -0.0062 | 33% | 0.2% | 0.729 |
| 55 | `uniform__top_title` | Uniform per-query budget + Title-focused selection | 0.3371 | -0.0051 | -0.0075 | 33% | 0.5% | 0.633 |
| 56 | `difficulty__top_char` | Difficulty-adaptive budget + Character-robust selection | 0.3382 | -0.0040 | -0.0059 | 17% | 0.8% | 0.575 |
| 57 | `uniform__rrf_consensus` | Uniform per-query budget + Multi-retriever consensus | 0.3370 | -0.0053 | -0.0074 | 33% | 0.8% | 0.518 |
| 58 | `uniform__false_positive` | Uniform per-query budget + Likely false-positive hunting | 0.3391 | -0.0032 | -0.0040 | 0% | 0.3% | 0.614 |
| 59 | `topic_balanced__random` | Topic-balanced budget + Random exploration | 0.3366 | -0.0057 | -0.0066 | 33% | 0.2% | 0.728 |
| 60 | `difficulty__mmr_uncertainty` | Difficulty-adaptive budget + Uncertainty–diversity MMR | 0.3359 | -0.0064 | -0.0142 | 67% | 0.2% | 0.804 |
| 61 | `difficulty__hard_negative` | Difficulty-adaptive budget + Hard-negative proxy | 0.3390 | -0.0033 | -0.0041 | 0% | 0.3% | 0.590 |
| 62 | `topic_balanced__d_optimal` | Topic-balanced budget + D-optimal experimental design | 0.3370 | -0.0052 | -0.0081 | 33% | 0.4% | 0.746 |
| 63 | `topic_balanced__false_positive` | Topic-balanced budget + Likely false-positive hunting | 0.3387 | -0.0036 | -0.0041 | 0% | 0.3% | 0.611 |
| 64 | `disagreement__random` | Disagreement-adaptive budget + Random exploration | 0.3368 | -0.0055 | -0.0086 | 33% | 0.2% | 0.727 |
| 65 | `uniform__false_negative` | Uniform per-query budget + Likely false-negative hunting | 0.3373 | -0.0049 | -0.0064 | 17% | 0.6% | 0.700 |
| 66 | `topic_balanced__uncertainty` | Topic-balanced budget + Current-model uncertainty | 0.3362 | -0.0061 | -0.0088 | 33% | 0.2% | 0.705 |
| 67 | `difficulty__portfolio` | Difficulty-adaptive budget + Diversified policy portfolio | 0.3361 | -0.0061 | -0.0089 | 33% | 0.5% | 0.684 |
| 68 | `difficulty__false_negative` | Difficulty-adaptive budget + Likely false-negative hunting | 0.3370 | -0.0052 | -0.0070 | 17% | 0.5% | 0.702 |
| 69 | `uniform__random` | Uniform per-query budget + Random exploration | 0.3352 | -0.0070 | -0.0078 | 33% | 0.2% | 0.729 |
| 70 | `difficulty__top_bm25` | Difficulty-adaptive budget + Lexical-first selection | 0.3369 | -0.0054 | -0.0093 | 17% | 0.8% | 0.587 |
| 71 | `difficulty__center_outlier_mix` | Difficulty-adaptive budget + Typical–outlier mixture | 0.3372 | -0.0051 | -0.0061 | 0% | 0.5% | 0.750 |
| 72 | `disagreement__rank_strata` | Disagreement-adaptive budget + Rank-stratified sampling | 0.3358 | -0.0065 | -0.0070 | 17% | 0.3% | 0.654 |
| 73 | `uniform__center_outlier_mix` | Uniform per-query budget + Typical–outlier mixture | 0.3369 | -0.0053 | -0.0057 | 0% | 0.5% | 0.743 |
| 74 | `disagreement__portfolio` | Disagreement-adaptive budget + Diversified policy portfolio | 0.3356 | -0.0067 | -0.0118 | 33% | 0.4% | 0.685 |
| 75 | `disagreement__farthest_first` | Disagreement-adaptive budget + Semantic k-center coverage | 0.3346 | -0.0077 | -0.0139 | 50% | 0.0% | 0.896 |
| 76 | `difficulty__top_title` | Difficulty-adaptive budget + Title-focused selection | 0.3348 | -0.0074 | -0.0102 | 33% | 0.5% | 0.637 |
| 77 | `uniform__uncertainty` | Uniform per-query budget + Current-model uncertainty | 0.3350 | -0.0073 | -0.0111 | 33% | 0.2% | 0.705 |
| 78 | `disagreement__uncertainty` | Disagreement-adaptive budget + Current-model uncertainty | 0.3350 | -0.0073 | -0.0112 | 33% | 0.2% | 0.704 |
| 79 | `difficulty__uncertainty` | Difficulty-adaptive budget + Current-model uncertainty | 0.3349 | -0.0074 | -0.0114 | 33% | 0.2% | 0.704 |
| 80 | `difficulty__head_tail` | Difficulty-adaptive budget + Head–tail mixture | 0.3347 | -0.0076 | -0.0118 | 33% | 0.5% | 0.694 |
| 81 | `uniform__coverage` | Uniform per-query budget + Query-term coverage | 0.3361 | -0.0062 | -0.0089 | 0% | 0.7% | 0.579 |
| 82 | `topic_balanced__cluster_representatives` | Topic-balanced budget + Candidate-cluster representatives | 0.3350 | -0.0073 | -0.0111 | 17% | 0.4% | 0.718 |
| 83 | `disagreement__false_negative` | Disagreement-adaptive budget + Likely false-negative hunting | 0.3339 | -0.0083 | -0.0132 | 33% | 0.5% | 0.699 |
| 84 | `topic_balanced__portfolio` | Topic-balanced budget + Diversified policy portfolio | 0.3343 | -0.0079 | -0.0117 | 17% | 0.5% | 0.682 |
| 85 | `uniform__top_char` | Uniform per-query budget + Character-robust selection | 0.3332 | -0.0091 | -0.0109 | 17% | 0.8% | 0.577 |
| 86 | `topic_balanced__center_outlier_mix` | Topic-balanced budget + Typical–outlier mixture | 0.3335 | -0.0088 | -0.0127 | 17% | 0.4% | 0.753 |
| 87 | `disagreement__committee` | Disagreement-adaptive budget + Bootstrap committee disagreement | 0.3326 | -0.0097 | -0.0159 | 33% | 0.1% | 0.674 |
| 88 | `difficulty__rank_strata` | Difficulty-adaptive budget + Rank-stratified sampling | 0.3330 | -0.0093 | -0.0129 | 17% | 0.3% | 0.656 |
| 89 | `disagreement__top_lsa` | Disagreement-adaptive budget + Semantic-first selection | 0.3334 | -0.0088 | -0.0153 | 17% | 0.6% | 0.374 |
| 90 | `topic_balanced__top_lsa` | Topic-balanced budget + Semantic-first selection | 0.3333 | -0.0090 | -0.0152 | 17% | 0.7% | 0.373 |
| 91 | `uniform__top_lsa` | Uniform per-query budget + Semantic-first selection | 0.3338 | -0.0085 | -0.0130 | 0% | 0.7% | 0.376 |
| 92 | `difficulty__coverage` | Difficulty-adaptive budget + Query-term coverage | 0.3342 | -0.0081 | -0.0191 | 17% | 0.7% | 0.579 |
| 93 | `difficulty__top_lsa` | Difficulty-adaptive budget + Semantic-first selection | 0.3332 | -0.0091 | -0.0149 | 0% | 0.7% | 0.377 |
| 94 | `uniform__head_tail` | Uniform per-query budget + Head–tail mixture | 0.3309 | -0.0113 | -0.0193 | 33% | 0.5% | 0.693 |
| 95 | `difficulty__committee` | Difficulty-adaptive budget + Bootstrap committee disagreement | 0.3309 | -0.0114 | -0.0194 | 33% | 0.2% | 0.673 |
| 96 | `uniform__committee` | Uniform per-query budget + Bootstrap committee disagreement | 0.3306 | -0.0117 | -0.0199 | 33% | 0.2% | 0.674 |
| 97 | `disagreement__cluster_representatives` | Disagreement-adaptive budget + Candidate-cluster representatives | 0.3311 | -0.0112 | -0.0174 | 17% | 0.2% | 0.730 |
| 98 | `topic_balanced__committee` | Topic-balanced budget + Bootstrap committee disagreement | 0.3299 | -0.0123 | -0.0212 | 33% | 0.2% | 0.671 |
| 99 | `disagreement__coverage` | Disagreement-adaptive budget + Query-term coverage | 0.3303 | -0.0120 | -0.0213 | 0% | 0.7% | 0.578 |
| 100 | `topic_balanced__head_tail` | Topic-balanced budget + Head–tail mixture | 0.3269 | -0.0153 | -0.0274 | 33% | 0.5% | 0.695 |

## What has already been done versus what this adds

Previously completed:
- A 50k-passage MS MARCO pilot comparing exact, HNSW and IVF candidate pools.
- A teacher-scored HNSW-versus-IVF micro-pilot.
- Evidence that similar ANN recall can hide large pair-identity differences.
- No evidence yet that ANN family choice improves post-training; all reported trained branches were below the untouched ANCE control.

Added by this screen:
- A much broader 100-configuration comparison centered on the general decision problem: where should a fixed feedback budget be spent?
- Two datasets and repeated query splits rather than a single seed.
- Untouched, standard top-candidate, random, uncertainty, diversity, disagreement and experimental-design controls under one pipeline.
- A concrete shortlist whose hypotheses can be moved next to the full MS MARCO + cross-encoder setup.

## Claim boundary

- This is a lightweight reranking/post-training proxy, not dense-encoder fine-tuning.
- Cranfield and SciFact are small; the results identify promising mechanisms, not final effectiveness claims.
- Relevance judgments act as the feedback oracle. A real LLM/cross-encoder can supply graded, noisy feedback and may change the ranking.
- The next decisive experiment is to take the best five selectors, keep the same fixed budget, and run them with ANCE (or another dense retriever), a real cross-encoder, full-corpus exact evaluation, and at least three training seeds.

## Files

- `SUMMARY_100_DIRECTIONS.csv`: one aggregate row per direction.
- `ALL_RUNS.csv`: every dataset × seed × direction run.
- `BASELINES.csv`: untouched and prior controls.
- `TOP5.csv`: promoted shortlist.
- `metadata.json`: exact run configuration.
