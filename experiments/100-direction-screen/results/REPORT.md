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
- Wall-clock runtime: 2.7 minutes.

## Evaluation of the submitted plan

The submitted ANN-backend question is valid as a diagnostic, but too narrow as the final project thesis. Its own pilot already showed the key limitation: candidate identities changed sharply, yet coarse difficulty and teacher distributions were almost unchanged, and both post-trained branches degraded. A stronger, general direction is therefore **budgeted feedback allocation for retrieval post-training**: candidate generation is one source of variation, but the research object is the policy that decides which query–document pairs receive expensive labels.

The screen below treats ANN choice as one component inside a larger data-selection problem and directly compares exploitation, uncertainty, disagreement, semantic coverage, experimental design, query allocation and policy portfolios.

## Best 5 directions

| Rank | Direction | Mean NDCG@10 | Δ vs standard top-candidate labeling | Worst-dataset Δ | Win rate | Positive yield | Semantic diversity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Difficulty-adaptive budget + Feature-space leverage | 0.5423 | +0.0009 | -0.0002 | 67% | 3.5% | 0.757 |
| 2 | Uniform per-query budget + Uncertainty–diversity MMR | 0.5412 | -0.0002 | -0.0022 | 50% | 1.6% | 0.764 |
| 3 | Disagreement-adaptive budget + Hard-negative proxy | 0.5413 | -0.0001 | -0.0008 | 33% | 0.9% | 0.719 |
| 4 | Disagreement-adaptive budget + Uniform rank coverage | 0.5414 | -0.0000 | -0.0058 | 50% | 2.4% | 0.672 |
| 5 | Topic-balanced budget + Uncertainty–diversity MMR | 0.5397 | -0.0017 | -0.0018 | 50% | 1.7% | 0.766 |

### How to interpret the top five

Promote a direction only when it is not merely high on the mean: it should be non-negative on the worse dataset, beat the standard top-candidate policy repeatedly, and retain a plausible mechanism such as finding hidden positives, covering distinct candidate regions, or allocating more budget to difficult queries.

## Baselines and controls

| Dataset | Untouched NDCG@10 | Untouched MRR@10 | Untouched R@100 | Cheap prior NDCG@10 | Candidate relevance recall |
|---|---:|---:|---:|---:|---:|
| cranfield | 0.3980 | 0.5265 | 0.7530 | 0.0000 | 0.8387 |
| scifact | 0.6719 | 0.6250 | 0.9322 | 0.1559 | 0.9390 |

## All 100 directions

| Rank | ID | Direction | Mean NDCG@10 | Δ top-candidate control | Worst-dataset Δ | Win rate | Positive yield | Diversity |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `difficulty__leverage` | Difficulty-adaptive budget + Feature-space leverage | 0.5423 | +0.0009 | -0.0002 | 67% | 3.5% | 0.757 |
| 2 | `uniform__mmr_uncertainty` | Uniform per-query budget + Uncertainty–diversity MMR | 0.5412 | -0.0002 | -0.0022 | 50% | 1.6% | 0.764 |
| 3 | `disagreement__hard_negative` | Disagreement-adaptive budget + Hard-negative proxy | 0.5413 | -0.0001 | -0.0008 | 33% | 0.9% | 0.719 |
| 4 | `disagreement__uniform_rank` | Disagreement-adaptive budget + Uniform rank coverage | 0.5414 | -0.0000 | -0.0058 | 50% | 2.4% | 0.672 |
| 5 | `topic_balanced__mmr_uncertainty` | Topic-balanced budget + Uncertainty–diversity MMR | 0.5397 | -0.0017 | -0.0018 | 50% | 1.7% | 0.766 |
| 6 | `uniform__leverage` | Uniform per-query budget + Feature-space leverage | 0.5407 | -0.0007 | -0.0048 | 50% | 3.7% | 0.758 |
| 7 | `uniform__mmr_relevance` | Uniform per-query budget + Relevance–diversity MMR | 0.5405 | -0.0009 | -0.0062 | 50% | 4.4% | 0.753 |
| 8 | `difficulty__d_optimal` | Difficulty-adaptive budget + D-optimal experimental design | 0.5392 | -0.0022 | -0.0033 | 50% | 3.2% | 0.747 |
| 9 | `disagreement__top_base` | Disagreement-adaptive budget + Top current-retriever candidates | 0.5407 | -0.0007 | -0.0031 | 33% | 8.5% | 0.524 |
| 10 | `disagreement__mmr_relevance` | Disagreement-adaptive budget + Relevance–diversity MMR | 0.5417 | +0.0003 | -0.0066 | 33% | 4.1% | 0.751 |
| 11 | `topic_balanced__uniform_rank` | Topic-balanced budget + Uniform rank coverage | 0.5400 | -0.0014 | -0.0027 | 33% | 2.4% | 0.677 |
| 12 | `uniform__top_base` | Uniform per-query budget + Top current-retriever candidates | 0.5414 | +0.0000 | +0.0000 | 0% | 8.8% | 0.530 |
| 13 | `topic_balanced__mmr_relevance` | Topic-balanced budget + Relevance–diversity MMR | 0.5398 | -0.0016 | -0.0053 | 33% | 4.5% | 0.751 |
| 14 | `uniform__portfolio` | Uniform per-query budget + Diversified policy portfolio | 0.5389 | -0.0026 | -0.0041 | 33% | 5.4% | 0.667 |
| 15 | `difficulty__mmr_relevance` | Difficulty-adaptive budget + Relevance–diversity MMR | 0.5375 | -0.0040 | -0.0049 | 50% | 4.4% | 0.753 |
| 16 | `difficulty__random` | Difficulty-adaptive budget + Random exploration | 0.5374 | -0.0040 | -0.0048 | 50% | 1.7% | 0.727 |
| 17 | `topic_balanced__leverage` | Topic-balanced budget + Feature-space leverage | 0.5376 | -0.0038 | -0.0055 | 50% | 3.7% | 0.756 |
| 18 | `difficulty__cluster_representatives` | Difficulty-adaptive budget + Candidate-cluster representatives | 0.5378 | -0.0036 | -0.0061 | 50% | 2.4% | 0.728 |
| 19 | `topic_balanced__top_base` | Topic-balanced budget + Top current-retriever candidates | 0.5386 | -0.0028 | -0.0046 | 33% | 8.9% | 0.525 |
| 20 | `uniform__uniform_rank` | Uniform per-query budget + Uniform rank coverage | 0.5397 | -0.0017 | -0.0079 | 33% | 2.4% | 0.677 |
| 21 | `disagreement__leverage` | Disagreement-adaptive budget + Feature-space leverage | 0.5375 | -0.0039 | -0.0069 | 50% | 3.4% | 0.760 |
| 22 | `difficulty__false_positive` | Difficulty-adaptive budget + Likely false-positive hunting | 0.5381 | -0.0033 | -0.0045 | 33% | 0.4% | 0.764 |
| 23 | `difficulty__uniform_rank` | Difficulty-adaptive budget + Uniform rank coverage | 0.5362 | -0.0052 | -0.0056 | 50% | 2.4% | 0.674 |
| 24 | `topic_balanced__rank_strata` | Topic-balanced budget + Rank-stratified sampling | 0.5376 | -0.0039 | -0.0050 | 33% | 2.6% | 0.626 |
| 25 | `uniform__d_optimal` | Uniform per-query budget + D-optimal experimental design | 0.5373 | -0.0041 | -0.0055 | 33% | 3.5% | 0.746 |
| 26 | `disagreement__rrf_consensus` | Disagreement-adaptive budget + Multi-retriever consensus | 0.5372 | -0.0042 | -0.0055 | 33% | 8.2% | 0.518 |
| 27 | `topic_balanced__top_bm25` | Topic-balanced budget + Lexical-first selection | 0.5380 | -0.0034 | -0.0083 | 33% | 8.0% | 0.587 |
| 28 | `topic_balanced__hard_negative` | Topic-balanced budget + Hard-negative proxy | 0.5391 | -0.0024 | -0.0030 | 0% | 1.0% | 0.720 |
| 29 | `difficulty__rrf_consensus` | Difficulty-adaptive budget + Multi-retriever consensus | 0.5376 | -0.0038 | -0.0043 | 17% | 8.3% | 0.517 |
| 30 | `difficulty__top_base` | Difficulty-adaptive budget + Top current-retriever candidates | 0.5377 | -0.0038 | -0.0100 | 33% | 8.6% | 0.524 |
| 31 | `uniform__cluster_representatives` | Uniform per-query budget + Candidate-cluster representatives | 0.5354 | -0.0061 | -0.0087 | 50% | 2.8% | 0.727 |
| 32 | `disagreement__d_optimal` | Disagreement-adaptive budget + D-optimal experimental design | 0.5358 | -0.0056 | -0.0061 | 33% | 3.2% | 0.747 |
| 33 | `disagreement__false_positive` | Disagreement-adaptive budget + Likely false-positive hunting | 0.5370 | -0.0044 | -0.0048 | 17% | 0.4% | 0.765 |
| 34 | `uniform__rank_strata` | Uniform per-query budget + Rank-stratified sampling | 0.5372 | -0.0042 | -0.0055 | 17% | 2.8% | 0.624 |
| 35 | `topic_balanced__rrf_consensus` | Topic-balanced budget + Multi-retriever consensus | 0.5369 | -0.0045 | -0.0050 | 17% | 8.7% | 0.518 |
| 36 | `uniform__random` | Uniform per-query budget + Random exploration | 0.5345 | -0.0069 | -0.0078 | 50% | 1.8% | 0.727 |
| 37 | `disagreement__rank_strata` | Disagreement-adaptive budget + Rank-stratified sampling | 0.5358 | -0.0056 | -0.0070 | 33% | 2.7% | 0.617 |
| 38 | `topic_balanced__random` | Topic-balanced budget + Random exploration | 0.5342 | -0.0072 | -0.0079 | 50% | 1.9% | 0.728 |
| 39 | `difficulty__hard_negative` | Difficulty-adaptive budget + Hard-negative proxy | 0.5378 | -0.0036 | -0.0041 | 0% | 0.9% | 0.721 |
| 40 | `disagreement__uncertainty` | Disagreement-adaptive budget + Current-model uncertainty | 0.5368 | -0.0046 | -0.0112 | 33% | 2.0% | 0.628 |
| 41 | `uniform__farthest_first` | Uniform per-query budget + Semantic k-center coverage | 0.5360 | -0.0054 | -0.0098 | 33% | 0.3% | 0.895 |
| 42 | `uniform__rrf_consensus` | Uniform per-query budget + Multi-retriever consensus | 0.5366 | -0.0049 | -0.0074 | 17% | 8.7% | 0.522 |
| 43 | `disagreement__random` | Disagreement-adaptive budget + Random exploration | 0.5352 | -0.0062 | -0.0086 | 33% | 1.7% | 0.727 |
| 44 | `difficulty__uncertainty` | Difficulty-adaptive budget + Current-model uncertainty | 0.5358 | -0.0056 | -0.0114 | 33% | 2.1% | 0.628 |
| 45 | `topic_balanced__uncertainty` | Topic-balanced budget + Current-model uncertainty | 0.5347 | -0.0067 | -0.0088 | 33% | 2.1% | 0.630 |
| 46 | `topic_balanced__d_optimal` | Topic-balanced budget + D-optimal experimental design | 0.5345 | -0.0070 | -0.0081 | 33% | 3.5% | 0.746 |
| 47 | `topic_balanced__portfolio` | Topic-balanced budget + Diversified policy portfolio | 0.5356 | -0.0059 | -0.0117 | 33% | 5.4% | 0.659 |
| 48 | `uniform__uncertainty` | Uniform per-query budget + Current-model uncertainty | 0.5353 | -0.0062 | -0.0111 | 33% | 2.2% | 0.629 |
| 49 | `uniform__retriever_disagreement` | Uniform per-query budget + Retriever disagreement | 0.5354 | -0.0061 | -0.0121 | 33% | 5.2% | 0.506 |
| 50 | `difficulty__portfolio` | Difficulty-adaptive budget + Diversified policy portfolio | 0.5357 | -0.0057 | -0.0089 | 17% | 5.3% | 0.661 |
| 51 | `difficulty__mmr_uncertainty` | Difficulty-adaptive budget + Uncertainty–diversity MMR | 0.5340 | -0.0074 | -0.0142 | 50% | 1.7% | 0.762 |
| 52 | `disagreement__portfolio` | Disagreement-adaptive budget + Diversified policy portfolio | 0.5330 | -0.0084 | -0.0118 | 50% | 5.1% | 0.661 |
| 53 | `difficulty__farthest_first` | Difficulty-adaptive budget + Semantic k-center coverage | 0.5339 | -0.0075 | -0.0099 | 33% | 0.4% | 0.894 |
| 54 | `topic_balanced__false_positive` | Topic-balanced budget + Likely false-positive hunting | 0.5350 | -0.0064 | -0.0087 | 17% | 0.4% | 0.766 |
| 55 | `topic_balanced__farthest_first` | Topic-balanced budget + Semantic k-center coverage | 0.5332 | -0.0082 | -0.0095 | 33% | 0.3% | 0.895 |
| 56 | `disagreement__mmr_uncertainty` | Disagreement-adaptive budget + Uncertainty–diversity MMR | 0.5354 | -0.0060 | -0.0124 | 17% | 1.6% | 0.764 |
| 57 | `disagreement__retriever_disagreement` | Disagreement-adaptive budget + Retriever disagreement | 0.5342 | -0.0072 | -0.0146 | 33% | 4.8% | 0.502 |
| 58 | `uniform__hard_negative` | Uniform per-query budget + Hard-negative proxy | 0.5355 | -0.0059 | -0.0088 | 0% | 1.0% | 0.720 |
| 59 | `difficulty__retriever_disagreement` | Difficulty-adaptive budget + Retriever disagreement | 0.5341 | -0.0073 | -0.0146 | 33% | 4.8% | 0.504 |
| 60 | `difficulty__rank_strata` | Difficulty-adaptive budget + Rank-stratified sampling | 0.5328 | -0.0086 | -0.0129 | 33% | 2.7% | 0.620 |
| 61 | `topic_balanced__cluster_representatives` | Topic-balanced budget + Candidate-cluster representatives | 0.5322 | -0.0093 | -0.0111 | 33% | 3.4% | 0.719 |
| 62 | `uniform__false_positive` | Uniform per-query budget + Likely false-positive hunting | 0.5338 | -0.0076 | -0.0112 | 17% | 0.5% | 0.766 |
| 63 | `topic_balanced__false_negative` | Topic-balanced budget + Likely false-negative hunting | 0.5341 | -0.0074 | -0.0125 | 17% | 3.3% | 0.634 |
| 64 | `disagreement__center_outlier_mix` | Disagreement-adaptive budget + Typical–outlier mixture | 0.5333 | -0.0081 | -0.0128 | 17% | 2.3% | 0.751 |
| 65 | `disagreement__head_tail` | Disagreement-adaptive budget + Head–tail mixture | 0.5326 | -0.0089 | -0.0121 | 17% | 5.4% | 0.701 |
| 66 | `difficulty__top_bm25` | Difficulty-adaptive budget + Lexical-first selection | 0.5318 | -0.0096 | -0.0100 | 17% | 7.7% | 0.583 |
| 67 | `difficulty__head_tail` | Difficulty-adaptive budget + Head–tail mixture | 0.5317 | -0.0097 | -0.0118 | 17% | 5.3% | 0.700 |
| 68 | `disagreement__farthest_first` | Disagreement-adaptive budget + Semantic k-center coverage | 0.5306 | -0.0108 | -0.0139 | 33% | 0.2% | 0.895 |
| 69 | `disagreement__top_title` | Disagreement-adaptive budget + Title-focused selection | 0.5315 | -0.0099 | -0.0122 | 17% | 5.3% | 0.637 |
| 70 | `uniform__top_title` | Uniform per-query budget + Title-focused selection | 0.5315 | -0.0099 | -0.0124 | 17% | 5.5% | 0.641 |
| 71 | `topic_balanced__top_title` | Topic-balanced budget + Title-focused selection | 0.5311 | -0.0103 | -0.0131 | 17% | 5.8% | 0.640 |
| 72 | `uniform__top_bm25` | Uniform per-query budget + Lexical-first selection | 0.5325 | -0.0089 | -0.0222 | 33% | 8.0% | 0.589 |
| 73 | `topic_balanced__retriever_disagreement` | Topic-balanced budget + Retriever disagreement | 0.5325 | -0.0090 | -0.0176 | 17% | 5.1% | 0.506 |
| 74 | `disagreement__top_bm25` | Disagreement-adaptive budget + Lexical-first selection | 0.5318 | -0.0096 | -0.0208 | 33% | 7.7% | 0.584 |
| 75 | `difficulty__top_title` | Difficulty-adaptive budget + Title-focused selection | 0.5299 | -0.0115 | -0.0128 | 17% | 5.3% | 0.639 |
| 76 | `topic_balanced__center_outlier_mix` | Topic-balanced budget + Typical–outlier mixture | 0.5291 | -0.0123 | -0.0127 | 17% | 2.6% | 0.753 |
| 77 | `disagreement__cluster_representatives` | Disagreement-adaptive budget + Candidate-cluster representatives | 0.5288 | -0.0126 | -0.0174 | 33% | 2.3% | 0.729 |
| 78 | `uniform__head_tail` | Uniform per-query budget + Head–tail mixture | 0.5292 | -0.0122 | -0.0193 | 33% | 5.6% | 0.701 |
| 79 | `uniform__false_negative` | Uniform per-query budget + Likely false-negative hunting | 0.5306 | -0.0108 | -0.0152 | 0% | 3.2% | 0.636 |
| 80 | `uniform__center_outlier_mix` | Uniform per-query budget + Typical–outlier mixture | 0.5307 | -0.0107 | -0.0158 | 0% | 2.6% | 0.745 |
| 81 | `topic_balanced__head_tail` | Topic-balanced budget + Head–tail mixture | 0.5297 | -0.0117 | -0.0274 | 50% | 5.6% | 0.700 |
| 82 | `disagreement__committee` | Disagreement-adaptive budget + Bootstrap committee disagreement | 0.5289 | -0.0125 | -0.0159 | 17% | 1.8% | 0.664 |
| 83 | `difficulty__top_char` | Difficulty-adaptive budget + Character-robust selection | 0.5292 | -0.0122 | -0.0184 | 17% | 7.9% | 0.576 |
| 84 | `uniform__top_lsa` | Uniform per-query budget + Semantic-first selection | 0.5287 | -0.0127 | -0.0130 | 0% | 8.5% | 0.381 |
| 85 | `disagreement__top_char` | Disagreement-adaptive budget + Character-robust selection | 0.5294 | -0.0120 | -0.0199 | 17% | 7.8% | 0.575 |
| 86 | `difficulty__false_negative` | Difficulty-adaptive budget + Likely false-negative hunting | 0.5295 | -0.0119 | -0.0168 | 0% | 3.0% | 0.637 |
| 87 | `difficulty__committee` | Difficulty-adaptive budget + Bootstrap committee disagreement | 0.5270 | -0.0144 | -0.0194 | 33% | 2.0% | 0.663 |
| 88 | `uniform__committee` | Uniform per-query budget + Bootstrap committee disagreement | 0.5286 | -0.0128 | -0.0199 | 17% | 2.1% | 0.664 |
| 89 | `topic_balanced__top_char` | Topic-balanced budget + Character-robust selection | 0.5290 | -0.0124 | -0.0220 | 17% | 8.1% | 0.581 |
| 90 | `disagreement__false_negative` | Disagreement-adaptive budget + Likely false-negative hunting | 0.5269 | -0.0146 | -0.0159 | 17% | 2.9% | 0.635 |
| 91 | `difficulty__center_outlier_mix` | Difficulty-adaptive budget + Typical–outlier mixture | 0.5292 | -0.0122 | -0.0183 | 0% | 2.4% | 0.751 |
| 92 | `difficulty__top_lsa` | Difficulty-adaptive budget + Semantic-first selection | 0.5276 | -0.0138 | -0.0149 | 0% | 8.5% | 0.379 |
| 93 | `topic_balanced__top_lsa` | Topic-balanced budget + Semantic-first selection | 0.5277 | -0.0137 | -0.0152 | 0% | 8.8% | 0.379 |
| 94 | `topic_balanced__coverage` | Topic-balanced budget + Query-term coverage | 0.5282 | -0.0132 | -0.0228 | 17% | 7.2% | 0.577 |
| 95 | `uniform__top_char` | Uniform per-query budget + Character-robust selection | 0.5271 | -0.0143 | -0.0213 | 17% | 7.8% | 0.581 |
| 96 | `disagreement__top_lsa` | Disagreement-adaptive budget + Semantic-first selection | 0.5266 | -0.0148 | -0.0153 | 0% | 8.3% | 0.378 |
| 97 | `topic_balanced__committee` | Topic-balanced budget + Bootstrap committee disagreement | 0.5263 | -0.0151 | -0.0212 | 17% | 2.1% | 0.663 |
| 98 | `uniform__coverage` | Uniform per-query budget + Query-term coverage | 0.5273 | -0.0142 | -0.0195 | 0% | 7.1% | 0.581 |
| 99 | `difficulty__coverage` | Difficulty-adaptive budget + Query-term coverage | 0.5216 | -0.0198 | -0.0205 | 0% | 7.0% | 0.577 |
| 100 | `disagreement__coverage` | Disagreement-adaptive budget + Query-term coverage | 0.5199 | -0.0215 | -0.0217 | 0% | 6.8% | 0.579 |

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
