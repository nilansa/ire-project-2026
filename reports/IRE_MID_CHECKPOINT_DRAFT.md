# IRE Mid-Checkpoint Progress Report — Draft

**Team Ceous · 22 October 2026 · Draft for the 31 October checkpoint**

## Completed evidence

This project asks whether approximate-nearest-neighbor (ANN) candidate generation changes which query–passage pairs receive limited relevance feedback and whether that changes the retriever after post-training. Evidence remains diagnostic; no index is shown to be better.

In the **50k-passage MS MARCO ANCE sanity pilot**, HNSW and IVF-Flat recovered 0.955975 and 0.962250 of exact top-200 neighbors; candidate Jaccard was 0.91736. The deterministic rank-position rule selected 20 passages per query and the selected pools shared 0.4344 of IDs (intersection divided by 20, not Jaccard). This pilot used qrel positives and unlabelled negatives, **no cross-encoder teacher**, and 300 updates. All trained branches fell below untouched ANCE MRR@10 (0.890777).

A **separate 100-update MS MARCO teacher pilot** scored 32,312 unique query–passage pairs with a cross-encoder. HNSW and IVF student MRR@10 were 0.866675 and 0.855858, below untouched ANCE (0.890777); teacher distributions concentrated similarly on known positives. This does not establish a teacher benefit or backend effect. The **separate SciFact teacher pilot** is not pooled with these MS MARCO results and has documented duplicate-query split issues.

## Completed synthetic control

The cheapest control is completed: hold 200 IDs fixed and perturb only their order. At full perturbation, rank-position selected overlap averaged 0.060 across ten fixed seeds; the stable-ID control stayed at 1.000. Independent random 20-of-200 pairs averaged 0.09865 (analytic expectation 0.10); the identity baseline was 1.000. The run is CPU-only, standard-library Python.

**Falsifiable hypothesis:** rank-only perturbations lower shared-ID overlap under the documented position rule, while stable-ID selection remains invariant. The synthetic run exhibits this mechanism, but **is NOT experimental validation of the real 0.4344 result**: the historical selector source, query-to-seed mapping, and per-query lists are absent.

## Proposed work, limitations and schedule

Both MS MARCO training pilots underperformed untouched ANCE; neither supports a downstream-improvement claim. The synthetic result is a mechanism demonstration only. Next, replay the control on historical per-query lists if recovered; before training, establish a working baseline, fixed feedback budget, paired seeds, clean splits, and exact-search evaluation. Treat SciFact split integrity separately.

**Schedule:** 22–24 Oct, review the completed selector control and locate (or mark unavailable) historical lists/source; by **31 Oct**, finalize the one-page progress report using verified evidence. 1–7 Nov, agree on the next design and split checks; 8–14 Nov, prepare the final written report and video; **15 Nov**, deliver both.

**Sources:** `survey/experiments/02-ann-backend-negative-selection-sanity.md`; `experiments/ann-micro-pilot-20260907/RESULTS.md`; separate SciFact summary in `docs/index.html`. The original `RESEARCH_PLAN.md` was not changed.
