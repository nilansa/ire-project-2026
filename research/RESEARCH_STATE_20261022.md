# IRE Research State — 22 October 2026

**Team:** Ceous · **Task:** `ire-mid-checkpoint-20261022-001` · **Canonical local branch:** `ire-mid-20261022-report`, based on `main`. It contains the reviewed controls commit (`07750c0`, cherry-picked as `65ee1bb`) and literature commit (`0edc68b`, cherry-picked as `49747c4`). No remote push was made.

## Verified results

The historical training results below are verified against the project records; they were not rerun in this checkpoint. MRR@10 is the mean reciprocal rank of the first relevant passage within the first ten results. ANN Recall@200 is the fraction of the exact-search top-200 passage IDs recovered; it is not relevance recall. Candidate-set Jaccard is the count of shared IDs divided by the count in the union. Selected shared fraction is the per-query count of common selected IDs divided by the 20-ID selection budget, then averaged; it is neither Jaccard nor relevance recall.

| Run | Setup and recorded result |
|---|---|
| **MS MARCO, no teacher, 300 updates** | 50k-passage ANCE sanity pilot; known relevance labels plus selected unlabelled negatives; no cross-encoder. HNSW/IVF calibration ANN Recall@200: **0.955975/0.962250**. Their top-200 candidate Jaccard: **0.9173598779**; selected-20 shared fraction: **0.4344**. Exact/HNSW/IVF trained MRR@10: **0.826063/0.833257/0.850488**, all below untouched ANCE **0.890777**. One seed. |
| **MS MARCO, cross-encoder teacher, 100 updates** | Separate pilot; teacher jointly scores query and passage text, producing soft relevance targets for **32,312** query–passage pairs. HNSW/IVF MRR@10: **0.866675/0.855858**, both below untouched ANCE **0.890777**. The teacher's highest-scored passage matched across pools in **993/1,000** queries; mean aligned-distribution Jensen–Shannon divergence (a measure of probability-distribution difference) was **0.00523051**. One run; no backend or teacher benefit established. |
| **SciFact, separate teacher pilot** | Keep separate from both MS MARCO runs. Its manifest reports two duplicate query-text groups spanning train/test IDs: IDs differ, but text overlap creates a split-contamination concern. Repair and audit before interpreting generalization. |

## Synthetic selector control and claim boundary

The CPU-only control fixes 200 synthetic candidate IDs and changes only their order. A deterministic rank-position selector samples 20 fixed positions; across ten perturbation seeds its selected-ID shared fraction fell to **0.060** at full reorder. A stable-ID selector, which chooses the same IDs regardless of rank order, stayed at **1.000**. Across 1,000 pairs of independent uniform 20-of-200 samples, mean shared fraction was **0.09865**; the analytic expectation is **0.10** because each draw selects 20 of 200 IDs.

This demonstrates that rank-based selection can be sensitive to reordered candidates under this synthetic construction. It does **not** validate the historical **0.4344 (43.44%)** MS MARCO result: the historical selector source, query-to-seed mapping, and per-query candidate lists are unavailable. Do not claim that the synthetic control reproduced that result, that candidate identities differ semantically, or that any ANN backend is superior.

## Nearest prior work and independent Claude findings

The 11-paper source-backed survey identifies **ANCE** as the precedent for ANN-mined training negatives, **RocketQA** for cross-encoder filtering, and **SimANS** as the nearest precedent for rank-sensitive negative selection. These establish the component ideas. The narrower open question is whether the ANN backend changes selected teacher supervision and final retrieval when the retriever, selector, teacher, scoring budget, objective, and evaluation are fixed. The survey is not an exhaustive novelty search; make no “first” claim.

The independent Claude review read the research plan and the two pilot records, ran no code, and did not reproduce results. It independently emphasized that the pilots are distinct, both underperform untouched ANCE, and their HNSW/IVF ordering flips between runs. It also flagged that rank-position sampling may amplify small candidate-list changes, while one-seed variance, query provenance, false negatives, and training details remain unresolved. For the teacher run, the 993/1,000 top-passage match and low distribution divergence weaken any claim that the different IDs created substantially different teacher targets.

## Blockers

- Historical selector replay is blocked until its source, query-to-seed mapping, and per-query candidate lists are recovered.
- There is no repeat-seed estimate for either MS MARCO training pilot, and neither trained result beats untouched ANCE.
- SciFact cross-split duplicate query text must be resolved before making a generalization claim.

## Next three experiments and STOP criteria

1. **Historical selector replay.** Recover the exact selector, mapping, and candidate lists; replay the recorded overlap. **STOP** if any required input is missing. Keep the synthetic result labeled as a mechanism demonstration.
2. **Split and positive-control audit.** Verify query text is disjoint across train, calibration, and evaluation; reproduce untouched-checkpoint evaluation; calibrate the teacher and establish a training recipe that beats untouched ANCE in three seeds. **STOP** if leakage remains, baseline evaluation is not reproducible, or the positive control fails in any of the three seeds. Do not start backend comparisons before this gate passes.
3. **Paired backend comparison.** With a fixed retriever, teacher, selector, per-query scoring budget, objective, and exact evaluation, compare exact/HNSW/IVF at matched calibration Recall@200 over at least three paired seeds. Include a shared-supervision replay, which feeds identical saved query–passage pairs and teacher scores to each nominal branch. **STOP** attribution if replay outputs differ; report a null backend result if observed differences do not exceed the paired-seed variability.

These are plans, not completed experiments. No training or GPU job is launched by this note.

## Reproduction

From the repository root, run the five unit tests and compare fresh CSV output byte-for-byte with the committed result:

```sh
python3 -m unittest discover -s experiments/selector-rank-falsification -p 'test_*.py' -v
./experiments/selector-rank-falsification/run.py | cmp - experiments/selector-rank-falsification/results.csv
```

## Deadlines

- **31 October 2026:** one-page mid-checkpoint progress report.
- **15 November 2026:** final written report and video.

## Reconcile and deduplicate next runs

Continue from local branch `ire-mid-20261022-report` for task `ire-mid-checkpoint-20261022-001`. For each run, record its dataset and split, starting checkpoint, teacher, candidate backend, selector and budget, seed, objective, and output path. Deduplicate repeated query–passage rows only within the same run and configuration; preserve different seeds, backends, training regimes, and datasets as distinct results. Reuse an existing artifact only when its identity and bytes match; reconcile report claims against the original records before editing the checkpoint. Do not rewrite `RESEARCH_PLAN.md` or merge SciFact metrics into MS MARCO.

## Original records

- [Original research plan](../RESEARCH_PLAN.md)
- [MS MARCO 300-update no-teacher record](../survey/experiments/02-ann-backend-negative-selection-sanity.md)
- [MS MARCO 100-update cross-encoder record](../experiments/ann-micro-pilot-20260907/RESULTS.md)
- [SciFact split caveat in the project evidence page](../docs/index.html)
- [Synthetic control code and results](../experiments/selector-rank-falsification/RESULTS.md)
- [11-paper prior-work survey](../survey/docs/IRE_MID_2026_NEAREST_PRIOR_WORK.md)
- [Checkpoint report draft](../reports/IRE_MID_CHECKPOINT_DRAFT.md)
- [Independent Claude review](</Users/nilan/.codex/scratch/cross-surface/ire-project/ire-mid-checkpoint-20261022-001/codex-to-chatgpt/claude-initial-review.md>)

**Do not claim a named external memory agent or ChatGPT Memory was updated.**
