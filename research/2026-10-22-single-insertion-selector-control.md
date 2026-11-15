# IRE Team Ceous — one-insertion selector sensitivity control (2026-10-22)

**Task:** `ire-mid-checkpoint-20261022-001`. **Local branch:** `ire-mid-20261022-report`. **Status:** completed deterministic CPU-only synthetic check; not a historical replay, ANN benchmark, or retriever-training run.

## One narrowly targeted falsifier

The existing random-rank permutation control holds all 200 candidate identities fixed. This additional control tests a distinct, more constrained failure mode: insert one new candidate at the top of a 200-element ranked list and remove the old rank-200 candidate. Thus **199/200 IDs remain unchanged** (candidate-set Jaccard **199/201 = 0.990049751**), but all retained candidates move down one rank.

Apply the existing `select_rank_positions` and seeded, rank-invariant `select_stable_ids` implementations in `experiments/selector-rank-falsification/run.py`, choosing 20 passages per query. The original ordered IDs are `D001` through `D200`; the changed list is `NEW, D001, ..., D199`. No real ANN search, teacher calls, score perturbations, or GPU were used.

| Control | Fixed seed 20261022: shared selected/20 | Ten additional fixed seeds: average shared selected/20 |
|---|---:|---:|
| Fixed rank-position selector | 0.10 | 0.065 |
| Hash-of-ID selector | 1.00 | 0.995 |

The ten additional seeds were `101, 211, 307, 401, 503, 601, 701, 809, 907, 1009` and use the same candidate insertion, not new stochastic ANN builds. They were explicitly printed by the local Python check. A unit regression test asserts 199 shared IDs and a large selector sensitivity difference without hard-coding one runtime's exact sampled ranks.

**Interpretation:** Very high candidate-set overlap is compatible with very low selected-pool overlap under a seeded *rank-position* selector. One candidate insertion and consequent positional shift are sufficient in this artificial example; therefore a 43.44% historical selected-pool overlap cannot be attributed uniquely to the search backend without replaying the original selector, ranks, and query seeds. The example does **not** establish how frequently top insertions occur in the MS MARCO/HNSW/IVF data, prove one backend is better, or change either pilot's final MRR.

**Reproduction, from repo root:** `python3 -m unittest discover -s experiments/selector-rank-falsification -p 'test_*.py' -v`. The test named `test_one_new_top_candidate_can_shift_rank_selector` is the new control. No historical large experiment was rerun. Next real-data gate remains recovery of the original per-query candidate lists, position-selector source, and query-to-seed map; stop without them.

**Provenance:** Existing `research/RESEARCH_STATE_20261022.md`, existing synthetic test code, and the previously recorded independent Claude review (which recommended a selector swap and pointed out rank shifts after an insertion). Scratch and Codex's transfer index were read but **not edited**, because no cross-surface artifact transfer was performed. Nothing was pushed or published.
