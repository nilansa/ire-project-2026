# IRE Team Ceous — seeded identity-sampler invariance control (2026-10-22)

**Task:** `ire-mid-checkpoint-20261022-001`. **Branch:** `ire-mid-20261022-report`. **Scope:** one CPU-only selector correctness check; no ANN retrieval, teacher calls, model training, or GPU.

## What was falsified and corrected

The synthetic script described `select_random_ids(candidate_ids, budget, seed)` as independent of candidate ranking, but it passed the *input sequence* directly to `random.Random(seed).sample`. The same fixed seed therefore selected different identities when the same 200 IDs were supplied in reverse order. Before correction, seed 73 yielded **0.10** selected-20 overlap between the original and reversed lists; the existing hash-of-ID control yielded **1.00**. This violated the identity sampler's stated rank-invariance contract. It does not invalidate the independent-random baseline, which used the same fixed sorted list on both draws.

Fix: sort candidate identities before seeded uniform sampling. This changes the implementation only, not the input pool or selection budget. The new regression checks that the same seed and the same IDs give identical selected identities under reversed rank order. After correction, the overlap is **1.00**. For the existing `D001`–`D200` baseline (already sorted), sampled identities and the committed CSV remain unchanged.

## Limits and next gate

This detects and removes one synthetic *selector/seed implementation confound*; it does not reproduce the historical HNSW–IVF 43.44% selected overlap or prove a backend effect. The historical 200-ID candidate lists, selector source, and query-to-seed mapping remain unavailable in the checked-out pilot records. Do not compare ANN backends or spend teacher/GPU budget until actual selector replay and a clean positive control are possible. Keep MS MARCO/ANCE and SciFact pilots separate.

**Reproduce from repository root:**

```sh
python3 -m unittest discover -s experiments/selector-rank-falsification -p 'test_*.py' -v
./experiments/selector-rank-falsification/run.py | cmp - experiments/selector-rank-falsification/results.csv
git diff --check
```

The historical evidence, original synthetic CSV, and Codex-owned `TRANSFER_INDEX.json` were not edited. No scratch was synced, no Git branch was overwritten, and nothing was pushed.
