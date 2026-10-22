# Fixed-candidate rank-selector control — synthetic results

**Status:** completed CPU-only mechanism demonstration; not a replay of historical query lists.

## Question and method

Can changing only the order of the same 200 candidate IDs change the 20 IDs selected by the documented deterministic rank-position rule? The run uses IDs `D001`–`D200`, budget 20, query seed `20261022`, and ten fixed perturbation seeds (`101, 211, 307, 401, 503, 601, 701, 809, 907, 1009`). At each strength, that fraction of positions is sampled and shuffled; the selected rank positions stay fixed. Results average selected-set intersection divided by 20. A stable ID control uses a seeded SHA-256 ordering, and the random control averages 1,000 pairs of independent uniform 20-of-200 identity samples.

| Perturbation strength | Mean ranks changed | Rank-position selected shared fraction | Stable-ID shared fraction |
|---:|---:|---:|---:|
| 0.00 (identity baseline) | 0.000 | 1.000 | 1.000 |
| 0.05 | 0.046 | 0.965 | 1.000 |
| 0.10 | 0.095 | 0.940 | 1.000 |
| 0.25 | 0.249 | 0.790 | 1.000 |
| 0.50 | 0.496 | 0.535 | 1.000 |
| 0.75 | 0.743 | 0.355 | 1.000 |
| 1.00 | 0.995 | 0.060 | 1.000 |

Across 1,000 independent random-selection pairs, shared fraction was **0.09865**; the analytic expectation is `20/200 = 0.10`.

## Interpretation and provenance

In this synthetic construction, order-only perturbation can reduce rank-position selected overlap, while stable ID selection is invariant. **This synthetic demonstration is NOT experimental validation of the real 43.44% result.** The checkout has the written description of the historical selector but not its executable source, query-to-seed mapping, or per-query candidate lists. `docs/index.html` contains an illustrative JavaScript toy, not the historical selector; it was not treated as source data. The result establishes only that the documented mechanism can be rank-sensitive under controlled reorderings.

The `documented_rank_position_rule` reconstruction samples a fixed set of ranks with a seeded standard-library RNG. Its seed and perturbation scheme are explicit here; they are not claimed to match the original run. No model, dataset, teacher, GPU, or external package was used.

## Reproduction

From the controls worktree:

```sh
./experiments/selector-rank-falsification/run.py > experiments/selector-rank-falsification/results.csv
python3 -m unittest discover -s experiments/selector-rank-falsification -p 'test_*.py' -v
```

The run writes the CSV table at `experiments/selector-rank-falsification/results.csv`. Five unit tests pass for identity overlap, deterministic rank selection, preserved IDs under reordering, rank-invariant ID selection, random sampling, and invalid-input handling.
