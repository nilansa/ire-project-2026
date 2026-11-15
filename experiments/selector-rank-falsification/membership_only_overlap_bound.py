#!/usr/bin/env python3
"""Exact bound for a fixed-ID-priority top-k selector (not historical replay).

For equally sized N-ID candidate pools A,B, let d=|A\\B|=|B\\A|.
A deterministic top-k rule using the SAME total order over passage IDs
in both arms has selected shared fraction at least max(0, 1-d/k).
As candidate Jaccard J=(N-d)/(N+d), this implies the sharper-to-average
linear bound overlap >= 1 - (N+k)/(2*k) * (1-J), even if d varies by query.
Neither this hypothetical selection rule nor the historic per-query inputs
have been recovered, so the calculation is not a Ceous experimental result.
"""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from itertools import combinations


N = 200
K = 20
REPORTED_MEAN_JACCARD = Decimal("0.9173598779")
REPORTED_MEAN_SELECTED_SHARED = Decimal("0.4344")


def lower_bound_from_mean_jaccard(mean_jaccard: Fraction, n: int, k: int) -> Fraction:
    if not 0 < k <= n or not 0 <= mean_jaccard <= 1:
        raise ValueError("requires 0 < k <= n and 0 <= mean_jaccard <= 1")
    return max(Fraction(0), 1 - Fraction(n + k, 2 * k) * (1 - mean_jaccard))


def check_exact_bounds(n: int, k: int) -> None:
    """Algebraic inequality over every possible candidate difference count."""
    for d in range(n + 1):
        jaccard = Fraction(n - d, n + d)
        per_query_floor = max(Fraction(0), 1 - Fraction(d, k))
        linear_floor = 1 - Fraction(n + k, 2 * k) * (1 - jaccard)
        assert per_query_floor >= linear_floor, (n, k, d)


def check_exhaustive_small_sets() -> None:
    """Test fixed-ID-order top-k against all pairs of five-ID toy pools."""
    n, k = 5, 2
    universe = range(8)
    pools = list(combinations(universe, n))
    for a in pools:
        for b in pools:
            d = n - len(set(a) & set(b))
            selected_overlap = Fraction(len(set(a[:k]) & set(b[:k])), k)
            assert selected_overlap >= max(Fraction(0), 1 - Fraction(d, k))


def main() -> None:
    check_exact_bounds(N, K)
    check_exhaustive_small_sets()
    j = Fraction(REPORTED_MEAN_JACCARD)
    observed = Fraction(REPORTED_MEAN_SELECTED_SHARED)
    floor = lower_bound_from_mean_jaccard(j, N, K)
    assert floor > observed
    print(f"candidate_count={N}, selection_budget={K}")
    print(f"reported_candidate_jaccard={float(j):.10f}")
    print(f"stable_priority_minimum_mean_selected_shared={float(floor):.10f}")
    print(f"reported_selected_shared={float(observed):.10f}")
    print(f"gap_percentage_points={float((floor - observed) * 100):.6f}")
    print("checks=PASS (all d=0..200 and exhaustive toy pairs)")
    print("conditional_on=same_queries, equal_N, same_fixed_ID_priority, fixed_k")


if __name__ == "__main__":
    main()
