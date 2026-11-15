#!/usr/bin/env python3
"""Exact combinatorial baseline for the existing one-top-insertion selector control.

No historical candidate lists, models, teacher scores, ANN runs, or stochastic
simulation are used. This describes uniformly sampled *rank positions*, not
the missing historical selector implementation.
"""

from collections import Counter
from fractions import Fraction
from itertools import combinations
from math import comb


def exact_adjacent_overlap_distribution(n: int, k: int) -> dict[int, Fraction]:
    """P(j shared IDs) under a one-place top insertion and shared k-of-n ranks.

    Original ranking: A1 ... An.
    Changed ranking: NEW A1 ... A(n-1).
    The shared chosen IDs are precisely adjacent selected rank-position pairs.
    For j adjacent pairs, the k selected positions form r=k-j contiguous runs.
    There are C(k-1,r-1) run lengths and C(n-k+1,r) run placements.
    """
    if not (1 <= k <= n):
        raise ValueError("require 1 <= k <= n")
    return {
        j: Fraction(comb(k - 1, k - j - 1) * comb(n - k + 1, k - j), comb(n, k))
        for j in range(k)
    }


def exhaustive_distribution(n: int, k: int) -> dict[int, Fraction]:
    """Enumerate only tiny cases to independently test the formula."""
    counts = Counter(
        sum(right == left + 1 for left, right in zip(s, s[1:]))
        for s in combinations(range(n), k)
    )
    return {j: Fraction(counts[j], comb(n, k)) for j in range(k)}


def main() -> None:
    for n, k in ((5, 2), (6, 3), (8, 3)):
        assert exact_adjacent_overlap_distribution(n, k) == exhaustive_distribution(n, k)

    n, k = 200, 20
    distribution = exact_adjacent_overlap_distribution(n, k)
    assert sum(distribution.values()) == 1
    mean_shared = sum(j * p for j, p in distribution.items())
    assert mean_shared == Fraction(k * (k - 1), n) == Fraction(19, 10)
    assert distribution[0] == Fraction(comb(181, 20), comb(200, 20))

    print(f"N={n}, selected={k}; all exact-rational checks PASS")
    print(f"expected shared IDs = {mean_shared} = {float(mean_shared):.8f}")
    print(f"expected shared fraction = {mean_shared / k} = {float(mean_shared / k):.8f}")
    print(f"probability of zero shared IDs = {float(distribution[0]):.8f}")


if __name__ == "__main__":
    main()
