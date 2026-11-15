#!/usr/bin/env python3
"""Exact overlap expectation for two hypothetical positive-exclusion orders.

This analytically extends positive_exclusion_order_control.py. It does not
replay any historical MS MARCO candidate list or identify the actual selector.
"""

from fractions import Fraction
from itertools import permutations
from math import factorial


def expected_shared_count(n: int, k: int, positive_position: int) -> Fraction:
    """Expectation for one shared uniform rank permutation (zero-based positive)."""
    if not (n >= 3 and 1 <= k <= n - 1 and 0 <= positive_position < n):
        raise ValueError("need n>=3, 1<=k<n, and a valid positive position")
    if positive_position == n - 1:
        return Fraction(k)  # both rules act on precisely the same ranked IDs

    # For an ID above the positive, both exclusion rules use its same rank.
    same_rank = Fraction(k, n) + Fraction(k * (k - 1), n * (n - 1) * (n - 2))
    # For an ID below the positive, filtering first shifts its sampled rank.
    shifted = Fraction(k * (k - 1), n * (n - 1)) + Fraction(
        2 * k * (k - 1), n * (n - 1) * (n - 2)
    )
    # Two boundary cases: the last rank, and the first rank below the positive.
    boundary = Fraction(k * (n - k - 1), n * (n - 1) * (n - 2))
    return positive_position * same_rank + (n - 1 - positive_position) * shifted + 2 * boundary


def exhaustive_shared_count(n: int, k: int, positive_position: int) -> Fraction:
    """Tiny-n independent verification of the combinatorial formula."""
    total = 0
    for order in permutations(range(n)):
        before = [
            (rank if rank < positive_position else rank + 1)
            for rank in order
            if rank < n - 1
        ][:k]
        after = [rank for rank in order if rank != positive_position][:k]
        total += len(set(before) & set(after))
    return Fraction(total, factorial(n))


def main() -> None:
    for n, k in ((5, 2), (6, 2), (6, 3)):
        for positive_position in range(n):
            assert expected_shared_count(n, k, positive_position) == exhaustive_shared_count(
                n, k, positive_position
            ), (n, k, positive_position)
    print("Exhaustive small-n formula verification: PASS")
    print("positive_rank,exact_expected_shared_fraction")
    for one_based_rank in (1, 50, 100, 150, 199, 200):
        fraction = expected_shared_count(200, 20, one_based_rank - 1) / 20
        print(f"{one_based_rank},{float(fraction):.9f}")


if __name__ == "__main__":
    main()
