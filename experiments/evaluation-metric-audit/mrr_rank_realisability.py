#!/usr/bin/env python3
"""CPU-only constructive feasibility audit of printed 500-query MRR@10 values.

Does not load original passage ranks, qrels, models, or evaluation code.
The synthetic witnesses are mathematical possibilities, NOT observed ranks.
"""

from collections import Counter
from decimal import Decimal
from math import lcm
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
N_QUERIES = 500
DENOM = lcm(*range(1, 11))
SCALE = N_QUERIES * DENOM

ROWS = (
    ("no-teacher", "untouched", "0.8907769841", "survey/experiments/02-ann-backend-negative-selection-sanity.md"),
    ("no-teacher", "exact", "0.8260634921", "survey/experiments/02-ann-backend-negative-selection-sanity.md"),
    ("no-teacher", "HNSW", "0.8332571429", "survey/experiments/02-ann-backend-negative-selection-sanity.md"),
    ("no-teacher", "IVF", "0.8504880952", "survey/experiments/02-ann-backend-negative-selection-sanity.md"),
    ("teacher", "untouched", "0.890777", "experiments/ann-micro-pilot-20260907/RESULTS.md"),
    ("teacher", "HNSW", "0.866675", "experiments/ann-micro-pilot-20260907/RESULTS.md"),
    ("teacher", "IVF", "0.855858", "experiments/ann-micro-pilot-20260907/RESULTS.md"),
)


def rounded_numerators(printed: str):
    observed = Decimal(printed)
    tolerance = Decimal(5).scaleb(-len(printed.split(".")[1]) - 1)
    near = int(observed * SCALE)
    return [
        n
        for n in range(max(0, near - 3), min(SCALE, near + 3) + 1)
        if abs(Decimal(n) / SCALE - observed) <= tolerance
    ]


def main():
    # One non-rank-one query contributes its deficit from a rank-one query.
    # Rank 0 means no relevant passage in the top ten.
    moves = [(rank, DENOM - DENOM // rank) for rank in range(2, 11)]
    moves += [(0, DENOM)]
    alternatives = [rounded_numerators(value) for _, _, value, _ in ROWS]
    assert all(alternatives)
    largest_deficit = max(SCALE - min(options) for options in alternatives)

    # Unbounded-coin feasibility on deficits. Any representation at these
    # reported MRR values uses fewer than 500 nonzero-deficit queries:
    # largest_deficit / min_positive_deficit < 500.
    assert largest_deficit // min(step for _, step in moves) < N_QUERIES
    parent = [None] * (largest_deficit + 1)
    parent[0] = (0, 1)
    for deficit in range(1, largest_deficit + 1):
        for rank, increment in moves:
            if deficit >= increment and parent[deficit - increment] is not None:
                parent[deficit] = (deficit - increment, rank)
                break

    for pilot, arm, printed, source in ROWS:
        assert printed in (ROOT / source).read_text(), (pilot, arm, printed)
        feasible = []
        for numerator in rounded_numerators(printed):
            deficit = SCALE - numerator
            if parent[deficit] is None:
                continue
            ranks = []
            while deficit:
                prior, rank = parent[deficit]
                ranks.append(rank)
                deficit = prior
            if len(ranks) > N_QUERIES:
                continue
            ranks.extend([1] * (N_QUERIES - len(ranks)))
            witness = Counter(ranks)
            assert sum(witness.values()) == N_QUERIES
            assert sum(count * (DENOM // rank if rank else 0) for rank, count in witness.items()) == numerator
            feasible.append((numerator, witness))
        assert feasible, (pilot, arm, printed)
        numerator, witness = feasible[0]
        compact = ", ".join(f"r{rank}:{count}" for rank, count in sorted(witness.items()))
        print(f"PASS {pilot} {arm} {printed} numerator={numerator} 500-query synthetic witness: {compact}")


if __name__ == "__main__":
    main()
