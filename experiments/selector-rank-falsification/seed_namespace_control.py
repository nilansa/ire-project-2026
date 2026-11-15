#!/usr/bin/env python3
"""Synthetic check: do backend-specific seed namespaces change a fixed selector?

This tests the *reconstructed* rank-position selector, not the missing original
MS MARCO selector or its query-to-seed implementation. No retrieval is performed.
"""

from __future__ import annotations

import hashlib
from statistics import mean

from run import select_rank_positions, shared_fraction


N_QUERIES = 1000
N_CANDIDATES = 200
BUDGET = 20


def seed_for(query_id: str, namespace: str = "") -> int:
    """Stable across Python processes; deliberately permits a seed namespace."""
    key = f"seed-namespace-control-v1\0{namespace}\0{query_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def main() -> None:
    ranking = tuple(f"D{i:03d}" for i in range(N_CANDIDATES))
    paired = []
    namespace_mismatch = []
    for index in range(N_QUERIES):
        query_id = f"synthetic-Q{index:04d}"
        base = select_rank_positions(ranking, BUDGET, seed_for(query_id))
        matched = select_rank_positions(ranking, BUDGET, seed_for(query_id))
        hnsw_seeded = select_rank_positions(
            ranking, BUDGET, seed_for(query_id, namespace="HNSW")
        )
        ivf_seeded = select_rank_positions(
            ranking, BUDGET, seed_for(query_id, namespace="IVF")
        )
        paired.append(shared_fraction(base, matched, BUDGET))
        namespace_mismatch.append(shared_fraction(hnsw_seeded, ivf_seeded, BUDGET))

    same_seed = mean(paired)
    different_namespace = mean(namespace_mismatch)
    independent_seed_expectation = BUDGET / N_CANDIDATES
    assert same_seed == 1.0
    # Hash-derived seeds are deterministic pseudorandom values, not a theorem
    # about independence; this loose bound is a reproducibility sanity check.
    assert abs(different_namespace - independent_seed_expectation) < 0.03
    assert 0.0 <= different_namespace <= 1.0

    print(f"queries: {N_QUERIES}; candidates: {N_CANDIDATES}; budget: {BUDGET}")
    print(f"shared fraction, same query-only seed: {same_seed:.6f}")
    print(f"shared fraction, backend-namespaced seeds: {different_namespace:.6f}")
    print(f"independent-uniform 20-of-200 expectation: {independent_seed_expectation:.6f}")
    print("PASS: same IDs/ranks and selector; only seed derivation differs")


if __name__ == "__main__":
    main()
