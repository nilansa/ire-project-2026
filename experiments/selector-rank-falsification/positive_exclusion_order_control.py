#!/usr/bin/env python3
"""Synthetic selector-parity probe; not a replay of MS MARCO experiments."""

import random
from statistics import mean


IDS = tuple(f"D{i:03d}" for i in range(1, 201))
SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)
BUDGET = 20


def overlap_for_positive_rank(rank, seed):
    """Compare positive filtering before and after one *shared* rank permutation.

    Pre: filter the ranked list first, then apply valid permutation positions.
    Post: apply positions to the original list, skip the positive, and refill
    from later positions of that same permutation. The random permutation is
    exactly the same in both branches; only exclusion order changes.
    """
    positive = IDS[rank - 1]
    eligible = tuple(pid for pid in IDS if pid != positive)
    positions = list(range(len(IDS)))
    random.Random(seed).shuffle(positions)

    pre = tuple(eligible[i] for i in positions if i < len(eligible))[:BUDGET]
    post = tuple(IDS[i] for i in positions if IDS[i] != positive)[:BUDGET]
    assert len(set(pre)) == len(set(post)) == BUDGET
    assert positive not in pre and positive not in post
    return len(set(pre) & set(post)) / BUDGET


if __name__ == "__main__":
    print("positive_rank,mean_shared_selected_fraction,seeds")
    for rank in (1, 100, 200):
        overlaps = [overlap_for_positive_rank(rank, seed) for seed in SEEDS]
        print(f"{rank},{mean(overlaps):.3f},{len(SEEDS)}")
        if rank == 200:
            assert all(overlap == 1.0 for overlap in overlaps)
