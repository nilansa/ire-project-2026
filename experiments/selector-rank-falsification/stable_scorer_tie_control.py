#!/usr/bin/env python3
"""CPU-only synthetic audit: deterministic scorer, seed, and tie-breaking.

This is NOT a replay of historical MS MARCO candidate rankings or teacher scores.
"""

from __future__ import annotations

from statistics import mean

from run import select_rank_positions, shared_fraction


IDS = tuple(f"D{i:03d}" for i in range(1, 201))
BUDGET = 20
SEEDS = (73, 101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)


def score_order(
    ids: tuple[str, ...], scores: dict[str, int], explicit_id_tie_break: bool
) -> tuple[str, ...]:
    if explicit_id_tie_break:
        return tuple(sorted(ids, key=lambda pid: (-scores[pid], pid)))
    # Python's stable sort preserves the *input order* of tied passages.
    return tuple(sorted(ids, key=lambda pid: -scores[pid]))


def evaluate(tie_group_size: int, explicit_id_tie_break: bool) -> float:
    # Same candidate identities and frozen integer scores in BOTH arms.
    scores = {pid: (index // tie_group_size) for index, pid in enumerate(IDS)}
    forward = score_order(IDS, scores, explicit_id_tie_break)
    reverse = score_order(IDS[::-1], scores, explicit_id_tie_break)
    return mean(
        shared_fraction(
            select_rank_positions(forward, BUDGET, seed),
            select_rank_positions(reverse, BUDGET, seed),
            BUDGET,
        )
        for seed in SEEDS
    )


def main() -> None:
    print("tie_group_size,input_order_ties,id_tie_break,replicates")
    for tie_group_size in (1, 10, 200):
        order_only = evaluate(tie_group_size, False)
        explicit_id = evaluate(tie_group_size, True)
        assert explicit_id == 1.0, "explicit ID tie break must be rank-invariant"
        if tie_group_size == 1:
            assert order_only == 1.0, "unique scores must be rank-invariant"
        else:
            assert order_only < 1.0, "tied scores should expose input-order sensitivity"
        print(f"{tie_group_size},{order_only:.6f},{explicit_id:.6f},{len(SEEDS)}")


if __name__ == "__main__":
    main()
