#!/usr/bin/env python3
"""Synthetic control for rank-sensitive selection on one fixed candidate set."""

from __future__ import annotations

import csv
import hashlib
import math
import random
import sys
from statistics import mean
from typing import Sequence


CANDIDATE_COUNT = 200
BUDGET = 20
QUERY_SEED = 20261022
PERTURBATION_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)
STRENGTHS = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
RANDOM_CONTROL_REPLICATES = 1000


def _validate(candidates: Sequence[str], budget: int) -> None:
    if len(candidates) != len(set(candidates)):
        raise ValueError("candidate IDs must be unique")
    if not 0 < budget <= len(candidates):
        raise ValueError("budget must be between 1 and the candidate count")


def select_rank_positions(
    ranked_ids: Sequence[str], budget: int, query_seed: int
) -> tuple[str, ...]:
    """Select a fixed, deterministic sample of ranks for one query."""
    _validate(ranked_ids, budget)
    positions = random.Random(query_seed).sample(range(len(ranked_ids)), budget)
    return tuple(ranked_ids[position] for position in positions)


def perturb_ranking(
    ranked_ids: Sequence[str], strength: float, seed: int
) -> tuple[str, ...]:
    """Shuffle a seeded fraction of rank positions while preserving all IDs."""
    _validate(ranked_ids, len(ranked_ids))
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")

    result = list(ranked_ids)
    count = round(len(result) * strength)
    rng = random.Random(seed)
    positions = rng.sample(range(len(result)), count)
    values = [result[position] for position in positions]
    rng.shuffle(values)
    for position, value in zip(positions, values):
        result[position] = value
    return tuple(result)


def select_stable_ids(
    candidate_ids: Sequence[str], budget: int, query_seed: int
) -> tuple[str, ...]:
    """Select a query-specific set by ID, independent of candidate rank."""
    _validate(candidate_ids, budget)
    prefix = f"{query_seed}:".encode()
    return tuple(
        sorted(
            candidate_ids,
            key=lambda item: hashlib.sha256(prefix + item.encode()).digest(),
        )[:budget]
    )


def select_random_ids(
    candidate_ids: Sequence[str], budget: int, seed: int
) -> tuple[str, ...]:
    """Select a uniform identity-based sample, independent of candidate rank."""
    _validate(candidate_ids, budget)
    return tuple(random.Random(seed).sample(list(candidate_ids), budget))


def shared_fraction(left: Sequence[str], right: Sequence[str], budget: int) -> float:
    return len(set(left).intersection(right)) / budget


def build_rows() -> list[dict[str, str]]:
    candidates = tuple(f"D{index:03}" for index in range(1, CANDIDATE_COUNT + 1))
    original_rank_selection = select_rank_positions(candidates, BUDGET, QUERY_SEED)
    stable_selection = select_stable_ids(candidates, BUDGET, QUERY_SEED)
    rows: list[dict[str, str]] = []

    def add_row(
        control: str,
        strength: str,
        replicates: int,
        changed_ranks: float,
        overlap: float,
        reference: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "control": control,
                "rank_perturbation_strength": strength,
                "replicates": str(replicates),
                "mean_fraction_of_ranks_changed": f"{changed_ranks:.6f}",
                "mean_selected_20_shared_fraction": f"{overlap:.6f}",
                "analytic_reference": reference,
                "notes": notes,
            }
        )

    add_row(
        "identity_baseline",
        "0.000000",
        1,
        0.0,
        1.0,
        "1.000000",
        "Same ordered list compared with itself",
    )

    for strength_index, strength in enumerate(STRENGTHS):
        rank_changes = []
        overlaps = []
        stable_overlaps = []
        for seed in PERTURBATION_SEEDS:
            perturbed = perturb_ranking(
                candidates, strength, seed + strength_index * 100_003
            )
            rank_changes.append(
                sum(a != b for a, b in zip(candidates, perturbed)) / CANDIDATE_COUNT
            )
            selected = select_rank_positions(perturbed, BUDGET, QUERY_SEED)
            overlaps.append(
                shared_fraction(original_rank_selection, selected, BUDGET)
            )
            stable = select_stable_ids(perturbed, BUDGET, QUERY_SEED)
            stable_overlaps.append(shared_fraction(stable_selection, stable, BUDGET))

        add_row(
            "documented_rank_position_rule",
            f"{strength:.6f}",
            len(PERTURBATION_SEEDS),
            mean(rank_changes),
            mean(overlaps),
            "1.000000 at strength 0",
            "Fixed query seed; same candidate IDs; rank positions are held fixed",
        )
        add_row(
            "stable_id_rank_invariant_control",
            f"{strength:.6f}",
            len(PERTURBATION_SEEDS),
            mean(rank_changes),
            mean(stable_overlaps),
            "1.000000",
            "Select the same query-specific IDs regardless of rank order",
        )

    random_overlaps = []
    for replicate in range(RANDOM_CONTROL_REPLICATES):
        left = select_random_ids(candidates, BUDGET, 2 * replicate + 17)
        right = select_random_ids(candidates, BUDGET, 2 * replicate + 18)
        random_overlaps.append(shared_fraction(left, right, BUDGET))
    add_row(
        "independent_random_20_of_200",
        "not_applicable",
        RANDOM_CONTROL_REPLICATES,
        0.0,
        mean(random_overlaps),
        f"{BUDGET / CANDIDATE_COUNT:.6f}",
        "Two independent identity-uniform samples; expected shared fraction is k/N",
    )
    return rows


def main() -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=(
            "control",
            "rank_perturbation_strength",
            "replicates",
            "mean_fraction_of_ranks_changed",
            "mean_selected_20_shared_fraction",
            "analytic_reference",
            "notes",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(build_rows())


if __name__ == "__main__":
    main()
