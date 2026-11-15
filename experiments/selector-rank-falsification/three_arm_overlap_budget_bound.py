#!/usr/bin/env python3
"""CPU-only aggregate bound on reusing 20-ID selection slates across three arms.

Uses only published no-teacher MS MARCO/ANCE aggregate pairwise overlaps.
It does not recover query-level IDs, teacher calls, or historical code.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "survey/experiments/02-ann-backend-negative-selection-sanity.md"
)
QUERY_COUNT = 1000
SELECTED_PER_QUERY = 20
SIZE = QUERY_COUNT * SELECTED_PER_QUERY


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    recorded = {
        "exact_hnsw": ("Exact/HNSW selected-20 shared fraction: **0.44885**", "0.44885"),
        "exact_ivf": ("Exact/IVF selected-20 shared fraction: **0.48055**", "0.48055"),
        "hnsw_ivf": ("HNSW vs IVF selected-20 **shared fraction**: **0.4344**", "0.4344"),
    }
    counts = {}
    for key, (source_literal, fraction) in recorded.items():
        assert source_literal in source, f"source mismatch: {key}"
        raw_count = Decimal(fraction) * SIZE
        assert raw_count == int(raw_count), f"nonintegral selected-pair count: {key}"
        counts[key] = int(raw_count)

    eh, ei, hi = (counts["exact_hnsw"], counts["exact_ivf"], counts["hnsw_ivf"])
    # Three sets E, H, I each have SIZE distinct (query ID, passage ID) keys.
    # Triple intersection t is constrained by each pair intersection and by
    # the capacity of each individual arm to hold its two pair intersections.
    t_min = max(0, eh + ei - SIZE, eh + hi - SIZE, ei + hi - SIZE)
    t_max = min(eh, ei, hi)

    def region_counts(t: int) -> tuple[int, ...]:
        # Mutually exclusive Venn regions: EHI, EH-only, EI-only, HI-only,
        # E-only, H-only, I-only. Nonnegative endpoints show aggregate feasibility.
        return (
            t, eh - t, ei - t, hi - t,
            SIZE - eh - ei + t,
            SIZE - eh - hi + t,
            SIZE - ei - hi + t,
        )

    for t in (t_min, t_max):
        regions = region_counts(t)
        assert min(regions) >= 0
        assert regions[0] + regions[1] + regions[2] + regions[4] == SIZE
        assert regions[0] + regions[1] + regions[3] + regions[5] == SIZE
        assert regions[0] + regions[2] + regions[3] + regions[6] == SIZE
        assert sum(regions) == 3 * SIZE - eh - ei - hi + t

    union_two = 2 * SIZE - hi
    union_three_min = 3 * SIZE - eh - ei - hi + t_min
    union_three_max = 3 * SIZE - eh - ei - hi + t_max
    exact_extra_min = union_three_min - union_two
    exact_extra_max = union_three_max - union_two
    assert (eh, ei, hi) == (8977, 9611, 8688)
    assert (t_min, t_max) == (0, 8688)
    assert (union_two, union_three_min, union_three_max) == (31312, 32724, 41412)
    assert (exact_extra_min, exact_extra_max) == (1412, 10100)

    print(f"arm_size={SIZE}; E∩H={eh}; E∩I={ei}; H∩I={hi}")
    print(f"triple_intersection_aggregate_bounds={t_min}..{t_max}")
    print(f"distinct_selected_keys_two_arms={union_two}")
    print(f"distinct_selected_keys_three_arms={union_three_min}..{union_three_max}")
    print(f"additional_distinct_exact_keys={exact_extra_min}..{exact_extra_max}")
    print("conditional_shared_positive_per_query_three_arm_keys=33724..42412")
    print("PASS: exact integer source/accounting checks (not historical replay)")


if __name__ == "__main__":
    main()
