#!/usr/bin/env python3
"""Read-only teacher-suitability screen on ORIGINAL exact-retrieval 21-way slates.

Uses only score rows already in the SHA-pinned 12 October private archive.
Never requests a new teacher logit, loads a student model, or trains.
The complete-slate subset is determined by overlap with prior HNSW/IVF
teacher acquisitions, so it is NOT representative of all 800 exact slates.
"""
import argparse
import json
import math
from collections import Counter
from statistics import mean, median

from guarded_exact_control_runner import build_plan


def audit():
    plan = build_plan()  # verifies original archive and slate identities
    gap_histogram = Counter()
    complete_positive_mass = []
    complete_normalized_entropy = []
    complete_misranks = 0
    definite_misranks = 0
    definite_negative_overruns = 0
    observed_count = 0
    positive_cached = 0

    for slate in plan.slates:
        scores = [plan.cached_raw_logits.get((slate.query_id, pid))
                  for pid in slate.passage_ids]
        assert len(scores) == 21
        positive = scores[0]
        if positive is None:
            raise ValueError("Known positive has no historical teacher logit")
        positive_cached += 1
        present_negatives = [x for x in scores[1:] if x is not None]
        missing = 20 - len(present_negatives)
        gap_histogram[missing] += 1
        observed_count += 1 + len(present_negatives)
        outruns = sum(x > positive for x in present_negatives)
        definite_negative_overruns += outruns
        definite_misranks += bool(outruns)

        if not missing:
            complete_misranks += bool(outruns)
            center = max(scores)
            exps = [math.exp(x - center) for x in scores]
            z = sum(exps)
            weights = [x / z for x in exps]
            complete_positive_mass.append(weights[0])
            entropy = -sum(w * math.log(w) for w in weights if w > 0)
            complete_normalized_entropy.append(entropy / math.log(21))

    full = len(complete_positive_mass)
    missing_total = sum(k * n for k, n in gap_histogram.items())
    if (len(plan.slates), positive_cached, observed_count, missing_total,
            len(plan.cached_raw_logits), len(plan.missing_teacher_pairs)) != (
            800, 800, 10282, 6518, 32312, 6518):
        raise ValueError("Frozen budget or teacher coverage changed")
    if (full, complete_misranks, definite_misranks) != (121, 12, 29):
        raise ValueError("Frozen exact-slate teacher-suitability counts changed")
    return {
        "status": "PASS_ARCHIVED_SCORES_ONLY_NO_NEW_INFERENCE",
        "training_exact_slates": 800,
        "slate_size": 21,
        "cached_scores_in_exact_consumed_slates": observed_count,
        "missing_exact_pair_scores": missing_total,
        "complete_21_way_slates": full,
        "complete_slate_teacher_anchor_top1": full - complete_misranks,
        "complete_slate_teacher_anchor_not_top1": complete_misranks,
        "certified_anchor_outranked_by_cached_negative_all_slates": definite_misranks,
        "cached_negative_scores_above_anchor_all_slates": definite_negative_overruns,
        "complete_slate_mean_teacher_positive_mass": mean(complete_positive_mass),
        "complete_slate_median_teacher_positive_mass": median(complete_positive_mass),
        "complete_slate_mean_normalized_teacher_entropy": mean(complete_normalized_entropy),
        "gaps_per_slate": {str(k): gap_histogram[k] for k in sorted(gap_histogram)},
        "interpretation_limit": (
            "121 complete slates form a prior-acquisition-overlap-selected subset; "
            "29/800 is only a lower bound on teacher-anchor misranking in all slates. "
            "Teacher ranking vs a known-positive anchor is not student-vs-teacher "
            "error correction and cannot establish post-training quality."
        ),
        "teacher_calls": 0, "model_forward_calls": 0, "optimizer_updates": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(audit(), sort_keys=True))
