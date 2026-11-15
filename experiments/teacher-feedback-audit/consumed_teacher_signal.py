#!/usr/bin/env python3
"""Measure the teacher target actually scheduled for training (archived 50k MS MARCO only).

Read-only analysis of the 12 October hash-locked archive. It does not infer
parameter-gradient contribution, qrel correctness, or a causal ANN effect.
No teacher inference, training, or external dependencies are needed.
"""

import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

from exact_teacher_cache_gap import ARCHIVE_SHA256, npy_matrix


def main(path):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA256
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("inputs/ire-audit-evidence.zip"))) as audit:
            with zipfile.ZipFile(io.BytesIO(outer.read("inputs/identity-and-search-evidence.zip"))) as ident:
                identity = json.loads(ident.read("marco/identity.json"))
                split = json.loads(audit.read("marco/runtime/split.json"))
                original = json.loads(audit.read("marco/teacher_sanity/RESULTS.json"))
                runs = {
                    arm: json.loads(audit.read(f"marco/teacher_sanity/train_{arm}.json"))
                    for arm in ("hnsw", "ivf")
                }
                schedule = list(map(int, runs["hnsw"]["training_query_ids"]))
                assert schedule == list(map(int, runs["ivf"]["training_query_ids"]))
                assert len(schedule) == len(set(schedule)) == 800
                assert original["config"]["temperature"] == 1.0
                assert original["config"]["updates"] == 100
                assert original["config"]["batch_queries"] == 8

                queries = list(map(int, split["train_qids"]))
                assert len(queries) == len(set(queries)) == 1000
                assert set(schedule) <= set(queries)
                passages = list(map(int, identity["pids"]))
                qrels = {int(q): set(map(int, ps)) for q, ps in identity["qrels"].items()}
                negatives = {
                    arm: npy_matrix(audit.read(f"marco/runtime/neg_{arm}.npy"))
                    for arm in ("hnsw", "ivf")
                }
                rows = list(csv.DictReader(io.StringIO(
                    audit.read("marco/teacher_sanity/teacher_scores.csv").decode()
                )))
                assert len(rows) == 32312
                logits = {}
                for row in rows:
                    key = (int(row["query_id"]), int(row["passage_id"]))
                    assert key not in logits
                    value = float(row["raw_teacher_logit"])
                    assert math.isfinite(value)
                    logits[key] = value
                diag = {
                    int(row["query_id"]): row
                    for row in csv.DictReader(io.StringIO(
                        audit.read("marco/teacher_sanity/teacher_diagnostics.csv").decode()
                    ))
                }
                assert len(diag) == 1000

                trained = set(schedule)
                results = {}
                under_half = {}
                for arm in ("hnsw", "ivf"):
                    prob_by_query = {}
                    not_top = set()
                    for i, q in enumerate(queries):
                        chosen = [passages[j] for j in negatives[arm][i]]
                        assert len(chosen) == len(set(chosen)) == 20
                        assert not set(chosen).intersection(qrels[q])
                        known_positives = [p for p in qrels[q] if (q, p) in logits]
                        assert len(known_positives) == 1
                        positives = known_positives[0]
                        slate = [positives, *chosen]
                        scores = [logits[q, p] for p in slate]
                        top = max(scores)
                        weights = [math.exp(x - top) for x in scores]
                        mass = weights[0] / sum(weights)
                        assert abs(mass - float(diag[q][f"{arm}_positive_soft_mass"])) < 2e-6
                        prob_by_query[q] = mass
                        if scores[0] < max(scores[1:]):
                            not_top.add(q)

                    original_mean = original["teacher_diagnostics"][arm]["positive_soft_mass"]["mean"]
                    assert abs(sum(prob_by_query.values()) / 1000 - original_mean) < 2e-6
                    masses = {q: 1.0 - prob_by_query[q] for q in trained}
                    total_mass = sum(masses.values())
                    low_positives = {q for q in trained if prob_by_query[q] < 0.5}
                    under_half[arm] = low_positives
                    assert total_mass > 0
                    results[arm] = {
                        "trained_800_mean_positive_probability": round(sum(prob_by_query[q] for q in trained) / 800, 9),
                        "trained_800_positive_probability_at_least_0_999": sum(prob_by_query[q] >= 0.999 for q in trained),
                        "trained_800_positive_probability_below_0_5": len(low_positives),
                        "trained_800_positive_not_teacher_top": len(not_top & trained),
                        "trained_800_sum_negative_probability_mass": round(total_mass, 9),
                        "fraction_negative_probability_mass_from_below_0_5_queries": round(
                            sum(masses[q] for q in low_positives) / total_mass, 9
                        ),
                        "unused_200_mean_positive_probability": round(
                            sum(prob_by_query[q] for q in set(queries) - trained) / 200, 9
                        ),
                        "archived_1000_mean_positive_probability_verified": True,
                    }

                result = {
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "cohort": "historical_50k_MS_MARCO_ANCE_100_update_teacher_pilot",
                    "teacher_temperature": 1.0,
                    "score_records_verified": len(logits),
                    "actually_consumed_query_lists_per_arm": len(schedule),
                    "paired_query_ids_with_positive_probability_below_0_5_in_both_arms": len(
                        under_half["hnsw"] & under_half["ivf"]
                    ),
                    "arms": results,
                    "interpretation_boundary": (
                        "These are teacher soft-target probabilities on historical training slates, "
                        "not observed student gradients, causal backend effects, or held-out performance. "
                        "Do not use a teacher score to alter the pre-teacher selection policy."
                    ),
                }
                print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 -B consumed_teacher_signal.py /path/to/verified_archive.zip")
    main(sys.argv[1])
