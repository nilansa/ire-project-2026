#!/usr/bin/env python3
"""Read-only, standard-library verification of frozen Ceous C5/C6 result exports.

This checks file hashes, split identities and per-query mean arithmetic. It does
not retrain models, rerun teacher inference, reconstruct confidence intervals,
or establish causal validity. Never writes into experiment directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import unittest
from pathlib import Path

SEEDS = (17, 29, 41)
DOMAINS = {"fiqa": 648, "nfcorpus": 323}
ARMS = ("G100", "K100", "G10", "K10")
C6_PROTOCOL_SHA = "6c27a7552f4413324110dd659a4b97659bb04b6a446457e43717b8c99eb7a09c"
C6_GATES_SHA = "ec7ea2a6450add488615f907b28fd06d97171c46b9112c7739ce8298c03295a4"
TOL = 1e-10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    require(math.isfinite(actual) and math.isfinite(expected), f"{label}: nonfinite")
    require(abs(actual - expected) <= TOL, f"{label}: {actual} != {expected}")


def split_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode("utf-8")).hexdigest()


def digest_order(ids: list[str], prefix: str) -> list[str]:
    return sorted(ids, key=lambda q: hashlib.sha256(f"{prefix}|{q}".encode()).hexdigest())


def model_scores(path: Path) -> dict[str, float]:
    data = read_json(path)
    require("by_qid" in data, f"{path}: missing by_qid")
    result = {}
    for qid, row in data["by_qid"].items():
        score = float(row["ndcg@10"])
        require(math.isfinite(score) and -TOL <= score <= 1 + TOL,
                f"{path}: invalid nDCG@10 for {qid}")
        result[str(qid)] = score
    return result


def seed_differences(rows: dict, a: str, b: str, qids: set[str]) -> list[float]:
    return [statistics.fmean(rows[a][seed][qid] - rows[b][seed][qid]
                             for qid in sorted(qids)) for seed in SEEDS]


def check_stat(stat: dict, diffs: list[float], n_queries: int, name: str) -> None:
    require(stat["n_queries"] == n_queries, f"{name}: wrong query count")
    assert_close(float(stat["mean"]), statistics.fmean(diffs), name)
    for i, seed in enumerate(SEEDS):
        assert_close(float(stat["per_seed_mean_diffs"][i]), diffs[i], f"{name} seed {seed}")
    require(stat["metric"] == "ndcg@10", f"{name}: metric changed")
    ci = stat["ci95"]
    require(len(ci) == 2 and ci[0] <= stat["mean"] <= ci[1],
            f"{name}: invalid stored CI interval (CI is not recomputed)")
    if "verdict" in stat:
        expected = ("PASS" if stat["mean"] >= 0.01 and ci[0] > 0
                    and all(d > 0 for d in diffs) else "FAIL")
        require(stat["verdict"] == expected, f"{name}: gate verdict inconsistent")


def verify_c6(c6_run: Path) -> None:
    root = c6_run / "results-collected"
    gates_path = root / "gates.json"
    require(sha256(gates_path) == C6_GATES_SHA, "C6 gates.json SHA256 differs from frozen evidence")
    gates = read_json(gates_path)
    state = read_json(root / "RUN_STATE.json")
    qualification = read_json(root / "qualification_tau.json")
    for name, data in (("gates", gates), ("state", state), ("qualification", qualification)):
        require(data["protocol_sha256"] == C6_PROTOCOL_SHA, f"{name}: protocol hash differs")
    require(gates["status"] == state["status"] == "COMPLETE", "C6 run is incomplete")
    require(gates["teacher_pairs_scored"] == 200_000, "C6 recorded teacher-pair count changed")
    require(gates["test_qrels_opened_only_in_final_test_eval"] is True,
            "C6 test-qrel handling flag absent")
    require(state["stage_status"]["training"]["trained_dataset_seed_pairs"] == 24,
            "C6 expected 24 trained arm/dataset/seed runs")
    require(state["stage_status"]["final_test_eval"]["evaluated_models"] == 26,
            "C6 expected 26 evaluated models including untouched controls")
    require(qualification["frozen_before_training"] is True, "C6 tau not marked frozen")
    for ds, tau in (("fiqa", 1.0), ("nfcorpus", 4.0)):
        require(qualification["datasets"][ds]["qualified"] is True,
                f"{ds}: teacher was not qualified")
        assert_close(float(qualification["datasets"][ds]["tau_t"]), tau,
                     f"{ds}: teacher temperature")

    combined = {key: [] for key in ("H6a", "H6b", "K10_minus_G100")}
    pooled_queries = 0
    split_root = c6_run / "nilan-stage" / "splits"
    for ds, expected_n in DOMAINS.items():
        per_ds = root / "test_results" / ds
        a0 = model_scores(per_ds / "A0.json")
        qids = set(a0)
        require(len(qids) == expected_n, f"{ds}: wrong official test-query count")
        rows = {"A0": {seed: a0 for seed in SEEDS}}
        for arm in ARMS:
            rows[arm] = {}
            for seed in SEEDS:
                scores = model_scores(per_ds / f"{arm}__seed{seed}.json")
                require(set(scores) == qids, f"{ds} {arm} seed {seed}: query-ID mismatch")
                rows[arm][seed] = scores
        pooled_queries += len(qids)
        for label, first, second in (("H6a", "K100", "G100"),
                                     ("H6b", "K10", "G10"),
                                     ("K10_minus_G100", "K10", "G100")):
            diffs = seed_differences(rows, first, second, qids)
            check_stat(gates["per_dataset"][ds][label], diffs, len(qids), f"{ds} {label}")
            combined[label].extend(
                statistics.fmean(rows[first][seed][qid] - rows[second][seed][qid]
                                 for seed in SEEDS) for qid in sorted(qids))
        require(gates["per_dataset"][ds]["teacher_pairs_scored"] == 100_000,
                f"{ds}: teacher-score pair count changed")
        split_path = split_root / f"{ds}.json"
        if split_path.exists():
            split = read_json(split_path)
            pool, train, val, gold = (split[key] for key in
                                      ("pool_qids", "train_qids", "validation_qids", "gold10_qids"))
            require((len(pool), len(train), len(val), len(gold)) == (2000, 1800, 200, 180),
                    f"{ds}: split sizes differ")
            require(len(set(pool)) == 2000 and set(train) | set(val) == set(pool),
                    f"{ds}: training pool malformed")
            require(not (set(train) & set(val) or set(train) & qids or set(val) & qids),
                    f"{ds}: train/validation/test query IDs overlap")
            require(set(gold) <= set(train), f"{ds}: gold10 IDs outside train")
            require(pool == digest_order(pool, f"c6-train|{ds}"),
                    f"{ds}: pool hash ordering differs")
            require(val == digest_order(pool, f"c6-val|{ds}")[:200],
                    f"{ds}: validation hash ordering differs")
            require(train == [qid for qid in pool if qid not in set(val)],
                    f"{ds}: training selection differs")
            require(gold == digest_order(train, f"c6-gold10|{ds}")[:180],
                    f"{ds}: gold10 hash ordering differs")
            for key, items in (("pool", pool), ("train", train), ("validation", val), ("gold10", gold)):
                require(split["digests"][key + "_sha256"] == split_hash(items),
                        f"{ds}: frozen {key} digest mismatch")
            print(f"C6 {ds}: {len(train)} train, {len(val)} validation, "
                  f"{len(gold)} gold10, {len(qids)} test; split digest checks PASS")
        else:
            print(f"C6 {ds}: {len(qids)} test; WARNING: no split file at {split_path}")
        print(f"C6 {ds}: raw per-query arm means and seed means PASS")
    require(pooled_queries == 971, "C6 expected 971 pooled official test queries")
    for label, output_key in (("H6a", "K100_minus_G100"),
                              ("H6b", "K10_minus_G10"),
                              ("K10_minus_G100", "K10_minus_G100")):
        mean = statistics.fmean(combined[label])
        assert_close(mean, float(gates["pooled"][output_key]["mean"]), f"pooled {label}")
        require(gates["pooled"][output_key]["n_queries"] == pooled_queries,
                f"pooled {label}: query count differs")
        print(f"C6 pooled {label}: n={pooled_queries}, mean={mean:+.9f} PASS")
    print("C6 checks PASS. Confidence intervals, teacher inference and training were not rerun.")


def verify_c5(c5_run: Path) -> None:
    gate = read_json(c5_run / "results-d4-combined" / "g5.json")
    sources = {17: "results-d3", 29: "results-d4-s29", 41: "results-d4-s41"}
    loaded = {}
    a0 = None
    for seed, part in sources.items():
        csv_path = c5_run / part / "per_query_holdout.csv"
        require(sha256(csv_path) == gate["combination"]["source_csv_sha256"][str(seed)],
                f"C5 seed {seed}: source CSV SHA mismatch")
        groups: dict[str, dict[str, float]] = {}
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                groups.setdefault(row["model_id"], {})[row["query_id"]] = float(row["ndcg@10"])
        controls = {}
        controls["A0"] = groups["A0"]
        for arm in ("G", "KD-full"):
            matches = [name for name in groups if name.startswith(arm + "-")
                       and name.endswith(f"seed{seed}")]
            require(len(matches) == 1, f"C5 {arm} seed {seed}: ambiguous model ID")
            controls[arm] = groups[matches[0]]
        require(all(len(v) == 200 for v in controls.values()), "C5 wrong holdout size")
        require(all(set(v) == set(controls["A0"]) for v in controls.values()),
                f"C5 seed {seed}: holdout IDs mismatch")
        if a0 is None:
            a0 = controls["A0"]
        else:
            require(a0 == controls["A0"], "C5 untouched checkpoint differs across seed jobs")
        loaded[seed] = controls
    assert a0 is not None
    qids = sorted(a0)
    for first, second, key in (("KD-full", "G", "kd_full_minus_g_full"),
                               ("G", "A0", "g_full_minus_a0")):
        diffs = [statistics.fmean(loaded[seed][first][qid] - loaded[seed][second][qid]
                                   for seed in SEEDS) for qid in qids]
        actual = statistics.fmean(diffs)
        assert_close(actual, float(gate[key]["mean"]), f"C5 {key}")
        print(f"C5 {key}: n=200, mean={actual:+.9f} PASS")
    require(gate["kd_full_minus_g_full"]["mean"] < 0.01,
            "C5 stored G5 fail threshold changed")
    print("C5 checks PASS. Confidence intervals, teacher inference and training were not rerun.")


class SelfTests(unittest.TestCase):
    def test_split_digest(self) -> None:
        self.assertEqual(split_hash(["a", "b"]), hashlib.sha256(b"a\nb").hexdigest())
        self.assertEqual(digest_order(["q2", "q1", "q3"], "test"),
                         digest_order(["q3", "q2", "q1"], "test"))

    def test_seed_differences(self) -> None:
        rows = {
            "new": {s: {"q1": 0.6, "q2": 0.8} for s in SEEDS},
            "old": {s: {"q1": 0.4, "q2": 0.6} for s in SEEDS},
        }
        for d in seed_differences(rows, "new", "old", {"q1", "q2"}):
            self.assertAlmostEqual(d, 0.2)

    def test_bad_verdict_rejected(self) -> None:
        bad = {"n_queries": 2, "mean": 0.02, "per_seed_mean_diffs": [0.02] * 3,
               "ci95": [-0.1, 0.1], "metric": "ndcg@10", "verdict": "PASS"}
        with self.assertRaisesRegex(ValueError, "gate verdict inconsistent"):
            check_stat(bad, [0.02] * 3, 2, "synthetic")


def main() -> None:
    if "--self-test" in sys.argv:
        unittest.main(argv=[sys.argv[0]], exit=False)
        return
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c5-run", type=Path, help="C5 run-002-ada directory")
    ap.add_argument("--c6-run", type=Path, help="C6 shared/run-001 directory")
    args = ap.parse_args()
    if args.c5_run is None and args.c6_run is None:
        ap.error("provide --c5-run and/or --c6-run, or --self-test")
    if args.c5_run is not None:
        verify_c5(args.c5_run)
    if args.c6_run is not None:
        verify_c6(args.c6_run)


if __name__ == "__main__":
    main()
