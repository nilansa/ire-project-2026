#!/usr/bin/env python3
"""Ceous Stage 0: qrel-observed top-K ordering/admission diagnostics.

No neural inference, no teacher scoring, no model training. Expects JSONL:
 runs: {"query_id":"q1","ranked_doc_ids":["d1",...]}
       optional "expanded_ranked_doc_ids": [...]
 qrels:{"query_id":"q1","doc_id":"d1","relevance":2}

Unjudged docs are treated as zero *only for qrel-conditioned evaluation*;
reported oracle scores are NOT full-corpus or human relevance upper bounds.
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def dcg(grades):
    return sum((2 ** max(0, g) - 1) / math.log2(i + 2)
               for i, g in enumerate(grades))


def observed_ndcg(doc_ids, judged, k):
    gains = [judged.get(doc_id, 0) for doc_id in doc_ids[:k]]
    best = sorted(judged.values(), reverse=True)[:k]
    denominator = dcg(best)
    return dcg(gains) / denominator if denominator else None


def observed_recall(doc_ids, judged, k):
    positives = {d for d, g in judged.items() if g > 0}
    return len(positives.intersection(doc_ids[:k])) / len(positives) if positives else None


def oracle_reorder(pool, judged, k):
    """Stable judged-grade sort of an existing candidate pool (offline ONLY)."""
    return [doc_id for _, doc_id in sorted(
        enumerate(pool), key=lambda p: (-judged.get(p[1], 0), p[0]))][:k]


def paired_bootstrap_ci(differences, seed=42, iterations=1000):
    if not differences:
        return None
    rng = random.Random(seed)
    n = len(differences)
    sampled_means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n
                           for _ in range(iterations))
    lo = sampled_means[int(0.025 * (iterations - 1))]
    hi = sampled_means[int(0.975 * (iterations - 1))]
    return [lo, hi]


def compute(runs, qrel_rows, k=10, b=100, seed=42, bootstrap=1000):
    if not 0 < k <= b:
        raise ValueError("require 0 < k <= b")
    if bootstrap < 100:
        raise ValueError("bootstrap must be at least 100 samples")
    judged_by_query = defaultdict(dict)
    for row in qrel_rows:
        qid, did, grade = str(row["query_id"]), str(row["doc_id"]), int(row["relevance"])
        if grade < 0:
            raise ValueError("qrel grades must be nonnegative")
        if did in judged_by_query[qid] and judged_by_query[qid][did] != grade:
            raise ValueError(f"conflicting qrel for query={qid!r}, doc={did!r}")
        judged_by_query[qid][did] = grade

    metrics = defaultdict(list)
    count = {"queries": 0, "queries_without_positive_qrels": 0,
             "queries_with_promotable_judged_positive": 0,
             "queries_with_new_judged_positive_from_expansion": 0,
             "qrels_judged_relevant_outside_initial_B": 0,
             "total_retrieved_unjudged_top_K": 0,
             "total_baseline_K_slots": 0}
    ids_seen = set()
    for row in runs:
        qid = str(row["query_id"])
        if qid in ids_seen:
            raise ValueError(f"duplicate run query_id: {qid}")
        ids_seen.add(qid)
        original = [str(d) for d in row["ranked_doc_ids"]]
        if len(original) < b or len(set(original)) != len(original):
            raise ValueError(f"query={qid}: need at least B={b} unique initial IDs")
        judged = judged_by_query.get(qid, {})
        if not judged or not any(g > 0 for g in judged.values()):
            count["queries_without_positive_qrels"] += 1
        in_k = original[:k]
        in_b = original[:b]
        baseline = observed_ndcg(in_k, judged, k)
        order = observed_ndcg(oracle_reorder(in_k, judged, k), judged, k)
        admission = observed_ndcg(oracle_reorder(in_b, judged, k), judged, k)
        for label, score in (("baseline_ndcg_at_k", baseline),
                             ("oracle_order_within_initial_k_ndcg", order),
                             ("oracle_select_from_initial_b_ndcg", admission),
                             ("qrel_conditional_recall_at_k", observed_recall(in_k, judged, k)),
                             ("qrel_conditional_recall_at_b", observed_recall(in_b, judged, b))):
            if score is not None:
                metrics[label].append(score)
        if baseline is not None:
            metrics["order_gain"].append(order - baseline)
            metrics["admission_gain_over_order"].append(admission - order)
            metrics["admission_gain_over_baseline"].append(admission - baseline)
        if any(judged.get(d, 0) > 0 for d in in_b[k:]):
            count["queries_with_promotable_judged_positive"] += 1
        count["qrels_judged_relevant_outside_initial_B"] += sum(
            grade > 0 and doc_id not in in_b for doc_id, grade in judged.items())
        count["total_retrieved_unjudged_top_K"] += sum(d not in judged for d in in_k)
        count["total_baseline_K_slots"] += k
        if "expanded_ranked_doc_ids" in row:
            expanded = [str(d) for d in row["expanded_ranked_doc_ids"]]
            if len(set(expanded)) != len(expanded):
                raise ValueError(f"query={qid}: duplicate expanded IDs")
            if any(d not in in_b and judged.get(d, 0) > 0 for d in expanded):
                count["queries_with_new_judged_positive_from_expansion"] += 1
        count["queries"] += 1
    if not count["queries"]:
        raise ValueError("empty run")
    mean = {key: sum(vals) / len(vals) for key, vals in metrics.items()}
    intervals = {key: paired_bootstrap_ci(metrics[key], seed, bootstrap)
                 for key in ("order_gain", "admission_gain_over_order", "admission_gain_over_baseline")}
    return {"method": "qrel_observed_headroom_not_ground_truth_oracle",
            "cutoffs": {"k": k, "b": b}, "counts": count,
            "mean_over_queries_with_positive_qrels": mean,
            "paired_query_bootstrap_95CI": intervals,
            "warning": "UNJUDGED != IRRELEVANT. Qrel-conditional results are not a full-corpus oracle; no teacher score was used."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--b", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    result = compute(read_jsonl(args.runs), read_jsonl(args.qrels), args.k, args.b,
                     args.seed, args.bootstrap)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
