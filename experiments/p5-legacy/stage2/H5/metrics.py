"""Question-level metrics and paired bootstrap used by the H5 protocol."""

from collections import defaultdict

import numpy as np


METRICS = ("full_support", "sp_recall", "evidence_tokens", "latency_seconds")


def question_scores(question, ranking, corpus_by_pid, k):
    gold = set(question["gold_pids"])
    if not gold:
        raise ValueError(f"question {question['qid']} has no gold passages")
    retrieved = list(ranking[:k])
    unknown = set(retrieved) - corpus_by_pid.keys()
    if unknown:
        raise ValueError(f"unknown retrieved passage IDs: {sorted(unknown)[:3]}")
    hits = gold.intersection(retrieved)
    return {
        "full_support": float(gold.issubset(retrieved)),
        "sp_recall": len(hits) / len(gold),
        "evidence_tokens": sum(
            len(str(corpus_by_pid[pid].get("text", "")).split()) for pid in retrieved
        ),
    }


def paired_bootstrap_ci(differences, n_resamples=1000, seed=0):
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired differences must be a nonempty 1-D sequence")
    rng = np.random.default_rng(seed)
    sample_ids = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[sample_ids].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(values.mean()),
        "ci95": [float(low), float(high)],
        "n": int(values.size),
        "resamples": int(n_resamples),
        "seed": int(seed),
    }


def _summarize(rows):
    if not rows:
        return {name: None for name in METRICS} | {"n": 0}
    return {
        **{name: float(np.mean([row[name] for row in rows])) for name in METRICS},
        "n": len(rows),
    }


def evaluate_dataset(dataset, questions, corpus_by_pid, rankings, latencies, k=10):
    """Evaluate method rankings for the same ordered question set."""
    qids = [q["qid"] for q in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("question IDs must be unique")
    methods = list(rankings)
    if len(methods) < 1:
        raise ValueError("at least one method is required")

    per_method = {method: {} for method in methods}
    by_type = defaultdict(lambda: defaultdict(list))
    for question in questions:
        qid = question["qid"]
        qtype = str(question.get("type") or "unspecified")
        for method in methods:
            if qid not in rankings[method] or qid not in latencies[method]:
                raise ValueError(f"missing {method} result for {qid}")
            score = question_scores(question, rankings[method][qid], corpus_by_pid, k)
            latency = float(latencies[method][qid])
            if latency < 0:
                raise ValueError("latency must be nonnegative")
            row = {**score, "latency_seconds": latency}
            per_method[method][qid] = row
            by_type[qtype][method].append(row)

    summary = {
        method: _summarize(list(per_method[method].values())) for method in methods
    }
    grouped = {
        qtype: {
            method: _summarize(by_type[qtype][method])
            for method in methods
        }
        for qtype in sorted(by_type)
    }
    paired = {}
    for i, method_a in enumerate(methods):
        for method_b in methods[i + 1 :]:
            diffs = {}
            for metric in METRICS:
                paired_diffs = [
                    per_method[method_a][qid][metric] - per_method[method_b][qid][metric]
                    for qid in qids
                ]
                diffs[metric] = paired_bootstrap_ci(paired_diffs)
            paired[f"{method_a}_minus_{method_b}"] = diffs

    return {
        "dataset": dataset,
        "split": "tune",
        "k": int(k),
        "n": len(questions),
        "methods": summary,
        "by_question_type": grouped,
        "paired_differences": paired,
    }
