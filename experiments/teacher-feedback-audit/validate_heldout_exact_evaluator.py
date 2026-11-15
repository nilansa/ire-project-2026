#!/usr/bin/env python3
"""Preflight the frozen 50k MS MARCO held-out evaluator on REAL saved rankings.

Read-only evidence sources: the SHA-locked October 12 archive and original
September pilot's 500 x 1,000 ranked passage-ID arrays. No encoder, teacher,
optimizer, index construction, or neural inference is ever invoked.

The ranked-ID evaluator is deliberately reusable by an *authorised future*
exact-control runner after it has re-encoded the entire frozen corpus. This
module neither authorises nor provides a training/inference entry point.
"""

import argparse
import ast
import csv
import hashlib
import io
import json
import struct
import zipfile
from pathlib import Path

from freeze_exact_positive_control import ARCHIVE, OUTPUT as FROZEN, digest_rows


HERE = Path(__file__).resolve().parent
RECIPE = HERE / "exact_execution_recipe.json"
RESULT = HERE / "heldout_exact_evaluator_validation.json"
HISTORICAL_RANKINGS = (
    Path.home() / ".codex/scratch/cross-surface/IRE/ann-micro-pilot-20260907/"
    "shared/runtime/results/teacher_sanity"
)
EXPECTED_ARCHIVE_SHA256 = "b51dea1c6dfe9c61fa7d245eeed004d9bfb347b3cf7e1a38ba1d665924426115"
QUERY_COUNT = 500
CORPUS_COUNT = 50000
RANK_DEPTH = 1000
ARMS = ("initial", "hnsw", "ivf")
METRICS = ("MRR@10", "Recall@100", "Recall@1000")


def read_saved_rankings(path):
    """Read strictly validated 500 x 1,000 little-endian int64 .npy IDs."""
    raw = Path(path).read_bytes()
    if raw[:6] != b"\x93NUMPY" or raw[6] not in (1, 2, 3):
        raise ValueError("Unexpected saved ranking array header")
    offset, fmt = (10, "<H") if raw[6] == 1 else (12, "<I")
    header_bytes = struct.unpack_from(fmt, raw, 8)[0]
    header = ast.literal_eval(raw[offset:offset + header_bytes].decode("latin1"))
    if header != {"descr": "<i8", "fortran_order": False,
                  "shape": (QUERY_COUNT, RANK_DEPTH)}:
        raise ValueError("Wrong raw ranking layout or ID encoding")
    start = offset + header_bytes
    if len(raw) != start + QUERY_COUNT * RANK_DEPTH * 8:
        raise ValueError("Truncated or padded saved ranking array")
    rows = [struct.unpack_from("<1000q", raw, start + i * RANK_DEPTH * 8)
            for i in range(QUERY_COUNT)]
    return rows, hashlib.sha256(raw).hexdigest()


def evaluate_ranked_ids(query_ids, rows, qrels, corpus_ids, *, expected_eval_qrels_sha256):
    """Compute source-matched held-out metrics from post-training exact ranks.

    The future caller must supply all 500 *frozen* query IDs and genuine
    full-corpus top-1000 passage IDs, in order. Never evaluate candidate slates
    or pretraining negatives as if they were final full-corpus rankings.
    """
    if len(query_ids) != QUERY_COUNT or len(rows) != QUERY_COUNT:
        raise ValueError("Exactly 500 frozen held-out query rankings required")
    if len(set(query_ids)) != QUERY_COUNT:
        raise ValueError("Repeated held-out query")
    if digest_rows([[q, sorted(qrels[q])] for q in query_ids]) != expected_eval_qrels_sha256:
        raise ValueError("Evaluation query identities/order/qrels differ from frozen cohort")
    if len(corpus_ids) != CORPUS_COUNT:
        raise ValueError("Wrong diagnostic corpus size")
    valid_ids = set(corpus_ids)
    if len(valid_ids) != CORPUS_COUNT:
        raise ValueError("Duplicate corpus document IDs")
    records = []
    for qid, row in zip(query_ids, rows):
        if qid not in qrels or not qrels[qid]:
            raise ValueError("Unjudged held-out query")
        if len(row) != RANK_DEPTH or len(set(row)) != RANK_DEPTH:
            raise ValueError("Incomplete or duplicated full-corpus ranking")
        if not set(row).issubset(valid_ids):
            raise ValueError("Ranking includes non-corpus document IDs")
        relevant = qrels[qid]
        if not relevant.issubset(valid_ids):
            raise ValueError("Frozen held-out relevance missing from corpus")
        hit_ranks = [rank for rank, pid in enumerate(row, 1) if pid in relevant]
        first = hit_ranks[0] if hit_ranks else 0
        records.append({
            "query_id": qid,
            "first_relevant_rank_in1000": first,
            "MRR@10": 1.0 / first if 0 < first <= 10 else 0.0,
            "Recall@100": sum(rank <= 100 for rank in hit_ranks) / len(relevant),
            "Recall@1000": len(hit_ranks) / len(relevant),
        })
    summary = {metric: sum(r[metric] for r in records) / QUERY_COUNT
               for metric in METRICS}
    return records, summary


def validate_sources(archive=ARCHIVE, rankings_root=HISTORICAL_RANKINGS):
    raw = Path(archive).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("October 12 source archive hash mismatch")
    frozen_bytes = FROZEN.read_bytes()
    frozen = json.loads(frozen_bytes)
    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if frozen["status"] != "FROZEN_PROTOCOL_NOT_EXECUTED":
        raise ValueError("Data freeze status changed")
    if recipe["frozen_data_manifest_sha256"] != hashlib.sha256(frozen_bytes).hexdigest():
        raise ValueError("Data/recipe lock mismatch")
    if recipe["execution_permission"] != "NO INFERENCE, TRAINING, OR TEACHER SCORING AUTHORIZED":
        raise ValueError("Execution gate must remain closed")
    if (frozen["training_queries"], frozen["evaluation_queries"],
            frozen["new_teacher_scores_needed"]) != (800, 500, 6518):
        raise ValueError("Unplanned change in teacher-positive-control cohort")

    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        source = outer.read("inputs/ire-audit-evidence.zip")
        identities = outer.read("inputs/identity-and-search-evidence.zip")
    if hashlib.sha256(source).hexdigest() != frozen["source_zip_sha256"]:
        raise ValueError("Nested pilot source archive changed")

    with zipfile.ZipFile(io.BytesIO(source)) as audit, \
         zipfile.ZipFile(io.BytesIO(identities)) as identity_zip:
        identity = json.loads(identity_zip.read("marco/identity.json"))
        split = json.loads(audit.read("marco/runtime/split.json"))
        if split != identity["split"]:
            raise ValueError("Archived held-out split disagrees with identity source")
        eval_ids = list(map(int, split["eval_qids"]))
        train_ids = list(map(int, split["train_qids"]))
        cal_ids = list(map(int, split["cal_qids"]))
        if (len(eval_ids), len(train_ids), len(cal_ids)) != (500, 1000, 200):
            raise ValueError("Unexpected split sizes")
        if len(set(eval_ids + train_ids + cal_ids)) != 1700:
            raise ValueError("Query-ID leakage between train/calibration/evaluation")
        pids = tuple(map(int, identity["pids"]))
        qrels = {int(q): set(map(int, ps)) for q, ps in identity["qrels"].items()}
        qrel_digest = digest_rows([[q, sorted(qrels[q])] for q in eval_ids])
        if qrel_digest != frozen["evaluation_500_qids_qrels_sha256"]:
            raise ValueError("Frozen held-out qrels/order changed")
        if not all(qrels[q] for q in eval_ids):
            raise ValueError("Query without qrels")
        if len(set().union(*(qrels[q] for q in eval_ids))) != 535:
            raise ValueError("Original evaluation qrel population changed")

        def normalized(q):
            return " ".join(identity["queries"][str(q)].casefold().split())

        train_texts = {normalized(q) for q in train_ids}
        cal_texts = {normalized(q) for q in cal_ids}
        eval_texts = {normalized(q) for q in eval_ids}
        overlaps = {
            "train_eval": len(train_texts & eval_texts),
            "calibration_eval": len(cal_texts & eval_texts),
        }
        if any(overlaps.values()):
            raise ValueError("Exact normalized query-text overlap across held-out boundary: "
                             + json.dumps(overlaps, sort_keys=True))

        results = json.loads(audit.read("marco/teacher_sanity/RESULTS.json"))
        if results["status"] != "completed":
            raise ValueError("Original pilot not completed")
        source_code = audit.read("marco/teacher_sanity.py")
        if hashlib.sha256(source_code).hexdigest() != recipe["source_teacher_runner_sha256"]:
            raise ValueError("Historical held-out evaluator source changed")

        summaries = {}
        ranking_hashes = {}
        for arm in ARMS:
            rank_path = Path(rankings_root) / f"ranking_{arm}.npy"
            ranking, ranking_hashes[arm] = read_saved_rankings(rank_path)
            records, summary = evaluate_ranked_ids(
                eval_ids, ranking, qrels, pids,
                expected_eval_qrels_sha256=frozen["evaluation_500_qids_qrels_sha256"],
            )
            csv_rows = list(csv.DictReader(io.StringIO(
                audit.read(f"marco/teacher_sanity/eval_{arm}.csv").decode("utf-8")
            )))
            if len(csv_rows) != QUERY_COUNT:
                raise ValueError(f"{arm}: unexpected archived evaluation CSV length")
            for i, (observed, archived) in enumerate(zip(records, csv_rows)):
                if archived["branch"] != arm or int(archived["query_id"]) != observed["query_id"]:
                    raise ValueError(f"{arm}: query order/identity mismatch at row {i}")
                if int(archived["first_relevant_rank_in1000"]) != observed["first_relevant_rank_in1000"]:
                    raise ValueError(f"{arm}: first relevant rank mismatch at row {i}")
                for metric in METRICS:
                    if abs(float(archived[metric]) - observed[metric]) > 1e-12:
                        raise ValueError(f"{arm}: {metric} mismatch at row {i}")
            reference = (results["baseline_eval"] if arm == "initial"
                         else results["branches"][arm]["eval"])
            for metric in METRICS:
                if abs(summary[metric] - reference[metric]) > 1e-12:
                    raise ValueError(f"{arm}: archived aggregate {metric} mismatch")
            summaries[arm] = summary

    return {
        "task_id": frozen["task_id"],
        "status": "ARCHIVED_RAW_RANKINGS_EVALUATOR_PARITY_PASS_NO_MODEL_RUN",
        "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "data_freeze_sha256": hashlib.sha256(frozen_bytes).hexdigest(),
        "eval_ordered_qrels_sha256": qrel_digest,
        "saved_ranking_array_sha256": ranking_hashes,
        "evaluation_queries": QUERY_COUNT,
        "corpus_passages": CORPUS_COUNT,
        "rank_depth": RANK_DEPTH,
        "evaluated_original_query_rows": QUERY_COUNT * len(ARMS),
        "verified_per_query_cells": QUERY_COUNT * len(ARMS) * (2 + len(METRICS)),
        "relevance_judgments_in_diagnostic_corpus": sum(len(qrels[q]) for q in eval_ids),
        "query_text_overlap": overlaps,
        "archived_per_query_and_aggregate_parity": summaries,
        "scientific_scope": "Original positive-enriched 50k diagnostic cohort only; NOT official full MS MARCO",
        "execution_permission": "NO GPU, MODEL INFERENCE, TEACHER SCORING OR TRAINING AUTHORIZED",
    }


def main():
    p = argparse.ArgumentParser()
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="create-once validated summary")
    action.add_argument("--check", action="store_true", help="check existing summary without edits")
    args = p.parse_args()
    result = validate_sources()
    content = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write:
        with RESULT.open("x", encoding="utf-8") as f:
            f.write(content)
    elif RESULT.read_text(encoding="utf-8") != content:
        raise ValueError("Saved held-out evaluator preflight changed")
    print("REAL_RANKING_EVALUATOR_PARITY PASS", result["evaluated_original_query_rows"],
          "query rows;", result["verified_per_query_cells"], "source cells")
    print("QUERY_TEXT_OVERLAP", json.dumps(result["query_text_overlap"], sort_keys=True))
    print("METRICS", json.dumps(result["archived_per_query_and_aggregate_parity"], sort_keys=True))
    print("NO_MODEL_RUN; EXACT TEACHER-SCORING/TRAINING STILL NO-GO")


if __name__ == "__main__":
    main()
