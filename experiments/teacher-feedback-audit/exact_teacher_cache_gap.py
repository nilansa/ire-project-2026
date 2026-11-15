#!/usr/bin/env python3
"""Read-only cost gate for an exact-candidate, fixed-budget teacher control.

Input: the original, hash-locked 12 October IRE reproducibility ZIP.
Requires only the Python standard library. No models are loaded or scored.
The labels requested by an arm count against its acquisition budget even
when an already-paid-for physical score is safely reused from another arm.
"""

import ast
import csv
import hashlib
import io
import json
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ARCHIVE_SHA256 = "b51dea1c6dfe9c61fa7d245eeed004d9bfb347b3cf7e1a38ba1d665924426115"


def npy_matrix(raw):
    """Read the original little-endian C-order integer .npy matrices."""
    assert raw[:6] == b"\x93NUMPY"
    major = raw[6]
    assert major in (1, 2, 3)
    start, size_format = (10, "<H") if major == 1 else (12, "<I")
    length = struct.unpack_from(size_format, raw, 8)[0]
    header = ast.literal_eval(raw[start : start + length].decode("latin1"))
    assert header["fortran_order"] is False
    assert header["descr"] in ("<i4", "<i8")
    rows, cols = header["shape"]
    assert (rows, cols) == (1000, 20)
    width = 8 if header["descr"] == "<i8" else 4
    assert len(raw) == start + length + rows * cols * width
    entries = struct.unpack_from("<" + ("q" if width == 8 else "i") * (rows * cols), raw, start + length)
    return [entries[i * cols : (i + 1) * cols] for i in range(rows)]


def csv_rows(archive, name):
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


def main(archive_path):
    outer_bytes = Path(archive_path).read_bytes()
    assert hashlib.sha256(outer_bytes).hexdigest() == ARCHIVE_SHA256, "Archive SHA differs"
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read("inputs/ire-audit-evidence.zip"))) as audit:
            with zipfile.ZipFile(io.BytesIO(outer.read("inputs/identity-and-search-evidence.zip"))) as identity_zip:
                identity = json.loads(identity_zip.read("marco/identity.json"))
                split = json.loads(audit.read("marco/runtime/split.json"))
                configuration = json.loads(audit.read("marco/teacher_sanity/RESULTS.json"))["config"]
                hnsw_run = json.loads(audit.read("marco/teacher_sanity/train_hnsw.json"))
                ivf_run = json.loads(audit.read("marco/teacher_sanity/train_ivf.json"))
                assert configuration["updates"] == 100 and configuration["batch_queries"] == 8
                assert configuration["trained_queries"] == 800 and configuration["train_pool_queries"] == 1000
                assert hnsw_run["updates"] == ivf_run["updates"] == 100
                consumed_order = list(map(int, hnsw_run["training_query_ids"]))
                assert consumed_order == list(map(int, ivf_run["training_query_ids"]))
                assert len(consumed_order) == len(set(consumed_order)) == 800

                queries = list(map(int, split["train_qids"]))
                passage_ids = list(map(int, identity["pids"]))
                qrels = {int(q): set(map(int, p)) for q, p in identity["qrels"].items()}
                assert len(queries) == len(set(queries)) == 1000 and len(passage_ids) == 50000
                assert set(consumed_order) <= set(queries)

                negatives = {
                    arm: npy_matrix(audit.read(f"marco/runtime/neg_{arm}.npy"))
                    for arm in ("flat", "hnsw", "ivf")
                }
                cached_rows = csv_rows(audit, "marco/teacher_sanity/teacher_scores.csv")
                cached = {(int(row["query_id"]), int(row["passage_id"])) for row in cached_rows}
                assert len(cached) == len(cached_rows) == 32312
                cached_by_query = defaultdict(set)
                for q, p in cached:
                    cached_by_query[q].add(p)

                # The original scorer charged one positive plus 20 preselected
                # negatives per query. The known positive is already cached.
                positives = {}
                for q in queries:
                    hits = qrels[q] & cached_by_query[q]
                    assert len(hits) == 1, f"Unresolved positive identity for query {q}"
                    positives[q] = next(iter(hits))

                def requested(arm):
                    result = set()
                    for i, q in enumerate(queries):
                        docs = [passage_ids[entry] for entry in negatives[arm][i]]
                        assert len(docs) == len(set(docs)) == 20
                        assert not qrels[q].intersection(docs), f"Relevant negative in {arm} query {q}"
                        result.update((q, doc) for doc in [positives[q], *docs])
                    assert len(result) == 21000
                    return result

                hnsw = requested("hnsw")
                ivf = requested("ivf")
                assert cached == hnsw | ivf, "Original score CSV differs from historical selected slates"
                exact = requested("flat")
                consumed = set(consumed_order)
                training_exact = {(q, p) for q, p in exact if q in consumed}
                assert len(training_exact) == 16800

                required_all = exact - cached
                required_training = training_exact - cached
                assert not any(p == positives[q] for q, p in required_all)
                by_query = Counter(q for q, _ in required_all)
                zero_missing_training = sum(by_query[q] == 0 for q in consumed)

                result = {
                    "source_archive_sha256": ARCHIVE_SHA256,
                    "case": "original_50k_MS_MARCO_ANCE_teacher_pilot_only",
                    "historical_teacher_cache_unique_pairs": len(cached),
                    "historical_hnsw_ivf_score_union_verified": True,
                    "exact_pre_teacher_rule": "Original exact top-200 -> stored selector-selected 20 negatives + one known positive",
                    "training_schedule": "Original common 800 query IDs, 100 updates x 8 queries",
                    "all_1000_queries": {
                        "requested_pair_budget": len(exact),
                        "physically_cached_pairs": len(exact & cached),
                        "previously_unscored_pairs": len(required_all),
                    },
                    "training_800_queries": {
                        "requested_pair_budget": len(training_exact),
                        "physically_cached_pairs": len(training_exact & cached),
                        "previously_unscored_pairs": len(required_training),
                        "cached_fraction": round(len(training_exact & cached) / len(training_exact), 8),
                        "queries_with_zero_new_scores": zero_missing_training,
                    },
                    "scope_limit": "Existing cache does NOT provide all teacher logits needed for a fixed 20-negative exact arm; no teacher inference or student training executed.",
                }
                assert result["training_800_queries"]["previously_unscored_pairs"] <= result["all_1000_queries"]["previously_unscored_pairs"]
                print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 -B exact_teacher_cache_gap.py /path/to/IRE_Audit_Reproducibility_2026-10-12.zip")
    main(sys.argv[1])
