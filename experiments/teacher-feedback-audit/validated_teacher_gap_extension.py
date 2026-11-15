#!/usr/bin/env python3
"""Fail-closed ingestion contract for FUTURE, separately authorized teacher scores.

This module only reads an existing local JSON file and checks 6,518 frozen
query/passage identities. It neither invokes the teacher nor modifies the
authenticated October 12 score archive. A self-declared model revision is NOT
independent evidence that future logits came from that model.
"""
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

from exact_control_objectives import SPEC


SCHEMA = "ire-exact-teacher-gap-v1"
TEACHER_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
TASK_ID = "ire-mid-checkpoint-20261022-001"


def attach_verified_extension(plan, extension_path):
    """Return an in-memory completed plan, never mutate the frozen original.

    Input is a JSON object with exactly: schema, task_id, teacher_model_id,
    teacher_revision, source_archive_sha256, frozen_training_pairs_sha256,
    pair_acquisition_charge, and rows. Rows must be sorted by (query_id,
    passage_id); each has query_id, passage_id and raw_teacher_logit.

    This validates *file integrity/identity and declared provenance*, not the
    actual model used to generate logits. Independent scoring attestation is
    still required before treating any eventual model result as scientific.
    """
    recipe = json.loads(SPEC.read_text(encoding="utf-8"))
    expected_gap = tuple(sorted(plan.missing_teacher_pairs))
    if len(expected_gap) != 6518:
        raise ValueError("Original exact-arm score gap is no longer frozen")
    if len(plan.cached_raw_logits) != 32312:
        raise ValueError("Original cached teacher-score set changed")
    path = Path(extension_path).expanduser()
    raw_bytes = path.read_bytes()  # existing local artifact only; no network
    payload = json.loads(raw_bytes)
    if not isinstance(payload, dict):
        raise ValueError("Teacher extension must be a JSON object")
    expected_header = {
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "teacher_model_id": TEACHER_ID,
        "teacher_revision": recipe["teacher_revision"],
        "source_archive_sha256": recipe["source_archive_sha256"],
        "frozen_training_pairs_sha256": recipe["frozen_training_pairs_sha256"],
        "pair_acquisition_charge": 16800,
    }
    if set(payload) != set(expected_header) | {"rows"}:
        raise ValueError("Unexpected or missing teacher-extension fields")
    if any(payload[k] != value or type(payload[k]) is not type(value)
           for k, value in expected_header.items()):
        raise ValueError("Teacher model, revision, archive or budget mismatch")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != len(expected_gap):
        raise ValueError("All 6,518 previously unscored teacher rows are required")
    additions = {}
    for pair, row in zip(expected_gap, rows):
        if not isinstance(row, dict) or set(row) != {
                "query_id", "passage_id", "raw_teacher_logit"}:
            raise ValueError("Unexpected teacher-score row format")
        qid, pid = row["query_id"], row["passage_id"]
        if (type(qid) is not int or type(pid) is not int or
                (qid, pid) != pair):
            raise ValueError("Missing, duplicate, out-of-order or extraneous scored pair")
        score = row["raw_teacher_logit"]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Teacher logit must be a finite raw number")
        if pair in plan.cached_raw_logits:
            raise ValueError("Original score rows must never be overwritten")
        additions[pair] = float(score)
    completed = replace(
        plan,
        cached_raw_logits={**plan.cached_raw_logits, **additions},
        missing_teacher_pairs=(),
    )
    if len(completed.batches("teacher_soft")) != 100:
        raise ValueError("Completed score set lacks the original 100 batches")
    provenance = {
        "extension_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "new_physical_pair_count_declared": len(additions),
        "original_cached_score_rows": 32312,
        "teacher_pair_acquisition_charge": 16800,
        "teacher_model_id_declared": TEACHER_ID,
        "teacher_revision_declared": recipe["teacher_revision"],
        "independent_teacher_inference_attestation": "NOT_VERIFIED_BY_THIS_LOADER",
    }
    return completed, provenance
