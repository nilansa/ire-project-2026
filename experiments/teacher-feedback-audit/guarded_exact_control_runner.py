#!/usr/bin/env python3
"""Authorization-disabled integration of frozen exact MS MARCO training/evaluation inputs.

Uses the original 12 October evidence, never generated or synthetic candidates.
No model loading, teacher inference, embedding, optimizer or training entry point.
An authorised future trainer can consume FrozenControlPlan.batches and call
FrozenControlPlan.loss_for_logits; only genuinely full-corpus final rankings
may be passed to FrozenControlPlan.evaluate_exact_rankings.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

from exact_teacher_cache_gap import npy_matrix
from exact_control_objectives import (ARCHIVE, EXPECTED_ZIP_SHA256,
                                      FREEZE, SPEC, query_loss)
from freeze_exact_positive_control import digest_rows
from validate_heldout_exact_evaluator import evaluate_ranked_ids, RESULT as EVAL_RECORD


@dataclass(frozen=True)
class Slate:
    query_id: int
    passage_ids: tuple[int, ...]  # positive at position zero


@dataclass
class FrozenControlPlan:
    slates: tuple[Slate, ...]
    cached_raw_logits: dict[tuple[int, int], float]
    missing_teacher_pairs: tuple[tuple[int, int], ...]
    eval_qids: tuple[int, ...]
    eval_qrels: dict[int, set[int]]
    corpus_ids: tuple[int, ...]
    eval_qrels_sha256: str

    def batches(self, objective):
        """Return exactly 100 fixed batches of eight query slates, no model work."""
        if objective not in ("hard_positive", "teacher_soft"):
            raise ValueError("Unexpected training objective")
        if objective == "teacher_soft" and self.missing_teacher_pairs:
            raise PermissionError(
                f"Teacher-soft control blocked: {len(self.missing_teacher_pairs)} "
                "unscored pairs; no acquisition or inference authorized")
        if len(self.slates) != 800:
            raise ValueError("Frozen query schedule is incomplete")
        return tuple(tuple(self.slates[i:i + 8]) for i in range(0, 800, 8))

    def loss_for_logits(self, slate, student_logits, objective):
        """Wires the validated loss only; never computes student logits."""
        if objective == "hard_positive":
            return query_loss(student_logits, objective="hard_positive")
        if objective != "teacher_soft":
            raise ValueError("Unknown objective")
        import torch  # imported only when an authorised trainer calls this method
        logits = []
        for pid in slate.passage_ids:
            key = (slate.query_id, pid)
            if key not in self.cached_raw_logits:
                raise PermissionError("Missing paid teacher score for a frozen slate")
            logits.append(self.cached_raw_logits[key])
        raw = torch.tensor(logits, dtype=torch.float64, device=student_logits.device)
        return query_loss(student_logits, objective="teacher_soft",
                          teacher_raw_logits=raw)

    def evaluate_exact_rankings(self, ranked_passage_ids):
        """Only post-training exhaustive full-50k top-1000 rankings are accepted."""
        return evaluate_ranked_ids(
            self.eval_qids, ranked_passage_ids, self.eval_qrels,
            self.corpus_ids, expected_eval_qrels_sha256=self.eval_qrels_sha256)

    def source_locked_model_inputs(self):
        """Original texts + frozen encoding contract; never loads a model."""
        from source_locked_model_io import load_model_inputs
        return load_model_inputs(self)

    def train(self, *args, **kwargs):
        raise PermissionError("TRAINING DISABLED: explicit authorization required")

    def acquire_teacher_scores(self, *args, **kwargs):
        raise PermissionError("TEACHER INFERENCE DISABLED: explicit authorization required")


def build_plan(archive=ARCHIVE):
    archive_bytes = Path(archive).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != EXPECTED_ZIP_SHA256:
        raise ValueError("Not the verified 12 October source archive")
    frozen_bytes = FREEZE.read_bytes()
    freeze = json.loads(frozen_bytes)
    recipe = json.loads(SPEC.read_text(encoding="utf-8"))
    eval_record = json.loads(EVAL_RECORD.read_text(encoding="utf-8"))
    if freeze["status"] != "FROZEN_PROTOCOL_NOT_EXECUTED":
        raise ValueError("Data freeze is not immutable")
    if recipe["execution_permission"] != "NO INFERENCE, TRAINING, OR TEACHER SCORING AUTHORIZED":
        raise PermissionError("Unexpected execution permission")
    if recipe["frozen_data_manifest_sha256"] != hashlib.sha256(frozen_bytes).hexdigest():
        raise ValueError("Training recipe/data manifest mismatch")
    if eval_record["data_freeze_sha256"] != hashlib.sha256(frozen_bytes).hexdigest():
        raise ValueError("Independently validated held-out evaluator has another cohort")
    if eval_record["eval_ordered_qrels_sha256"] != freeze["evaluation_500_qids_qrels_sha256"]:
        raise ValueError("Held-out qrel hash mismatch")
    if eval_record["status"] != "ARCHIVED_RAW_RANKINGS_EVALUATOR_PARITY_PASS_NO_MODEL_RUN":
        raise ValueError("Saved real-ranking evaluator parity has not passed")

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
        pilot_bytes = outer.read("inputs/ire-audit-evidence.zip")
        identities_bytes = outer.read("inputs/identity-and-search-evidence.zip")
    if hashlib.sha256(pilot_bytes).hexdigest() != freeze["source_zip_sha256"]:
        raise ValueError("Nested pilot source changed")
    with zipfile.ZipFile(io.BytesIO(pilot_bytes)) as pilot, (
         zipfile.ZipFile(io.BytesIO(identities_bytes))) as identities:
        identity = json.loads(identities.read("marco/identity.json"))
        split = json.loads(pilot.read("marco/runtime/split.json"))
        if identity["split"] != split:
            raise ValueError("Two original split records disagree")
        qrels = {int(q): set(map(int, ps))
                 for q, ps in identity["qrels"].items()}
        pids = tuple(map(int, identity["pids"]))
        corpus = set(pids)
        train_qids = list(map(int, split["train_qids"]))
        eval_qids = tuple(map(int, split["eval_qids"]))
        if (len(train_qids), len(eval_qids), len(pids)) != (1000, 500, 50000):
            raise ValueError("Original cohort size changed")
        run = json.loads(pilot.read("marco/teacher_sanity/train_hnsw.json"))
        run_other = json.loads(pilot.read("marco/teacher_sanity/train_ivf.json"))
        consumed = list(map(int, run["training_query_ids"]))
        if consumed != list(map(int, run_other["training_query_ids"])):
            raise ValueError("Original 800-query schedule differs between arms")
        if digest_rows([[q] for q in consumed]) != freeze["train_schedule_800_qids_sha256"]:
            raise ValueError("Trained query schedule differs from locked source")
        by_q = dict(zip(train_qids, npy_matrix(pilot.read("marco/runtime/neg_flat.npy"))))
        cache = {}
        rows = csv.DictReader(io.StringIO(
            pilot.read("marco/teacher_sanity/teacher_scores.csv").decode("utf-8")))
        for row in rows:
            key = (int(row["query_id"]), int(row["passage_id"]))
            raw = float(row["raw_teacher_logit"])
            if key in cache or not math.isfinite(raw):
                raise ValueError("Teacher cache duplicates or nonfinite scores")
            cache[key] = raw
        if len(cache) != 32312:
            raise ValueError("Not the original teacher-scored pair cache")
        slates = []
        missing = []
        for q in consumed:
            pos = sorted(qrels[q] & corpus)[0]
            docs = tuple(pids[ix] for ix in by_q[q])
            if len(docs) != len(set(docs)) or len(docs) != 20:
                raise ValueError("Wrong exact selected slate")
            if any(pid in qrels[q] for pid in docs) or pos in docs:
                raise ValueError("A known positive appears among selected negatives")
            slate = Slate(q, (pos, *docs))
            if (q, pos) not in cache:
                raise ValueError("Original teacher known positive missing")
            slates.append(slate)
            missing.extend((q, pid) for pid in slate.passage_ids
                           if (q, pid) not in cache)
        ordered_pairs = [(s.query_id, pid) for s in slates for pid in s.passage_ids]
        if digest_rows(ordered_pairs) != freeze["train_exact_ordered_16800_pairs_sha256"]:
            raise ValueError("Real exact selected ID order differs from protocol")
        if digest_rows(sorted(missing)) != freeze["missing_6518_pairs_sha256"]:
            raise ValueError("Teacher-cache gap identities differ from protocol")
        if (len(slates), len(ordered_pairs), len(missing)) != (800, 16800, 6518):
            raise ValueError("Training slates or exact teacher charge changed")
        eval_digest = digest_rows([[q, sorted(qrels[q])] for q in eval_qids])
        if eval_digest != freeze["evaluation_500_qids_qrels_sha256"]:
            raise ValueError("Held-out evaluation query/qrel mapping changed")
        return FrozenControlPlan(
            tuple(slates), cache, tuple(sorted(missing)), eval_qids, qrels, pids,
            eval_digest)


def check():
    plan = build_plan()
    batches = plan.batches("hard_positive")
    if len(batches) != 100 or any(len(b) != 8 for b in batches):
        raise ValueError("Cannot reproduce fixed training update batches")
    if tuple(s.query_id for b in batches for s in b) != tuple(s.query_id for s in plan.slates):
        raise ValueError("Batch materialization changed training order")
    try:
        plan.batches("teacher_soft")
    except PermissionError as e:
        if "6518" not in str(e):
            raise
    else:
        raise AssertionError("Incomplete teacher-soft arm must fail closed")
    for forbidden in (plan.train, plan.acquire_teacher_scores):
        try:
            forbidden()
        except PermissionError:
            pass
        else:
            raise AssertionError("Expensive execution gate unexpectedly open")
    if len(plan.eval_qids) != 500 or len(plan.corpus_ids) != 50000:
        raise ValueError("Evaluation contract incomplete")
    return {
        "status": "FROZEN_SLATES_OBJECTIVES_EVALUATOR_WIRED; EXECUTION DISABLED",
        "hard_positive": "100 fixed 8-query batches available; zero teacher inputs",
        "teacher_soft": "BLOCKED: 6518 previously unscored pairs; fixed charge 16800",
        "heldout": "500 qrel-locked queries, 50000 passages, exact top1000 required",
        "model_inference": "NONE", "optimizer_steps": 0,
        "teacher_calls": 0, "rankings_generated": 0,
        "future_training_implementation": "NOT AUTHORIZED OR EXECUTED",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    args = parser.parse_args()
    verified = check()
    print("EXACT_CONTROL_RUNNER_PREFLIGHT PASS")
    print(json.dumps(verified, sort_keys=True))
