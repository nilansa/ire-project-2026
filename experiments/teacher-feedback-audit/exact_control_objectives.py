#!/usr/bin/env python3
"""Source-locked loss contract for the MS MARCO exact-retrieval positive control.

This module never loads a model, scores a passage, or updates parameters.
Future training must separately implement frozen slate loading, provenance
checks, optimizer steps, checkpointing and qrel-based held-out evaluation.
"""

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

TASK = "ire-mid-checkpoint-20261022-001"
HERE = Path(__file__).resolve().parent
ARCHIVE = (Path.home() / ".codex/scratch/cross-surface/ire-project" / TASK /
           "shared/incoming/IRE_Audit_Reproducibility_2026-10-12.zip")
EXPECTED_ZIP_SHA256 = "b51dea1c6dfe9c61fa7d245eeed004d9bfb347b3cf7e1a38ba1d665924426115"
FREEZE = HERE / "exact_positive_control_freeze.json"
SPEC = HERE / "exact_execution_recipe.json"
SLATE_SIZE = 21
POSITIVE_INDEX = 0

# The original runner trained on exactly 100 batches of eight query slates.
# Hard-positive is a ZERO-teacher-call baseline, not an equal-label comparator.
OBJECTIVES = ("hard_positive", "teacher_soft")


def query_loss(student_logits, *, objective, teacher_raw_logits=None):
    """Compute one 21-way loss without invoking a model or an optimizer.

    student_logits has passage order [known positive, 20 frozen negatives].
    Teacher targets follow original teacher_sanity.py: float64 softmax of raw
    unactivated MiniLM logits, then cast to float32 before the student loss.
    For hard_positive, teacher labels are forbidden entirely.
    """
    import torch
    import torch.nn.functional as F

    if objective not in OBJECTIVES:
        raise ValueError("Unknown objective")
    if not isinstance(student_logits, torch.Tensor) or student_logits.ndim != 1:
        raise ValueError("Expected one tensor of 21 student passage logits")
    if student_logits.numel() != SLATE_SIZE or not student_logits.is_floating_point():
        raise ValueError("The frozen slate must contain exactly 21 float logits")
    if not bool(torch.isfinite(student_logits).all()):
        raise ValueError("Nonfinite student logits")

    if objective == "hard_positive":
        if teacher_raw_logits is not None:
            raise ValueError("Hard-positive control must not access teacher logits")
        # Equal to cross_entropy(student_logits[None, :], class_zero).
        return -F.log_softmax(student_logits, dim=-1)[POSITIVE_INDEX]

    if teacher_raw_logits is None:
        raise ValueError("Teacher-soft loss requires all 21 pre-acquired raw logits")
    if not isinstance(teacher_raw_logits, torch.Tensor):
        raise ValueError("Teacher logits must be a tensor aligned to the same slate")
    if teacher_raw_logits.ndim != 1 or teacher_raw_logits.numel() != SLATE_SIZE:
        raise ValueError("Incomplete or misaligned teacher slate")
    if not bool(torch.isfinite(teacher_raw_logits).all()):
        raise ValueError("Nonfinite teacher logits")

    # Original: np.asarray(scores, dtype=float64) -> exp-normalize -> torch.float32.
    raw = teacher_raw_logits.detach().to(dtype=torch.float64)
    weights = torch.softmax(raw, dim=-1).to(dtype=torch.float32).detach()
    return -(weights * F.log_softmax(student_logits, dim=-1)).sum()


def source_locked_execution_spec(archive=ARCHIVE):
    """Recover optimizer/evaluation settings from the verified historical ZIP.

    Does not mutate the earlier DATA freeze. The output is a separate recipe
    contract; recording it does not authorize or launch a training run.
    """
    blob = Path(archive).read_bytes()
    if hashlib.sha256(blob).hexdigest() != EXPECTED_ZIP_SHA256:
        raise ValueError("Historical ZIP hash mismatch")
    frozen_blob = FREEZE.read_bytes()
    frozen = json.loads(frozen_blob)
    if frozen["archive_sha256"] != EXPECTED_ZIP_SHA256:
        raise ValueError("Frozen protocol refers to another source archive")
    if frozen["status"] != "FROZEN_PROTOCOL_NOT_EXECUTED":
        raise ValueError("Original frozen protocol state changed")

    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        inner_bytes = outer.read("inputs/ire-audit-evidence.zip")
    if hashlib.sha256(inner_bytes).hexdigest() != frozen["source_zip_sha256"]:
        raise ValueError("Archived evidence ZIP mismatch")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        results = json.loads(inner.read("marco/teacher_sanity/RESULTS.json"))
        original_source = inner.read("marco/teacher_sanity.py")
    cfg = results["config"]

    for field, expected in {
        "seed": 20260907, "schedule_seed": 20260940,
        "updates": 100, "batch_queries": 8, "trained_queries": 800,
        "temperature": 1.0, "query_length": 32, "passage_length": 128,
        "teacher_max_length": 256, "teacher_batch": 64,
        "precision": "float32", "fixed_training_padding": True,
        "optimizer": "AdamW", "lr": 2e-5, "weight_decay": 0.01,
        "betas": [0.9, 0.999], "eps": 1e-8, "clip_norm": 1.0,
        "deterministic_algorithms_enforced": False,
        "student_checkpoint": frozen["student"],
    }.items():
        if cfg.get(field) != expected:
            raise ValueError("Unexpected historical recipe: " + field)
    if results["teacher_revision"] != frozen["teacher_revision"]:
        raise ValueError("Historical teacher model revision mismatch")
    if (frozen["training_queries"], frozen["per_arm_teacher_labels_charged"],
            frozen["cached_exact_teacher_pairs"], frozen["new_teacher_scores_needed"]) != (
            800, 16800, 10282, 6518):
        raise ValueError("Data freeze budget changed")
    if b"return -(teacher * F.log_softmax(student, dim=-1)).sum()" not in original_source:
        raise ValueError("Archived reference soft-loss signature changed")

    settings = {
        name: cfg[name] for name in (
            "seed", "schedule_seed", "updates", "batch_queries",
            "trained_queries", "temperature", "query_length", "passage_length",
            "teacher_max_length", "teacher_batch", "precision",
            "fixed_training_padding", "optimizer", "lr", "weight_decay",
            "betas", "eps", "clip_norm", "deterministic_algorithms_enforced",
            "student_checkpoint"
        )
    }
    return {
        "task_id": TASK,
        "status": "SOURCE_LOCKED_LOSS_RECIPE_ONLY_NO_MODEL_RUN",
        "source_archive_sha256": EXPECTED_ZIP_SHA256,
        "source_teacher_runner_sha256": hashlib.sha256(original_source).hexdigest(),
        "frozen_data_manifest_sha256": hashlib.sha256(frozen_blob).hexdigest(),
        "frozen_training_pairs_sha256": frozen["train_exact_ordered_16800_pairs_sha256"],
        "frozen_eval_qrels_sha256": frozen["evaluation_500_qids_qrels_sha256"],
        "student_initial_state_sha256": frozen["student_initial_state_sha256"],
        "teacher_revision": frozen["teacher_revision"],
        "historical_optimizer_and_encoding_settings": settings,
        "objectives": {
            "hard_positive": {"known_positive_index": 0, "teacher_pairs_charged": 0},
            "teacher_soft": {"known_positive_index": 0,
                             "teacher_pairs_charged": frozen["per_arm_teacher_labels_charged"],
                             "cached_pairs": frozen["cached_exact_teacher_pairs"],
                             "missing_pairs": frozen["new_teacher_scores_needed"]},
        },
        "evaluation": frozen["evaluation"],
        "execution_permission": "NO INFERENCE, TRAINING, OR TEACHER SCORING AUTHORIZED",
    }


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--write", action="store_true")
    choice.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = json.dumps(source_locked_execution_spec(), indent=2, sort_keys=True) + "\n"
    if args.write:
        with SPEC.open("x", encoding="utf-8") as f:
            f.write(content)
        print("SOURCE_LOCKED_EXECUTION_RECIPE_WRITTEN", SPEC)
    else:
        if SPEC.read_text(encoding="utf-8") != content:
            raise ValueError("Execution recipe drift")
        print("SOURCE_LOCKED_EXECUTION_RECIPE_CHECK PASS")
    print("OBJECTIVES_IMPLEMENTED hard_positive teacher_soft; NO_MODEL_RUN")


if __name__ == "__main__":
    main()
