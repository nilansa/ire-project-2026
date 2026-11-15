#!/usr/bin/env python3
"""Dormant untouched-ANCE evaluator reusing the trained arm's exact path.

Only --check is callable from the CLI; no model load, inference or write
occurs in this preflight. Future execution requires separate explicit approval.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path
from exact_control_objectives import SPEC
from guarded_exact_control_runner import build_plan
from source_locked_model_io import load_model_inputs
from pinned_ance_snapshot_contract import verify_ance_snapshot_metadata
from source_locked_train_and_evaluate import (
    INITIAL_HASH, state_hash, final_encode, exhaustive_top1000_ids,
)


def run_baseline_authorized(*, student_snapshot_dir, output_dir,
                            device="cpu", execution_authorized=False):
    """Zero-update exact retrieval; cannot run without a new authorization."""
    if execution_authorized is not True:
        raise PermissionError("NO USER AUTHORIZATION: encoder inference disabled")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Explicit cpu/mps/cuda device required")
    plan = build_plan()
    inputs = load_model_inputs(plan)
    recipe = json.loads(SPEC.read_text(encoding="utf-8"))
    if recipe["student_initial_state_sha256"] != INITIAL_HASH:
        raise ValueError("Frozen ANCE initialization contract changed")
    if recipe["frozen_eval_qrels_sha256"] != plan.eval_qrels_sha256:
        raise ValueError("Held-out evaluation qrel contract changed")
    snapshot = Path(student_snapshot_dir).expanduser()
    destination = Path(output_dir).expanduser()
    if not snapshot.is_dir():
        raise FileNotFoundError("Pinned student checkpoint is not locally available")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite any prior run")
    if snapshot.resolve() in destination.resolve().parents:
        raise ValueError("Output inside the original checkpoint is forbidden")
    # Verify tokenizer/module metadata before loading the untouched baseline.
    snapshot_metadata = verify_ance_snapshot_metadata(snapshot)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    seed = recipe["historical_optimizer_and_encoding_settings"]["seed"]
    if seed != 20260907:
        raise ValueError("Original seed contract changed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = SentenceTransformer(str(snapshot), device=device, local_files_only=True)
    initial = state_hash(model)
    if initial != INITIAL_HASH:
        raise ValueError("Snapshot is not untouched original ANCE")
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Float32 student model expected")
    model.eval()  # Same evaluation mode as trained-arm source.
    eval_texts, corpus_texts = inputs.evaluation_inputs()
    passage_vectors = final_encode(model, corpus_texts, 128, device)
    query_vectors = final_encode(model, eval_texts, 32, device)
    ranking = exhaustive_top1000_ids(query_vectors, passage_vectors,
                                     plan.corpus_ids)
    records, metrics = plan.evaluate_exact_rankings(ranking.tolist())
    if len(records) != 500 or state_hash(model) != initial:
        raise ValueError("Evaluation incomplete or untouched checkpoint mutated")

    # Evaluation must pass before creating a NEW output folder.
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / "heldout_500x1000_passage_ids.npy", ranking)
    with (destination / "heldout_metrics.csv").open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    record = {
        "task_id": "ire-mid-checkpoint-20261022-001",
        "status": "UNTOUCHED_ANCE_EVALUATED_AFTER_SEPARATE_AUTHORIZATION",
        "objective": "untouched_ance_zero_update",
        "initial_state_sha256": initial,
        "final_state_sha256": initial,
        "pinned_student_snapshot_metadata": snapshot_metadata,
        "source_slice_sha256": inputs.source_slice_sha256,
        "frozen_eval_qrels_sha256": recipe["frozen_eval_qrels_sha256"],
        "frozen_training_pairs_sha256": recipe["frozen_training_pairs_sha256"],
        "optimizer_updates": 0,
        "teacher_labels_acquired": 0,
        "teacher_labels_consumed": 0,
        "evaluation": metrics,
        "corpus_count": len(corpus_texts),
        "evaluation_query_count": len(eval_texts),
        "evaluation_scope": "positive-enriched 50k diagnostic, not official MS MARCO",
        "shared_evaluator_source_sha256": hashlib.sha256(
            Path(__file__).with_name("source_locked_train_and_evaluate.py").read_bytes()
        ).hexdigest(),
    }
    with (destination / "run.json").open("x", encoding="utf-8") as file:
        json.dump(record, file, sort_keys=True, indent=2)
        file.write("\n")
    return record


def check():
    """Strictly metadata-only static contract/authorization preflight."""
    import source_locked_train_and_evaluate as trained
    if (final_encode is not trained.final_encode or
        exhaustive_top1000_ids is not trained.exhaustive_top1000_ids or
        state_hash is not trained.state_hash):
        raise ValueError("Evaluation code is not identical to trained-arm path")
    plan = build_plan()
    inputs = load_model_inputs(plan)
    recipe = json.loads(SPEC.read_text(encoding="utf-8"))
    if recipe["student_initial_state_sha256"] != INITIAL_HASH:
        raise ValueError("Original ANCE checkpoint hash contract changed")
    if recipe["frozen_eval_qrels_sha256"] != plan.eval_qrels_sha256:
        raise ValueError("Frozen qrel mapping changed")
    if len(plan.eval_qids) != 500 or len(inputs.corpus_ids) != 50000:
        raise ValueError("Original evaluation cohort changed")
    try:
        run_baseline_authorized(student_snapshot_dir="/not-loaded",
                                output_dir="/not-written")
    except PermissionError:
        pass
    else:
        raise AssertionError("Unauthorized encoder evaluation unexpectedly allowed")
    return {
        "status": "SAME_PATH_UNTOUCHED_BASELINE_STATIC_ONLY",
        "frozen_eval_queries": len(plan.eval_qids),
        "frozen_corpus_passages": len(inputs.corpus_ids),
        "initial_state_sha256": INITIAL_HASH,
        "shared_evaluation_code": "source_locked_train_and_evaluate.py",
        "model_loads": 0,
        "model_forward_calls": 0,
        "teacher_scoring_calls": 0,
        "optimizer_updates": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print("SAME_PATH_UNTOUCHED_BASELINE_PREFLIGHT PASS")
    print(json.dumps(check(), sort_keys=True))
