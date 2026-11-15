#!/usr/bin/env python3
"""Dormant, source-matched MS MARCO student training and exact-evaluation adapter.

This file is implementation only, NOT an experimental run or authorization.
The CLI permits --check exclusively: no checkpoint loading, tokenization,
forward pass, optimization, corpus encoding, or new teacher scoring.

An authorized future caller must explicitly pass execution_authorized=True
to run_authorized(), a LOCAL verified student snapshot and a NEW output path.
Teacher-soft cannot start with the 6,518 missing frozen teacher logits.
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
from validated_teacher_gap_extension import attach_verified_extension

INITIAL_HASH = "e6f42f74ca80eb282b5b7fa68df335eb4f57586227c601d0c60da85378e954bd"
ALLOWED = ("hard_positive", "teacher_soft")


def state_hash(model):
    """Byte-compatible with the authenticated original teacher_sanity.py."""
    h = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def original_forward(model, texts, max_length, device):
    """Original fixed-pad tokenizer and sentence_embedding, not model.encode."""
    tokens = model.tokenizer(
        list(texts), padding="max_length", truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    return model({k: v.to(device) for k, v in tokens.items()})["sentence_embedding"]


def final_encode(model, texts, max_length, device):
    """Fresh full-corpus/query encoding from the final trained model state."""
    import numpy as np

    old_length = model.max_seq_length
    try:
        model.max_seq_length = max_length
        encoded = model.encode(
            list(texts), batch_size=256, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False, device=device
        )
    finally:
        model.max_seq_length = old_length
    arr = np.asarray(encoded, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts) or arr.shape[1] != 768:
        raise ValueError("Unexpected final embedding dimensions")
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite final embeddings")
    return arr


def exhaustive_top1000_ids(query_vectors, passage_vectors, passage_ids):
    """Score ALL 50k passages, with deterministic global tie order.

    Unlike selecting an arbitrary top-k with argpartition, this full lexsort
    resolves a score tie at the k-th boundary by the original corpus row.
    This is a strict exact-ranking implementation, not a claim that the
    historical runner's tie behavior was identical at every boundary.
    """
    import numpy as np

    if query_vectors.shape != (500, 768):
        raise ValueError("Wrong held-out query embedding shape")
    if passage_vectors.shape != (50000, 768):
        raise ValueError("Wrong held-out passage embedding shape")
    if len(passage_ids) != 50000 or len(set(passage_ids)) != 50000:
        raise ValueError("Wrong frozen corpus IDs")
    document_ids = np.asarray(passage_ids, dtype=np.int64)
    index = np.arange(50000, dtype=np.int64)
    transposed = passage_vectors.T
    ranks = np.empty((500, 1000), dtype=np.int64)
    for start in range(0, 500, 16):
        scores = query_vectors[start:start + 16] @ transposed
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite exhaustive inner-product scores")
        for j, row in enumerate(scores):
            order = np.lexsort((index, -row))
            ranks[start + j] = document_ids[order[:1000]]
    return ranks


def run_authorized(*, student_snapshot_dir, output_dir, objective="hard_positive",
                   device="cpu", execution_authorized=False,
                   teacher_score_extension_path=None):
    """Execute ONE arm only after an explicit future approval.

    The scheduled check must NEVER call this with execution_authorized=True.
    There is no --run CLI argument. Teacher-soft fails closed until all 21
    frozen pair logits per query are present, preserving the PRE-TEACHER slate.
    """
    if execution_authorized is not True:
        raise PermissionError("NO USER AUTHORIZATION: model execution disabled")
    if objective not in ALLOWED:
        raise ValueError("Unknown frozen experimental objective")
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Explicit cpu/mps/cuda device required")

    plan = build_plan()
    extension_provenance = None
    if objective == "teacher_soft" and teacher_score_extension_path is not None:
        # Read-only merge of a complete, identity-checked FUTURE score extension.
        # It cannot overwrite historical cached rows or the immutable archive.
        plan, extension_provenance = attach_verified_extension(
            plan, teacher_score_extension_path)
    elif objective == "hard_positive" and teacher_score_extension_path is not None:
        raise ValueError("Zero-teacher control cannot ingest teacher-score extensions")
    batches = plan.batches(objective)  # remains blocked without all 6,518 logits
    inputs = load_model_inputs(plan)
    recipe = json.loads(SPEC.read_text(encoding="utf-8"))
    if recipe["student_initial_state_sha256"] != INITIAL_HASH:
        raise ValueError("Pinned student initialization changed")
    settings = recipe["historical_optimizer_and_encoding_settings"]
    expected = {
        "seed": 20260907, "updates": 100, "batch_queries": 8,
        "trained_queries": 800, "query_length": 32, "passage_length": 128,
        "lr": 2e-5, "weight_decay": 0.01, "betas": [0.9, 0.999],
        "eps": 1e-8, "clip_norm": 1.0, "precision": "float32",
        "fixed_training_padding": True
    }
    if any(settings.get(k) != value for k, value in expected.items()):
        raise ValueError("Source-locked execution recipe changed")
    snapshot = Path(student_snapshot_dir).expanduser()
    destination = Path(output_dir).expanduser()
    if not snapshot.is_dir():
        raise FileNotFoundError("Pinned student snapshot is not locally available")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite or resume an existing run")
    if snapshot.resolve() in destination.resolve().parents:
        raise ValueError("Output must not be inside the original checkpoint")
    # Parameter-state hashes alone do not pin tokenizer or pooling semantics.
    snapshot_metadata = verify_ance_snapshot_metadata(snapshot)

    # No network fetch is permitted. Set before importing model libraries.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings["seed"])
    model = SentenceTransformer(
        str(snapshot), device=device, local_files_only=True
    )
    original = state_hash(model)
    if original != INITIAL_HASH:
        raise ValueError("Student state differs from historical ANCE init")
    if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Expected original float32 trainable weights")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"],
        weight_decay=settings["weight_decay"],
        betas=tuple(settings["betas"]), eps=settings["eps"]
    )
    model.train()
    losses, gradient_norms = [], []
    for step, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for slate in batch:
            query_text, passages = inputs.slate_texts(slate)
            q = original_forward(model, [query_text], 32, device)
            d = original_forward(model, passages, 128, device)
            if q.shape != (1, 768) or d.shape != (21, 768):
                raise ValueError("Unexpected student training embedding shape")
            logits = (q * d).sum(-1)
            loss = plan.loss_for_logits(slate, logits, objective)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite loss")
            (loss / 8).backward()
            mean_loss += float(loss.detach().cpu()) / 8
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite gradient norm")
        optimizer.step()
        losses.append(mean_loss)
        gradient_norms.append(float(norm.detach().cpu()))
    if len(losses) != 100:
        raise ValueError("Incomplete frozen optimizer schedule")

    model.eval()
    final = state_hash(model)
    if final == original:
        raise ValueError("Parameters did not change after 100 updates")
    eval_texts, corpus_texts = inputs.evaluation_inputs()
    passage_vectors = final_encode(model, corpus_texts, 128, device)
    query_vectors = final_encode(model, eval_texts, 32, device)
    ranking = exhaustive_top1000_ids(
        query_vectors, passage_vectors, plan.corpus_ids
    )
    records, metrics = plan.evaluate_exact_rankings(ranking.tolist())

    # Save only after full evaluation passes; destination must be a NEW folder.
    destination.mkdir(parents=True, exist_ok=False)
    model.save(str(destination / "student_final"))
    np.save(destination / "heldout_500x1000_passage_ids.npy", ranking)
    with (destination / "heldout_metrics.csv").open("x", newline="") as f:
        cols = list(records[0])
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(records)
    record = {
        "task_id": "ire-mid-checkpoint-20261022-001",
        "status": "TRAINED_AND_EVALUATED_AFTER_EXPLICIT_AUTHORIZATION",
        "objective": objective,
        "initial_state_sha256": original,
        "final_state_sha256": final,
        "pinned_student_snapshot_metadata": snapshot_metadata,
        "frozen_training_pairs_sha256": recipe["frozen_training_pairs_sha256"],
        "frozen_eval_qrels_sha256": recipe["frozen_eval_qrels_sha256"],
        "training_query_ids": [s.query_id for b in batches for s in b],
        "optimizer_updates": len(losses),
        "losses": losses,
        "gradient_norms_before_clip": gradient_norms,
        "teacher_labels_charged": recipe["objectives"][objective]["teacher_pairs_charged"],
        "teacher_extension_provenance": extension_provenance,
        "evaluation": metrics,
        "corpus_count": 50000, "evaluation_query_count": 500,
        "evaluation_scope": "positive-enriched 50k diagnostic, not official MS MARCO"
    }
    (destination / "run.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    return record


def check():
    """Source-based implementation preflight only. No model import or forward."""
    plan = build_plan()
    inputs = load_model_inputs(plan)
    if len(plan.batches("hard_positive")) != 100:
        raise ValueError("Frozen 100-update schedule unavailable")
    if len(inputs.corpus_texts) != 50000 or len(plan.eval_qids) != 500:
        raise ValueError("Full 50k qrel-evaluation inputs unavailable")
    if len(plan.missing_teacher_pairs) != 6518:
        raise ValueError("The teacher-score consumption gap has changed")
    try:
        plan.batches("teacher_soft")
    except PermissionError:
        pass
    else:
        raise ValueError("Unexpected teacher-soft execution readiness")
    try:
        run_authorized(student_snapshot_dir="/nonexistent", output_dir="/nonexistent")
    except PermissionError:
        pass
    else:
        raise ValueError("Model execution authorization guard failed")
    return {
        "status": "STATIC_SOURCE_LOCKED_TRAIN_AND_EVAL_ADAPTER_ONLY",
        "hard_positive": "IMPLEMENTED_BUT_NOT_EXECUTED",
        "teacher_soft": "BLOCKED_ON_6518_SCORES",
        "exact_evaluation": "FULL_50K_REENCODE_UNRUN",
        "model_loads": 0, "model_forward_calls": 0,
        "teacher_inference_calls": 0, "optimizer_updates": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    result = check()
    print("SOURCE_LOCKED_TRAIN_EVAL_ADAPTER_PREFLIGHT PASS")
    print(json.dumps(result, sort_keys=True))
