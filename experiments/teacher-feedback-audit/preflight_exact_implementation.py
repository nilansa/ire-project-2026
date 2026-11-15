#!/usr/bin/env python3
"""Read-only implementation contract audit for the frozen MS MARCO exact control.

Inspect verified archived source, not synthetic selector data. Never load a model,
score a passage, train, rewrite the freeze, or modify the transfer index.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path


TASK = "ire-mid-checkpoint-20261022-001"
HERE = Path(__file__).resolve().parent
ARCHIVE = (Path.home() / ".codex/scratch/cross-surface/ire-project" / TASK /
           "shared/incoming/IRE_Audit_Reproducibility_2026-10-12.zip")
EXPECTED_ARCHIVE_SHA = "b51dea1c6dfe9c61fa7d245eeed004d9bfb347b3cf7e1a38ba1d665924426115"


def main():
    freeze = json.loads((HERE / "exact_positive_control_freeze.json").read_text())
    assert freeze["task_id"] == TASK
    assert freeze["status"] == "FROZEN_PROTOCOL_NOT_EXECUTED"
    assert (freeze["training_queries"], freeze["updates"], freeze["batch_queries"]) == (800, 100, 8)
    assert freeze["per_query_positive_plus_negatives"] == [1, 20]
    assert freeze["per_arm_teacher_labels_charged"] == 16800
    assert freeze["cached_exact_teacher_pairs"] + freeze["new_teacher_scores_needed"] == 16800

    raw = ARCHIVE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_ARCHIVE_SHA
    assert freeze["archive_sha256"] == EXPECTED_ARCHIVE_SHA
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        audit_raw = outer.read("inputs/ire-audit-evidence.zip")
    assert hashlib.sha256(audit_raw).hexdigest() == freeze["source_zip_sha256"]

    with zipfile.ZipFile(io.BytesIO(audit_raw)) as audit:
        archived = json.loads(audit.read("marco/teacher_sanity/RESULTS.json"))
        cfg = archived["config"]
        source = audit.read("marco/teacher_sanity.py").decode("utf-8")
        source_no_teacher = audit.read("marco/run_pilot.py").decode("utf-8")
        hnsw = json.loads(audit.read("marco/teacher_sanity/train_hnsw.json"))
        ivf = json.loads(audit.read("marco/teacher_sanity/train_ivf.json"))

    assert cfg["student_checkpoint"] == freeze["student"]
    assert archived["teacher_revision"] == freeze["teacher_revision"]
    assert (cfg["temperature"], cfg["teacher_max_length"], cfg["precision"]) == (1.0, 256, "float32")
    assert (cfg["query_length"], cfg["passage_length"]) == (32, 128)
    assert (cfg["updates"], cfg["batch_queries"], cfg["trained_queries"]) == (100, 8, 800)
    assert (cfg["optimizer"], cfg["lr"], cfg["weight_decay"], cfg["clip_norm"]) == ("AdamW", 2e-5, 0.01, 1.0)
    assert cfg["betas"] == [0.9, 0.999] and cfg["eps"] == 1e-8

    # Protect exact archived implementation semantics. A source change must be
    # deliberately reviewed rather than silently inheriting the old approval.
    for fragment in (
        "default_activation_function=torch.nn.Identity()",
        "activation_fct=torch.nn.Identity()",
        "teacher.predict([(tq[train[qi]],texts[row]) for qi,row in batch]",
        "def soft_loss(student, teacher):",
        "return -(teacher * F.log_softmax(student, dim=-1)).sum()",
        "model.train(); opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)",
        "(loss/8).backward()",
        "torch.nn.utils.clip_grad_norm_(model.parameters(),1.)",
        "ranking=core.exact_topk(qe,de,1000)",
        "sum(rank<=100 for rank in hits)/len(relevant)",
    ):
        assert fragment in source, f"Historical teacher training contract changed: {fragment}"
    assert "F.cross_entropy(logits,torch.zeros(len(b)" in source_no_teacher

    clipped = {}
    for name, run in (("HNSW", hnsw), ("IVF", ivf)):
        norms = run["gradient_norms_before_clip"]
        assert len(norms) == 100 and all(isinstance(v, (int, float)) for v in norms)
        clipped[name] = sum(v > cfg["clip_norm"] for v in norms)
    assert clipped == {"HNSW": 92, "IVF": 96}

    # The frozen manifest locks pair/schedule identities, but not a complete
    # optimizer/evaluation implementation. These must be explicitly approved
    # as a separate executable contract before any expensive run.
    unspecified = sorted(set(("optimizer", "lr", "weight_decay", "clip_norm", "betas", "eps",
                              "query_length", "passage_length", "schedule_seed", "seed",
                              "fixed_training_padding", "deterministic_algorithms_enforced")) - set(freeze))
    assert len(unspecified) == 12
    assert cfg["deterministic_algorithms_enforced"] is False
    print("EXACT_IMPLEMENTATION_SOURCE_PREFLIGHT PASS")
    print("ARCHIVE_AND_FREEZE_PROVENANCE PASS")
    print("HISTORICAL_GRADIENT_CLIP_EVENTS", json.dumps(clipped, sort_keys=True))
    print("FROZEN_MANIFEST_UNSPECIFIED_EXECUTION_FIELDS", ",".join(unspecified))
    print("HARD_POSITIVE_21_WAY_CONTROL NOT_IMPLEMENTED_IN_ARCHIVED_RUNNER")
    print("HARD_POSITIVE_ZERO_TEACHER_LABELS IS_A_BASELINE_NOT_A_MATCHED_16800_LABEL_COMPARATOR")
    print("DECISION REVIEWED_SOURCE_ONLY; NO_GO_NEW_TEACHER_INFERENCE_OR_TRAINING")


if __name__ == "__main__":
    main()
