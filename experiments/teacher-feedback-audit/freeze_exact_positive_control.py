#!/usr/bin/env python3
"""Freeze the original 800-query exact-teacher control without teacher/model calls.

Source: hash-locked IRE 12 October reproducibility ZIP, not the synthetic
selector test. Outputs hashes and budgets only, never query/passage text.
--write creates a new local manifest; --check compares without modification.
No training, inference, downloading, or TRANSFER_INDEX writes.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from exact_teacher_cache_gap import ARCHIVE_SHA256, npy_matrix


TASK = "ire-mid-checkpoint-20261022-001"
ARCHIVE = (Path.home() / ".codex/scratch/cross-surface/ire-project" /
           TASK / "shared/incoming/IRE_Audit_Reproducibility_2026-10-12.zip")
OUTPUT = Path(__file__).with_name("exact_positive_control_freeze.json")
STUDENT_HASH = "e6f42f74ca80eb282b5b7fa68df335eb4f57586227c601d0c60da85378e954bd"
TEACHER_REV = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def digest_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, separators=(",", ":")).encode("utf-8") + b"\n")
    return digest.hexdigest()


def frozen_manifest(archive):
    blob = archive.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == ARCHIVE_SHA256, "Wrong source archive"
    with zipfile.ZipFile(io.BytesIO(blob)) as outer:
        audit_bytes = outer.read("inputs/ire-audit-evidence.zip")
        ident_bytes = outer.read("inputs/identity-and-search-evidence.zip")
        with zipfile.ZipFile(io.BytesIO(audit_bytes)) as audit, \
             zipfile.ZipFile(io.BytesIO(ident_bytes)) as ident:
            identity = json.loads(ident.read("marco/identity.json"))
            split = json.loads(audit.read("marco/runtime/split.json"))
            result = json.loads(audit.read("marco/teacher_sanity/RESULTS.json"))
            cfg = result["config"]
            assert result["status"] == "completed"
            assert cfg["train_pool_queries"] == 1000
            assert (cfg["trained_queries"], cfg["updates"], cfg["batch_queries"]) == (800, 100, 8)
            assert (cfg["temperature"], cfg["teacher_max_length"]) == (1.0, 256)
            assert cfg["precision"] == "float32" and cfg["fixed_training_padding"]
            assert result["teacher_revision"] == TEACHER_REV

            train = list(map(int, split["train_qids"]))
            cal = list(map(int, split["cal_qids"]))
            ev = list(map(int, split["eval_qids"]))
            assert (len(train), len(cal), len(ev)) == (1000, 200, 500)
            assert len(set(train + cal + ev)) == 1700, "Non-disjoint ID partitions"
            qrels = {int(q): set(map(int, ps)) for q, ps in identity["qrels"].items()}
            assert all(q in qrels and qrels[q] for q in train + cal + ev)
            pids = list(map(int, identity["pids"]))
            assert len(pids) == len(set(pids)) == 50000
            valid_pid = set(pids)
            assert len(set().union(*(qrels[q] & valid_pid for q in ev))) == 535

            run_h = json.loads(audit.read("marco/teacher_sanity/train_hnsw.json"))
            run_i = json.loads(audit.read("marco/teacher_sanity/train_ivf.json"))
            assert run_h["initial_state_sha256"] == run_i["initial_state_sha256"] == STUDENT_HASH
            consumed = list(map(int, run_h["training_query_ids"]))
            assert consumed == list(map(int, run_i["training_query_ids"]))
            assert len(consumed) == len(set(consumed)) == 800 and set(consumed) <= set(train)
            original_order = list(range(1000))
            random.Random(cfg["schedule_seed"]).shuffle(original_order)
            assert consumed == [train[i] for i in original_order[:800]], "Schedule was not restored"

            negatives = npy_matrix(audit.read("marco/runtime/neg_flat.npy"))
            score_rows = list(csv.DictReader(io.StringIO(
                audit.read("marco/teacher_sanity/teacher_scores.csv").decode("utf-8")
            )))
            cached = set()
            for row in score_rows:
                pair = (int(row["query_id"]), int(row["passage_id"]))
                assert pair not in cached, "Duplicated scored pair"
                cached.add(pair)
            assert len(cached) == 32312
            cached_by_query = defaultdict(set)
            for q, p in cached:
                cached_by_query[q].add(p)

            by_train_q = dict(zip(train, negatives))
            slates = []
            missing = []
            complete_queries = 0
            for q in consumed:
                possible = sorted(qrels[q] & valid_pid)
                assert possible, "No positive in corpus"
                positive = possible[0]  # original run_pilot.py pos_rows rule
                assert (q, positive) in cached, "Original teacher positive unverified"
                docs = [pids[i] for i in by_train_q[q]]
                assert len(docs) == len(set(docs)) == 20 and not qrels[q].intersection(docs)
                slate = [(q, p) for p in [positive, *docs]]
                assert len(slate) == len(set(slate)) == 21
                slates.extend(slate)
                gaps = [pair for pair in slate if pair not in cached]
                missing.extend(gaps)
                complete_queries += not gaps

            assert len(slates) == len(set(slates)) == 800 * 21
            assert len(missing) == 6518 and complete_queries == 121
            assert len(slates) - len(missing) == 10282
            assert not any(p in qrels[q] for q, p in missing)
            assert result["baseline_eval"]["MRR@10"] == 0.8907769841269841
            assert result["baseline_eval"]["Recall@100"] == 0.993

            return {
                "task_id": TASK,
                "status": "FROZEN_PROTOCOL_NOT_EXECUTED",
                "archive_sha256": ARCHIVE_SHA256,
                "source_zip_sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "student_initial_state_sha256": STUDENT_HASH,
                "teacher_revision": TEACHER_REV,
                "teacher_temperature": 1.0,
                "student": cfg["student_checkpoint"],
                "train_schedule_800_qids_sha256": digest_rows([[q] for q in consumed]),
                "train_exact_ordered_16800_pairs_sha256": digest_rows(slates),
                "missing_6518_pairs_sha256": digest_rows(sorted(missing)),
                "evaluation_500_qids_qrels_sha256": digest_rows(
                    [[q, sorted(qrels[q])] for q in ev]
                ),
                "evaluation_corpus": "positive-enriched diagnostic 50k MS MARCO, NOT official full-corpus",
                "training_queries": 800,
                "evaluation_queries": 500,
                "per_query_positive_plus_negatives": [1, 20],
                "per_arm_teacher_labels_charged": 16800,
                "cached_exact_teacher_pairs": 10282,
                "new_teacher_scores_needed": 6518,
                "fully_cached_training_queries": complete_queries,
                "updates": cfg["updates"],
                "batch_queries": cfg["batch_queries"],
                "zero_teacher_baseline": "untouched ANCE MRR@10=0.8907769841269841",
                "matched_slate_control": "hard-positive CE on the SAME exact 21 document IDs and query schedule; zero teacher calls",
                "teacher_objective": "listwise teacher-soft CE on the SAME exact 21 IDs; teacher softmax T=1",
                "evaluation": "post-training full-50k re-encoding + exhaustive IP top1000; MRR@10 primary, R@100 secondary, all available qrels",
                "execution_permission": "PROTOCOL_ONLY; requires explicit authorization for 6518 teacher scores or training",
            }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    report = frozen_manifest(args.archive)
    content = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        with OUTPUT.open("x", encoding="utf-8") as f:
            f.write(content)
        print("FROZEN_EXACT_CONTROL_WRITTEN", OUTPUT)
    elif args.check:
        assert OUTPUT.read_text(encoding="utf-8") == content, "Frozen manifest drift"
        print("FROZEN_EXACT_CONTROL_CHECK PASS")
    else:
        print(content, end="")
    print("VALIDATED: 800 queries, 16800 scored slots, 10282 cached, 6518 missing, 500 held-out eval", flush=True)


if __name__ == "__main__":
    main()
