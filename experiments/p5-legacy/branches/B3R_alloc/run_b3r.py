from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

BASE = Path(__file__).resolve().parent
PILOT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot")
DATA = PILOT / "data" / "hotpot_validation.parquet"
SPLIT_MANIFEST = BASE / "split_manifest.json"
STUDENT_SNAPSHOT = PILOT / ".hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TEACHER_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
TEACHER_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
TEACHER_CACHE = PILOT / ".hf-cache/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots" / TEACHER_REVISION
TEACHER_MODEL_SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
B3_ALLOCATION_SEED = 20261031
TRAIN_SAMPLE_SEED = 0
TRAIN_N = 1000
DEV_N = 200
HOLDOUT_N = 400
TRAIN_SEEDS = (0, 1, 2)
BUDGETS = (2, 5)
ALLOCATIONS = ("random", "topk", "prd_rrdiff", "set_aware")
BOOTSTRAPS = 1000
MAX_LENGTH = 256
THREADS = 4
CHECKPOINT_EVERY = 2000
TEACHER_BATCH = 24

for key in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = str(THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(BASE / ".hf-home")
os.environ["HF_HUB_CACHE"] = str(BASE / ".hf-home" / "hub")
os.environ["TORCH_HOME"] = str(BASE / ".torch-home")
os.environ["XDG_CACHE_HOME"] = str(BASE / ".cache")

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(THREADS)
DEVICE = torch.device("cpu")
STARTED = time.monotonic()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[{time.monotonic() - STARTED:8.1f}s] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_no_clobber(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace different existing artifact: {path}")
        return
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise
    finally:
        if tmp.exists():
            tmp.unlink()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_progress(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return result


def normalize_title(title: str) -> str:
    return " ".join(str(title).split()).casefold()


def projected_nested_column(table, parent: str, child: str) -> list:
    if parent in table.column_names:
        column = table[parent].combine_chunks()
        return pc.struct_field(column, child).to_pylist()
    dotted = f"{parent}.{child}"
    if dotted in table.column_names:
        return table[dotted].to_pylist()
    raise ValueError(f"Parquet projection omitted required metadata field {dotted}")


def load_split_metadata() -> tuple[list[str], list[str], list[str], dict]:
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(DATA) != manifest["source_sha256"] or DATA.stat().st_size != manifest["source_bytes"]:
        raise ValueError("HotpotQA source differs from the B3 split manifest")

    # Only eligibility metadata is projected here. Question text and sentence text
    # are loaded later for the selected training sample and DEV IDs only.
    table = pq.read_table(
        DATA,
        columns=["id", "type", "context.title", "supporting_facts.title"],
    )
    ids = [str(value) for value in table["id"].to_pylist()]
    kinds = table["type"].to_pylist()
    if table.column_names == ["id", "type", "title", "title"]:
        candidate_titles = table.column(2).to_pylist()
        gold_titles = table.column(3).to_pylist()
    else:
        candidate_titles = projected_nested_column(table, "context", "title")
        gold_titles = projected_nested_column(table, "supporting_facts", "title")

    reasons: Counter[str] = Counter()
    eligible_indices: list[int] = []
    for index, (kind, candidates, row_gold) in enumerate(zip(kinds, candidate_titles, gold_titles)):
        reason = None
        if kind not in {"bridge", "comparison"}:
            reason = "unsupported_type"
        elif candidates is None or len(candidates) != 10:
            reason = "candidate_count_not_10"
        else:
            gold = {normalize_title(value) for value in (row_gold or [])}
            if len(gold) != 2:
                reason = "unique_gold_titles_not_2"
            elif sum(normalize_title(value) in gold for value in candidates) != 2:
                reason = "gold_candidates_not_exactly_2"
        if reason is None:
            eligible_indices.append(index)
        else:
            reasons[reason] += 1

    if len(ids) != manifest["source_rows"] or len(eligible_indices) != manifest["eligible_rows"]:
        raise ValueError(
            f"Eligibility audit differs from B3: rows={len(ids)}, eligible={len(eligible_indices)}"
        )
    eligible_ids = [ids[index] for index in eligible_indices]
    sampled_indices = np.random.default_rng(manifest["sample_seed"]).choice(
        len(eligible_ids), manifest["sample_n"], replace=False
    )
    sampled = [eligible_ids[int(index)] for index in sampled_indices]
    sampled.sort(key=lambda qid: (hashlib.sha256(qid.encode("utf-8")).hexdigest(), qid))
    dev_ids = sampled[:DEV_N]
    holdout_ids = sampled[DEV_N:]
    sampled_set = set(sampled)
    train_pool_ids = [qid for qid in eligible_ids if qid not in sampled_set]

    checks = {
        "dev_ids_sha256_sorted": sha256_ids(dev_ids),
        "sampled_600_ids_sha256_sorted": sha256_ids(sampled),
        "training_pool_ids_sha256_sorted": sha256_ids(train_pool_ids),
    }
    for name, actual in checks.items():
        if actual != manifest[name]:
            raise ValueError(f"Reconstructed B3 split hash mismatch: {name}")
    if len(dev_ids) != DEV_N or len(holdout_ids) != HOLDOUT_N or len(train_pool_ids) != 6742:
        raise ValueError("B3 DEV, HOLDOUT, or training-pool size mismatch")
    if dict(sorted(reasons.items())) != manifest["excluded_rows_by_reason"]:
        raise ValueError("Eligibility exclusion reasons differ from B3")
    log(f"Verified B3 split: train pool={len(train_pool_ids)}, DEV={len(dev_ids)}, HOLDOUT IDs={len(holdout_ids)}")
    return dev_ids, holdout_ids, train_pool_ids, manifest


def select_training_ids(train_pool_ids: list[str]) -> list[str]:
    indices = np.random.default_rng(TRAIN_SAMPLE_SEED).choice(
        len(train_pool_ids), TRAIN_N, replace=False
    )
    return [train_pool_ids[int(index)] for index in indices]


def load_rows(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    dataset = ds.dataset(DATA, format="parquet")
    table = dataset.to_table(
        columns=["id", "question", "type", "context", "supporting_facts"],
        filter=ds.field("id").isin(ids),
    )
    rows = table.to_pylist()
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(ids) or set(by_id) != set(ids):
        raise ValueError("Materialized rows do not exactly match requested non-HOLDOUT IDs")
    return [by_id[qid] for qid in ids]


def prepare_rows(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        context = row["context"]
        titles, sentence_groups = context["title"], context["sentences"]
        gold_titles = {normalize_title(value) for value in row["supporting_facts"]["title"]}
        if len(titles) != 10 or len(sentence_groups) != 10 or len(gold_titles) != 2:
            raise ValueError(f"Malformed eligible row {row['id']}")
        paragraphs = [
            (str(title).strip() + " " + " ".join(str(sentence).strip() for sentence in group)).strip()
            for title, group in zip(titles, sentence_groups)
        ]
        gold = [i for i, title in enumerate(titles) if normalize_title(title) in gold_titles]
        if len(gold) != 2:
            raise ValueError(f"Expected exactly two gold candidate paragraphs in {row['id']}")
        prepared.append({
            "qid": str(row["id"]),
            "question": str(row["question"]),
            "type": str(row["type"]),
            "paragraphs": paragraphs,
            "gold": gold,
        })
    return prepared


def snapshot_files(path: Path) -> list[dict]:
    result = []
    for file in sorted(path.rglob("*")):
        if file.is_file():
            result.append({
                "path": str(file.relative_to(path)),
                "bytes": file.stat().st_size,
                "sha256": sha256_file(file),
            })
    return result


def load_models():
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    if not TEACHER_CACHE.is_dir() or not STUDENT_SNAPSHOT.is_dir():
        raise FileNotFoundError("Pinned local teacher cache or student snapshot is missing")
    teacher_files = snapshot_files(TEACHER_CACHE)
    weight = next((item for item in teacher_files if item["path"] == "model.safetensors"), None)
    if weight is None or weight["sha256"] != TEACHER_MODEL_SHA256:
        raise ValueError("Pinned teacher weights fail the B3 SHA256 check")
    student_files = snapshot_files(STUDENT_SNAPSHOT)
    student_weight = next((item for item in student_files if item["path"] == "model.safetensors"), None)
    if student_weight is None or student_weight["bytes"] > 150_000_000:
        raise ValueError("Cached MiniLM student weights are missing or exceed 150 MB")

    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_CACHE, local_files_only=True, use_fast=True)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_CACHE, local_files_only=True, use_safetensors=True
    ).to(DEVICE).eval()
    student_tok = AutoTokenizer.from_pretrained(STUDENT_SNAPSHOT, local_files_only=True, use_fast=True)
    student = AutoModel.from_pretrained(
        STUDENT_SNAPSHOT, local_files_only=True, use_safetensors=True
    ).to(DEVICE).eval()
    return teacher_tok, teacher, teacher_files, student_tok, student, student_files


def tokenizer_batch(tokenizer, texts, max_length: int) -> tuple[dict, int]:
    encoded = tokenizer(texts, padding=False, truncation=False)
    features = []
    truncated_count = 0
    for index, ids in enumerate(encoded["input_ids"]):
        token_types = encoded.get("token_type_ids")
        if len(ids) > max_length:
            truncated_count += 1
            ids = ids[:max_length]
            if tokenizer.sep_token_id is not None:
                ids[-1] = tokenizer.sep_token_id
        feature = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if token_types is not None:
            feature["token_type_ids"] = token_types[index][:len(ids)]
        features.append(feature)
    return tokenizer.pad(features, padding=True, return_tensors="pt"), truncated_count


@torch.inference_mode()
def encode_texts(texts: list[str], tokenizer, model, role: str) -> tuple[np.ndarray, int]:
    vectors = []
    truncated_total = 0
    for start in range(0, len(texts), 32):
        batch, truncated = tokenizer_batch(tokenizer, texts[start:start + 32], MAX_LENGTH)
        truncated_total += truncated
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors.append(F.normalize(pooled, p=2, dim=1).cpu().numpy().astype(np.float32))
        if start and start % 3200 == 0:
            log(f"Encoded {role}: {min(start + 32, len(texts))}/{len(texts)}")
    return np.concatenate(vectors, axis=0), truncated_total


@torch.inference_mode()
def teacher_batch(items: list[tuple[str, str]], tokenizer, model) -> tuple[np.ndarray, np.ndarray, int]:
    batch, truncated = tokenizer_batch(tokenizer, items, MAX_LENGTH)
    lengths = batch["attention_mask"].sum(dim=1).cpu().numpy().astype(np.int32)
    batch = {key: value.to(DEVICE) for key, value in batch.items()}
    scores = model(**batch).logits.reshape(-1).float().cpu().numpy().astype(np.float32)
    return scores, lengths, truncated


def teacher_token_counts(items: list[tuple[str, str]], tokenizer) -> tuple[np.ndarray, int]:
    counts = []
    truncated_total = 0
    for start in range(0, len(items), TEACHER_BATCH):
        batch, truncated = tokenizer_batch(tokenizer, items[start:start + TEACHER_BATCH], MAX_LENGTH)
        counts.extend(batch["attention_mask"].sum(dim=1).cpu().numpy().astype(np.int32).tolist())
        truncated_total += truncated
    return np.asarray(counts, dtype=np.int32), truncated_total


def teacher_scores(items: list[tuple[str, str]], tokenizer, model) -> tuple[np.ndarray, np.ndarray]:
    scores, counts = [], []
    for start in range(0, len(items), TEACHER_BATCH):
        values, token_counts, _ = teacher_batch(items[start:start + TEACHER_BATCH], tokenizer, model)
        scores.extend(values.tolist())
        counts.extend(token_counts.tolist())
    return np.asarray(scores, dtype=np.float32), np.asarray(counts, dtype=np.int32)


def dense_scores(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    return np.einsum("qd,qpd->qp", queries, passages, optimize=True)


def full_support_at(scores: np.ndarray, examples: list[dict], k: int) -> np.ndarray:
    result = np.zeros(len(examples), dtype=np.float32)
    for index, example in enumerate(examples):
        order = np.argsort(-scores[index], kind="stable")[:k]
        result[index] = float(set(example["gold"]).issubset(set(order.tolist())))
    return result


def bootstrap_mean(values: np.ndarray, bootstrap_indices: np.ndarray) -> tuple[float, float, float]:
    point = float(np.mean(values))
    boot = values[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def pair_orders(
    examples: list[dict],
    query_vectors: np.ndarray,
    passage_vectors: np.ndarray,
) -> dict[str, list[list[tuple[int, int]]]]:
    all_pairs = list(itertools.combinations(range(10), 2))
    result = {name: [] for name in ALLOCATIONS}
    for q, example in enumerate(examples):
        qid = example["qid"]
        relevance = passage_vectors[q] @ query_vectors[q]
        rank = np.empty(10, dtype=np.int64)
        rank[np.argsort(-relevance, kind="stable")] = np.arange(1, 11)
        seed_bytes = hashlib.sha256(f"{B3_ALLOCATION_SEED}\0{qid}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
        random_order = [all_pairs[i] for i in rng.permutation(len(all_pairs))[:max(BUDGETS)]]
        top_order = sorted(
            all_pairs,
            key=lambda pair: (
                min(rank[pair[0]], rank[pair[1]]),
                max(rank[pair[0]], rank[pair[1]]),
                pair,
            ),
        )[:max(BUDGETS)]
        weights = np.asarray(
            [abs(1.0 / rank[i] - 1.0 / rank[j]) for i, j in all_pairs],
            dtype=np.float64,
        )
        prd_indices = rng.choice(
            len(all_pairs), size=max(BUDGETS), replace=False, p=weights / weights.sum()
        )
        prd_order = [all_pairs[int(index)] for index in prd_indices]
        passage_similarity = passage_vectors[q] @ passage_vectors[q].T
        remaining = set(range(len(all_pairs)))
        counts = np.zeros(10, dtype=np.int32)
        set_order = []
        for _ in range(max(BUDGETS)):
            best_index, best_score = None, -math.inf
            for pair_index in sorted(remaining):
                i, j = all_pairs[pair_index]
                joint = 0.5 * (relevance[i] + relevance[j]) + 0.20 * (
                    1.0 - float(passage_similarity[i, j])
                )
                score = joint / math.sqrt((1 + counts[i]) * (1 + counts[j]))
                if score > best_score:
                    best_index, best_score = pair_index, score
            assert best_index is not None
            i, j = all_pairs[best_index]
            set_order.append((i, j))
            counts[i] += 1
            counts[j] += 1
            remaining.remove(best_index)
        result["random"].append(random_order)
        result["topk"].append(top_order)
        result["prd_rrdiff"].append(prd_order)
        result["set_aware"].append(set_order)
    return result


class QueryAdapter(nn.Module):
    def __init__(self, dimension: int = 384, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return F.normalize(query + self.up(F.gelu(self.down(query))), p=2, dim=-1)


def fit_adapter(
    query_vectors: np.ndarray,
    passage_vectors: np.ndarray,
    pair_rows: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int,
) -> QueryAdapter:
    torch.manual_seed(seed)
    adapter = QueryAdapter(query_vectors.shape[1]).to(DEVICE)
    queries, passages = torch.from_numpy(query_vectors), torch.from_numpy(passage_vectors)
    qidx, left, right, signs = (torch.from_numpy(array) for array in pair_rows)
    with torch.no_grad():
        if not torch.allclose(adapter(queries), queries, atol=1e-6, rtol=1e-6):
            raise AssertionError("Zero-initialized query adapter must reproduce the base encoder")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(3):
        permutation = np.random.default_rng(seed + epoch).permutation(len(qidx))
        for start in range(0, len(permutation), 512):
            indices = torch.from_numpy(permutation[start:start + 512])
            qi, li, ri, labels = qidx[indices], left[indices], right[indices], signs[indices]
            output = adapter(queries[qi])
            left_score = (output * passages[qi, li]).sum(dim=1)
            right_score = (output * passages[qi, ri]).sum(dim=1)
            loss = F.softplus(-labels * (left_score - right_score)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return adapter.eval()


def adapter_scores(
    adapter: QueryAdapter, query_vectors: np.ndarray, passage_vectors: np.ndarray
) -> np.ndarray:
    with torch.inference_mode():
        adapted = adapter(torch.from_numpy(query_vectors)).cpu().numpy()
    return dense_scores(adapted, passage_vectors)


def collect_pair_rows(
    orders: list[list[tuple[int, int]]], budget: int, teacher_matrix: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]:
    qidx, left, right, signs = [], [], [], []
    skipped_ties = 0
    for q, order in enumerate(orders):
        for i, j in order[:budget]:
            difference = float(teacher_matrix[q, i] - teacher_matrix[q, j])
            if not math.isfinite(difference):
                raise ValueError("Missing or non-finite teacher score on a selected pair")
            if difference == 0.0:
                skipped_ties += 1
                continue
            qidx.append(q)
            left.append(i)
            right.append(j)
            signs.append(1.0 if difference > 0 else -1.0)
    if not qidx:
        raise ValueError("No non-tied teacher comparisons to train on")
    return (
        np.asarray(qidx, dtype=np.int64),
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(signs, dtype=np.float32),
    ), skipped_ties


def selection_plan(
    train: list[dict], orders: dict[str, list[list[tuple[int, int]]]]
) -> tuple[list[tuple[int, int]], dict[tuple[str, int], set[tuple[int, int]]], str]:
    arm_keys: dict[tuple[str, int], set[tuple[int, int]]] = {}
    union: set[tuple[int, int]] = set()
    order_payload = {}
    for method in ALLOCATIONS:
        order_payload[method] = []
        for q, order in enumerate(orders[method]):
            order_payload[method].append([list(pair) for pair in order[:max(BUDGETS)]])
        for budget in BUDGETS:
            keys = {
                (q, endpoint)
                for q, order in enumerate(orders[method])
                for pair in order[:budget]
                for endpoint in pair
            }
            arm_keys[(method, budget)] = keys
            union.update(keys)
    global_keys = sorted(union)
    payload = {
        "train_qids": [example["qid"] for example in train],
        "orders": order_payload,
        "budgets": BUDGETS,
        "global_keys": global_keys,
    }
    plan_hash = sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return global_keys, arm_keys, plan_hash


def checkpoint_bytes(
    keys: list[tuple[int, int]],
    scores: np.ndarray,
    token_counts: np.ndarray,
    completed: int,
    plan_hash: str,
) -> bytes:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        keys=np.asarray(keys, dtype=np.int64).reshape(-1, 2),
        scores=np.asarray(scores, dtype=np.float32),
        token_counts=np.asarray(token_counts, dtype=np.int32),
        completed_count=np.asarray(completed, dtype=np.int64),
        plan_sha256=np.asarray(plan_hash),
    )
    return buffer.getvalue()


def write_score_checkpoint(
    directory: Path,
    keys: list[tuple[int, int]],
    scores: np.ndarray,
    token_counts: np.ndarray,
    plan_hash: str,
) -> dict:
    completed = len(keys)
    data = checkpoint_bytes(keys, scores, token_counts, completed, plan_hash)
    digest = sha256_bytes(data)
    filename = f"scores_{completed:08d}_{digest[:12]}.npz"
    target = directory / filename
    write_no_clobber(target, data)
    sidecar = Path(str(target) + ".sha256")
    write_no_clobber(sidecar, f"{digest}  {filename}\n".encode("utf-8"))
    return {"completed": completed, "file": filename, "sha256": digest}


def load_latest_checkpoint(
    directory: Path,
    global_keys: list[tuple[int, int]],
    expected_tokens: np.ndarray,
    plan_hash: str,
) -> tuple[int, np.ndarray, list[dict]]:
    pattern = re.compile(r"^scores_(\d{8})_([0-9a-f]{12})\.npz$")
    valid = []
    for path in directory.glob("scores_*.npz"):
        match = pattern.match(path.name)
        if not match:
            continue
        sidecar = Path(str(path) + ".sha256")
        if not sidecar.is_file():
            continue
        digest = sha256_file(path)
        if sidecar.read_text(encoding="utf-8").strip() != f"{digest}  {path.name}":
            continue
        try:
            with np.load(path, allow_pickle=False) as saved:
                completed = int(saved["completed_count"].item())
                stored_plan = str(saved["plan_sha256"].item())
                keys = np.asarray(saved["keys"], dtype=np.int64).reshape(-1, 2)
                scores = np.asarray(saved["scores"], dtype=np.float32)
                token_counts = np.asarray(saved["token_counts"], dtype=np.int32)
            if completed != int(match.group(1)) or stored_plan != plan_hash:
                continue
            expected_keys = np.asarray(global_keys[:completed], dtype=np.int64).reshape(-1, 2)
            if not np.array_equal(keys, expected_keys):
                continue
            if len(scores) != completed or not np.isfinite(scores).all():
                continue
            if not np.array_equal(token_counts, expected_tokens[:completed]):
                continue
            valid.append({
                "completed": completed,
                "scores": scores,
                "path": path,
                "sha256": digest,
            })
        except (OSError, ValueError, KeyError):
            continue
    if not valid:
        return 0, np.asarray([], dtype=np.float32), []
    valid.sort(key=lambda item: (item["completed"], item["path"].name))
    latest_count = valid[-1]["completed"]
    same_count = [item for item in valid if item["completed"] == latest_count]
    for item in same_count[1:]:
        if not np.array_equal(item["scores"], same_count[0]["scores"]):
            raise ValueError("Conflicting valid score checkpoints have the same plan and count")
    chosen = same_count[0]
    summaries = [
        {"completed": item["completed"], "file": item["path"].name, "sha256": item["sha256"]}
        for item in valid
    ]
    return chosen["completed"], chosen["scores"], summaries


def score_selected_with_checkpoints(
    train: list[dict],
    global_keys: list[tuple[int, int]],
    expected_tokens: np.ndarray,
    tokenizer,
    model,
    plan_hash: str,
    progress_path: Path,
) -> tuple[np.ndarray, list[dict]]:
    done, saved_scores, checkpoints = load_latest_checkpoint(
        BASE, global_keys, expected_tokens, plan_hash
    )
    all_scores = saved_scores.tolist()
    records = read_progress(progress_path)
    recorded = {record.get("checkpoint_file") for record in records}
    for checkpoint in checkpoints:
        if checkpoint["file"] not in recorded:
            append_jsonl(progress_path, {
                "event": "teacher_checkpoint",
                "time_utc": utc_now(),
                "teacher_items_completed": checkpoint["completed"],
                "checkpoint_file": checkpoint["file"],
                "sha256": checkpoint["sha256"],
                "reconciled_from_verified_file": True,
            })
    verified = list(checkpoints)
    while done < len(global_keys):
        boundary = min(((done // CHECKPOINT_EVERY) + 1) * CHECKPOINT_EVERY, len(global_keys))
        end = min(done + TEACHER_BATCH, boundary)
        items = [
            (train[q]["question"], train[q]["paragraphs"][candidate])
            for q, candidate in global_keys[done:end]
        ]
        values, counts, truncated = teacher_batch(items, tokenizer, model)
        if not np.array_equal(counts, expected_tokens[done:end]):
            raise ValueError("Scored teacher token lengths differ from the cost plan")
        all_scores.extend(values.tolist())
        done = end
        if done % CHECKPOINT_EVERY == 0 or done == len(global_keys):
            prefix_keys = global_keys[:done]
            info = write_score_checkpoint(
                BASE,
                prefix_keys,
                np.asarray(all_scores, dtype=np.float32),
                expected_tokens[:done],
                plan_hash,
            )
            append_jsonl(progress_path, {
                "event": "teacher_checkpoint",
                "time_utc": utc_now(),
                "teacher_items_completed": done,
                "total_teacher_items": len(global_keys),
                "checkpoint_file": info["file"],
                "sha256": info["sha256"],
                "batch_truncations": int(truncated),
                "elapsed_seconds": round(time.monotonic() - STARTED, 3),
            })
            verified.append(info)
            log(f"Teacher scores checkpointed: {done}/{len(global_keys)}")
    return np.asarray(all_scores, dtype=np.float32), verified


def make_cost_records(
    train: list[dict],
    global_keys: list[tuple[int, int]],
    arm_keys: dict[tuple[str, int], set[tuple[int, int]]],
    token_counts: np.ndarray,
) -> list[dict]:
    tokens_by_key = dict(zip(global_keys, token_counts.tolist()))
    result = []
    for method in ALLOCATIONS:
        for budget in BUDGETS:
            keys = arm_keys[(method, budget)]
            result.append({
                "allocation": method,
                "budget_pair_labels_per_question": budget,
                "selected_pair_labels": len(train) * budget,
                "unique_teacher_query_passage_calls": len(keys),
                "teacher_input_tokens": int(sum(tokens_by_key[key] for key in keys)),
                "selection_stage_teacher_calls": 0,
                "selection_stage_teacher_input_tokens": 0,
                "cost_status": "valid",
                "accounting": "arm-specific unique selected endpoints, counted as uncached calls",
            })
    return result


def train_and_evaluate(
    train: list[dict],
    dev: list[dict],
    orders: dict[str, list[list[tuple[int, int]]]],
    teacher_matrix: np.ndarray,
    train_qvec: np.ndarray,
    train_pvec: np.ndarray,
    dev_qvec: np.ndarray,
    dev_pvec: np.ndarray,
    seeds: tuple[int, ...],
) -> tuple[dict, list[dict]]:
    outcomes = {}
    records = []
    for method in ALLOCATIONS:
        for budget in BUDGETS:
            pair_rows, tie_count = collect_pair_rows(orders[method], budget, teacher_matrix)
            for seed in seeds:
                adapter = fit_adapter(train_qvec, train_pvec, pair_rows, seed)
                scores = adapter_scores(adapter, dev_qvec, dev_pvec)
                for k in (2, 4):
                    outcomes[(method, budget, seed, k)] = full_support_at(scores, dev, k)
                records.append({
                    "allocation": method,
                    "budget_pair_labels_per_question": budget,
                    "seed": seed,
                    "train_pair_labels": int(len(pair_rows[0])),
                    "teacher_ties_skipped": tie_count,
                    "epochs": 3,
                })
                del adapter, scores
                gc.collect()
                log(f"Fit complete: {method}, budget={budget}, seed={seed}")
    return outcomes, records


def summarize_outcomes(outcomes: dict, n_questions: int) -> tuple[list[dict], list[dict]]:
    bootstrap_indices = np.random.default_rng(0).integers(
        0, n_questions, size=(BOOTSTRAPS, n_questions)
    )
    summaries, contrasts = [], []
    for method in ALLOCATIONS:
        for budget in BUDGETS:
            for k in (2, 4):
                by_question = np.stack([
                    outcomes[(method, budget, seed, k)] for seed in TRAIN_SEEDS
                ]).mean(axis=0)
                point, low, high = bootstrap_mean(by_question, bootstrap_indices)
                summaries.append({
                    "allocation": method,
                    "budget_pair_labels_per_question": budget,
                    "metric": f"FULL-SUPPORT@{k}",
                    "mean_across_seeds": point,
                    "ci95_question_bootstrap": [low, high],
                    "bootstrap_resamples": BOOTSTRAPS,
                    "bootstrap_unit": "paired question; seed outcomes averaged within question",
                })
    for method in ALLOCATIONS:
        for comparator in ("random", "topk"):
            for budget in BUDGETS:
                for k in (2, 4):
                    arm = np.stack([
                        outcomes[(method, budget, seed, k)] for seed in TRAIN_SEEDS
                    ]).mean(axis=0)
                    reference = np.stack([
                        outcomes[(comparator, budget, seed, k)] for seed in TRAIN_SEEDS
                    ]).mean(axis=0)
                    point, low, high = bootstrap_mean(arm - reference, bootstrap_indices)
                    contrasts.append({
                        "allocation": method,
                        "comparator": comparator,
                        "budget_pair_labels_per_question": budget,
                        "metric": f"FULL-SUPPORT@{k}",
                        "difference": point,
                        "ci95_paired_question_bootstrap": [low, high],
                        "bootstrap_resamples": BOOTSTRAPS,
                    })
    return summaries, contrasts


def write_plan_record(progress_path: Path, plan_hash: str, costs: list[dict], physical_calls: int) -> None:
    records = read_progress(progress_path)
    plans = [record for record in records if record.get("event") == "selection_plan"]
    payload = {
        "event": "selection_plan",
        "time_utc": utc_now(),
        "plan_sha256": plan_hash,
        "physical_unique_teacher_calls": physical_calls,
        "arm_costs": costs,
    }
    for old in plans:
        if old.get("plan_sha256") != plan_hash or old.get("arm_costs") != costs:
            raise ValueError("Existing progress file contains a different selection plan")
    if not plans:
        append_jsonl(progress_path, payload)


def ensure_progress_started(progress_path: Path) -> str:
    records = read_progress(progress_path)
    starts = [record for record in records if record.get("event") == "started"]
    if not starts:
        started_at = utc_now()
        append_jsonl(progress_path, {
            "event": "started",
            "time_utc": started_at,
            "pid": os.getpid(),
            "threads": THREADS,
            "protocol": "protocol.md",
        })
        return started_at
    append_jsonl(progress_path, {
        "event": "resumed",
        "time_utc": utc_now(),
        "pid": os.getpid(),
    })
    return starts[0]["time_utc"]


def result_markdown(results: dict) -> str:
    costs = results["arm_costs"]
    lines = [
        "# B3R allocation pilot",
        "",
        "**Label:** DEV pilot, tuned-free, closed setting",
        "",
        "## Teacher-call budgets",
        "",
        "| Arm | Pair labels/q | Unique teacher calls | Teacher input tokens | Selection-stage calls | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in costs:
        lines.append(
            f"| {row['allocation']} | {row['budget_pair_labels_per_question']} | "
            f"{row['unique_teacher_query_passage_calls']} | {row['teacher_input_tokens']} | "
            f"{row['selection_stage_teacher_calls']} | {row['cost_status']} |"
        )
    lines.extend([
        "",
        "## DEV FULL-SUPPORT (mean over seeds; 95% question-bootstrap CI)",
        "",
        "| Arm | Pair labels/q | Metric | Mean [95% CI] |",
        "|---|---:|---|---:|",
    ])
    for row in results["summaries"]:
        low, high = row["ci95_question_bootstrap"]
        lines.append(
            f"| {row['allocation']} | {row['budget_pair_labels_per_question']} | "
            f"{row['metric']} | {row['mean_across_seeds']:.3f} [{low:.3f}, {high:.3f}] |"
        )
    lines.extend([
        "",
        "## Paired differences versus uniform-random and top-rank",
        "",
        "| Arm | Comparator | Pair labels/q | Metric | Difference [95% CI] |",
        "|---|---|---:|---|---:|",
    ])
    for row in results["paired_contrasts"]:
        low, high = row["ci95_paired_question_bootstrap"]
        lines.append(
            f"| {row['allocation']} | {row['comparator']} | "
            f"{row['budget_pair_labels_per_question']} | {row['metric']} | "
            f"{row['difference']:+.3f} [{low:+.3f}, {high:+.3f}] |"
        )
    lines.extend([
        "",
        "## Accounting and scope",
        "",
        f"- Physical shared-cache teacher scores: {results['teacher_accounting']['physical_unique_scores']}; "
        "arm costs above count each arm's own selected unique endpoints as uncached calls.",
        "- Selection-stage teacher calls are zero for all arms; selection uses frozen student similarities or seeded randomness.",
        "- DEV contains 200 questions. HOLDOUT question and paragraph rows were not materialized for training or evaluation.",
        "- Teacher scores were requested only for endpoints of selected training pairs. No teacher DEV headroom pass was run.",
        f"- Split manifest SHA256: {results['split']['manifest_sha256']}.",
        f"- Selection plan SHA256: {results['selection_plan_sha256']}.",
        "",
    ])
    return "\n".join(lines)


def write_final_outputs(results: dict) -> None:
    results_bytes = (json.dumps(results, indent=2, sort_keys=True) + "\n").encode("utf-8")
    report_bytes = result_markdown(results).encode("utf-8")
    write_no_clobber(BASE / "results.json", results_bytes)
    write_no_clobber(BASE / "RESULT.md", report_bytes)

    files = [
        BASE / "FAILURE_PROVENANCE.md",
        BASE / "protocol.md",
        BASE / "run_b3r.py",
        BASE / "split_manifest.json",
        BASE / "progress.jsonl",
        BASE / "results.json",
        BASE / "RESULT.md",
    ]
    files.extend(sorted(BASE.glob("scores_*.npz")))
    files.extend(sorted(BASE.glob("scores_*.npz.sha256")))
    lines = [f"{sha256_file(path)}  {path.name}" for path in files if path.is_file()]
    manifest_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    write_no_clobber(BASE / "manifest.sha256", manifest_bytes)


def build_results(
    split: dict,
    selected_train_ids: list[str],
    teacher_files: list[dict],
    student_files: list[dict],
    train_q_trunc: int,
    train_p_trunc: int,
    dev_q_trunc: int,
    dev_p_trunc: int,
    outcomes: dict,
    seed_records: list[dict],
    cost_records: list[dict],
    global_keys: list[tuple[int, int]],
    token_counts: np.ndarray,
    plan_hash: str,
    checkpoints: list[dict],
    started_at: str,
) -> dict:
    summaries, contrasts = summarize_outcomes(outcomes, DEV_N)
    return {
        "label": "DEV pilot, tuned-free, closed setting",
        "status": "COMPLETE",
        "experiment_started_utc": started_at,
        "protocol": "protocol.md",
        "split": {
            "manifest_sha256": sha256_file(SPLIT_MANIFEST),
            "source_sha256": split["source_sha256"],
            "dev_n": DEV_N,
            "dev_ids_sha256_sorted": split["dev_ids_sha256_sorted"],
            "holdout_n": HOLDOUT_N,
            "holdout_content_materialized_for_training_or_evaluation": False,
            "training_pool_n": split["training_pool_n"],
            "training_pool_ids_sha256_sorted": split["training_pool_ids_sha256_sorted"],
            "training_sample_n": len(selected_train_ids),
            "training_sample_seed": TRAIN_SAMPLE_SEED,
            "training_sample_ids_sha256_sorted": sha256_ids(selected_train_ids),
        },
        "selection_plan_sha256": plan_hash,
        "arms": list(ALLOCATIONS),
        "budgets_pair_labels_per_question": list(BUDGETS),
        "training_seeds": list(TRAIN_SEEDS),
        "student_training": {
            "student": "sentence-transformers/all-MiniLM-L6-v2",
            "query_adapter": "residual MLP 384->64->384; frozen text encoder; fixed passage vectors",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "epochs": 3,
            "batch_size": 512,
            "device": str(DEVICE),
            "torch_threads": torch.get_num_threads(),
        },
        "teacher": {
            "id": TEACHER_ID,
            "revision": TEACHER_REVISION,
            "files": teacher_files,
            "max_length": MAX_LENGTH,
        },
        "student_model_files": student_files,
        "teacher_accounting": {
            "physical_unique_scores": len(global_keys),
            "physical_unique_input_tokens": int(token_counts.sum()),
            "arm_costs_count_each_arm_own_unique_selected_endpoints": True,
            "selection_stage_teacher_calls_are_zero": True,
            "cost_validity": "valid for all four arms",
            "verified_checkpoints": checkpoints,
        },
        "arm_costs": cost_records,
        "seed_runs": seed_records,
        "summaries": summaries,
        "paired_contrasts": contrasts,
        "bootstrap": {
            "resamples": BOOTSTRAPS,
            "seed": 0,
            "unit": "DEV question; seed outcomes averaged within question",
            "confidence_level": 0.95,
        },
        "model_input_truncations": {
            "student_train_queries": train_q_trunc,
            "student_train_passages": train_p_trunc,
            "student_dev_queries": dev_q_trunc,
            "student_dev_passages": dev_p_trunc,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "huggingface_hub": importlib.metadata.version("huggingface-hub"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": str(DEVICE),
        },
    }


def run_unit_test() -> None:
    torch.set_num_interop_threads(THREADS)
    dev_ids, holdout_ids, train_pool_ids, _ = load_split_metadata()
    selected_ids = select_training_ids(train_pool_ids)
    train_ids = selected_ids[:20]
    test_dev_ids = dev_ids[:20]
    if set(train_ids) & set(holdout_ids) or set(test_dev_ids) & set(holdout_ids):
        raise AssertionError("HOLDOUT IDs entered the 20-question unit test")

    train = prepare_rows(load_rows(train_ids))
    dev = prepare_rows(load_rows(test_dev_ids))
    teacher_tok, teacher, _, student_tok, student, _ = load_models()
    train_qvec, _ = encode_texts([item["question"] for item in train], student_tok, student, "unit train queries")
    train_pflat, _ = encode_texts(
        [paragraph for item in train for paragraph in item["paragraphs"]],
        student_tok, student, "unit train passages",
    )
    train_pvec = train_pflat.reshape(len(train), 10, -1)
    dev_qvec, _ = encode_texts([item["question"] for item in dev], student_tok, student, "unit DEV queries")
    dev_pflat, _ = encode_texts(
        [paragraph for item in dev for paragraph in item["paragraphs"]],
        student_tok, student, "unit DEV passages",
    )
    dev_pvec = dev_pflat.reshape(len(dev), 10, -1)
    orders = pair_orders(train, train_qvec, train_pvec)
    global_keys, arm_keys, plan_hash = selection_plan(train, orders)
    items = [
        (train[q]["question"], train[q]["paragraphs"][candidate])
        for q, candidate in global_keys
    ]
    token_counts, _ = teacher_token_counts(items, teacher_tok)
    values, measured_tokens = teacher_scores(items, teacher_tok, teacher)
    if not np.array_equal(token_counts, measured_tokens):
        raise AssertionError("Unit-test teacher token accounting differs from scoring")
    teacher_matrix = np.full((len(train), 10), np.nan, dtype=np.float32)
    for (q, candidate), score in zip(global_keys, values):
        teacher_matrix[q, candidate] = score
    costs = make_cost_records(train, global_keys, arm_keys, token_counts)
    if any(row["selection_stage_teacher_calls"] != 0 for row in costs):
        raise AssertionError("Cheap-signal selection unexpectedly requested teacher scores")
    if set(global_keys) != set.union(*arm_keys.values()) or len(global_keys) > 20 * 10:
        raise AssertionError("Teacher scoring included unselected endpoints")

    with tempfile.TemporaryDirectory(prefix="b3r-unit-", dir=str(BASE)) as temp:
        small_keys = global_keys[:min(3, len(global_keys))]
        info = write_score_checkpoint(
            Path(temp), small_keys, values[:len(small_keys)],
            token_counts[:len(small_keys)], plan_hash,
        )
        count, restored, loaded = load_latest_checkpoint(
            Path(temp), global_keys, token_counts, plan_hash,
        )
        if count != len(small_keys) or not np.array_equal(restored, values[:count]) or not loaded:
            raise AssertionError("Score-checkpoint write/verify/resume self-check failed")
        if info["sha256"] != loaded[-1]["sha256"]:
            raise AssertionError("Checkpoint sidecar hash self-check failed")

    outcomes, records = train_and_evaluate(
        train, dev, orders, teacher_matrix, train_qvec, train_pvec, dev_qvec, dev_pvec, TRAIN_SEEDS
    )
    if len(records) != len(ALLOCATIONS) * len(BUDGETS) * len(TRAIN_SEEDS):
        raise AssertionError("Unit-test arm/budget/seed fit count is wrong")
    if any(outcomes[(method, budget, seed, 2)].shape != (20,)
           for method in ALLOCATIONS for budget in BUDGETS for seed in TRAIN_SEEDS):
        raise AssertionError("Unit-test DEV metric shape is wrong")
    print(
        f"UNIT_TEST PASS: train_questions=20 dev_questions=20 arms=4 budgets=2/5 "
        f"adapter_fits={len(records)} selected_teacher_calls={len(global_keys)} "
        f"checkpoint_round_trip=PASS HOLDOUT_rows_materialized=0 plan={plan_hash[:12]}",
        flush=True,
    )


def run_real() -> None:
    torch.set_num_interop_threads(THREADS)
    if not (BASE / "protocol.md").is_file():
        raise FileNotFoundError("protocol.md must exist before the real run")
    if not DATA.is_file() or not STUDENT_SNAPSHOT.is_dir() or not TEACHER_CACHE.is_dir():
        raise FileNotFoundError("Required local data or pinned model cache is missing")

    print(
        f"B3R PID={os.getpid()} START_UTC={utc_now()} THREADS={THREADS} "
        f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}",
        flush=True,
    )
    dev_ids, holdout_ids, train_pool_ids, split = load_split_metadata()
    selected_train_ids = select_training_ids(train_pool_ids)
    if len(selected_train_ids) != TRAIN_N or set(selected_train_ids) & set(holdout_ids):
        raise AssertionError("Training sample size or HOLDOUT exclusion check failed")
    if set(dev_ids) & set(holdout_ids):
        raise AssertionError("DEV and HOLDOUT IDs overlap")

    train = prepare_rows(load_rows(selected_train_ids))
    dev = prepare_rows(load_rows(dev_ids))
    if len(train) != TRAIN_N or len(dev) != DEV_N:
        raise AssertionError("Training or DEV materialization size mismatch")
    if ({item["qid"] for item in train} & set(holdout_ids)
            or {item["qid"] for item in dev} & set(holdout_ids)):
        raise AssertionError("HOLDOUT row entered model data")

    log("Loading pinned local cross-encoder teacher and cached MiniLM student")
    teacher_tok, teacher, teacher_files, student_tok, student, student_files = load_models()
    train_qvec, train_q_trunc = encode_texts(
        [item["question"] for item in train], student_tok, student, "training queries"
    )
    train_pflat, train_p_trunc = encode_texts(
        [paragraph for item in train for paragraph in item["paragraphs"]],
        student_tok, student, "training passages",
    )
    train_pvec = train_pflat.reshape(TRAIN_N, 10, -1)
    dev_qvec, dev_q_trunc = encode_texts(
        [item["question"] for item in dev], student_tok, student, "DEV queries"
    )
    dev_pflat, dev_p_trunc = encode_texts(
        [paragraph for item in dev for paragraph in item["paragraphs"]],
        student_tok, student, "DEV passages",
    )
    dev_pvec = dev_pflat.reshape(DEV_N, 10, -1)

    log("Selecting all arm/budget pairs from student and cheap signals")
    orders = pair_orders(train, train_qvec, train_pvec)
    global_keys, arm_keys, plan_hash = selection_plan(train, orders)
    items = [
        (train[q]["question"], train[q]["paragraphs"][candidate])
        for q, candidate in global_keys
    ]
    token_counts, _ = teacher_token_counts(items, teacher_tok)
    cost_records = make_cost_records(train, global_keys, arm_keys, token_counts)
    progress_path = BASE / "progress.jsonl"
    started_at = ensure_progress_started(progress_path)
    write_plan_record(progress_path, plan_hash, cost_records, len(global_keys))
    log(
        f"Plan frozen: {len(global_keys)} physical selected teacher calls; "
        f"arm budgets recorded in progress.jsonl"
    )

    teacher_values, checkpoints = score_selected_with_checkpoints(
        train, global_keys, token_counts, teacher_tok, teacher, plan_hash, progress_path
    )
    teacher_matrix = np.full((TRAIN_N, 10), np.nan, dtype=np.float32)
    for (q, candidate), score in zip(global_keys, teacher_values):
        teacher_matrix[q, candidate] = score

    log("Training and evaluating the 24 preregistered student adapter fits")
    outcomes, seed_records = train_and_evaluate(
        train, dev, orders, teacher_matrix, train_qvec, train_pvec, dev_qvec, dev_pvec, TRAIN_SEEDS
    )
    results = build_results(
        split, selected_train_ids, teacher_files, student_files,
        train_q_trunc, train_p_trunc, dev_q_trunc, dev_p_trunc,
        outcomes, seed_records, cost_records, global_keys, token_counts,
        plan_hash, checkpoints, started_at,
    )
    write_final_outputs(results)
    log("COMPLETE: results.json, RESULT.md, and manifest.sha256 written")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-test", action="store_true")
    args = parser.parse_args()
    if args.unit_test:
        run_unit_test()
    else:
        run_real()


if __name__ == "__main__":
    main()
