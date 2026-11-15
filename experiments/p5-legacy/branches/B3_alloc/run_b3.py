from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
PILOT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot")
DATA = PILOT / "data" / "hotpot_validation.parquet"
STUDENT_SNAPSHOT = PILOT / ".hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TEACHER_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
TEACHER_REVISION = "ce0834f22110de6d9222af7a7a03628121708969"
TEACHER_MODEL_SHA256 = "821d1aa69520101d6e0737f78a042ae25b19e5cb9160701909d10434f4aeb0ae"
SEED = 20261031
TRAIN_SEEDS = (20261031, 20261032, 20261033)
SAMPLE_N, DEV_N, HOLDOUT_N = 600, 200, 400
TRAIN_POOL_N = 6742
MAX_LENGTH = 256
TEACHER_BUDGETS = (2, 5, 10)
ALLOCATIONS = ("random", "topk", "prd_rrdiff", "set_aware")
BOOTSTRAPS = 1000
WALL_LIMIT = 3 * 60 * 60

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = str(BASE / ".model-cache")
os.environ["HF_HUB_CACHE"] = str(BASE / ".model-cache" / "hub")
os.environ["TORCH_HOME"] = str(BASE / ".torch-cache")
os.environ["XDG_CACHE_HOME"] = str(BASE / ".cache")

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
DEVICE = torch.device("cpu")


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[{elapsed:8.1f}s] {message}", flush=True)


def check_time() -> None:
    if time.monotonic() - STARTED >= WALL_LIMIT - 45:
        raise TimeoutError("Stopping with 45 seconds reserved inside the 3-hour wall limit")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: list[str] | tuple[str, ...]) -> str:
    payload = "\n".join(sorted(ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_title(title: str) -> str:
    return " ".join(str(title).split()).casefold()


def load_split_metadata() -> tuple[list[str], list[str], list[str], dict]:
    log("Reading eligibility columns to reconstruct the registered split")
    table = pq.read_table(DATA, columns=["id", "type", "context", "supporting_facts"])
    ids = [str(x) for x in table.column("id").to_pylist()]
    kinds = table.column("type").to_pylist()
    context = table.column("context").combine_chunks()
    supporting = table.column("supporting_facts").combine_chunks()
    titles = pc.struct_field(context, "title").to_pylist()
    sentence_counts = pc.list_value_length(pc.struct_field(context, "sentences")).to_pylist()
    gold_titles = pc.struct_field(supporting, "title").to_pylist()
    reasons: Counter[str] = Counter()
    eligible_indices: list[int] = []
    for index, (kind, candidate_titles, n_sentences, row_gold) in enumerate(
        zip(kinds, titles, sentence_counts, gold_titles)
    ):
        reason = None
        if kind not in {"bridge", "comparison"}:
            reason = "unsupported_type"
        elif candidate_titles is None or n_sentences is None or len(candidate_titles) != 10 or n_sentences != 10:
            reason = "candidate_count_not_10"
        else:
            gold = {normalize_title(x) for x in (row_gold or [])}
            if len(gold) != 2:
                reason = "unique_gold_titles_not_2"
            elif sum(normalize_title(x) in gold for x in candidate_titles) != 2:
                reason = "gold_candidates_not_exactly_2"
        if reason is None:
            eligible_indices.append(index)
        else:
            reasons[reason] += 1
    del context, supporting, titles, sentence_counts, gold_titles, table
    gc.collect()
    if len(ids) != 7405 or len(eligible_indices) != 7342:
        raise ValueError(f"Input audit changed: rows={len(ids)}, eligible={len(eligible_indices)}")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source question ids")
    eligible_ids = [ids[i] for i in eligible_indices]
    sampled_indices = np.random.default_rng(SEED).choice(len(eligible_ids), SAMPLE_N, replace=False)
    sampled = [eligible_ids[int(i)] for i in sampled_indices]
    sampled.sort(key=lambda qid: (hashlib.sha256(qid.encode("utf-8")).hexdigest(), qid))
    dev_ids, holdout_ids = sampled[:DEV_N], sampled[DEV_N:]
    sampled_set = set(sampled)
    train_ids = [qid for qid in eligible_ids if qid not in sampled_set]
    if len(sampled_set) != SAMPLE_N or len(train_ids) != TRAIN_POOL_N:
        raise ValueError("Registered sample sizes, uniqueness, or training exclusion check failed")
    meta = {
        "source": str(DATA),
        "source_url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
        "source_bytes": DATA.stat().st_size,
        "source_sha256": sha256_file(DATA),
        "source_rows": len(ids),
        "eligible_rows": len(eligible_ids),
        "excluded_rows_by_reason": dict(sorted(reasons.items())),
        "sample_seed": SEED,
        "sample_n": SAMPLE_N,
        "sampling": "numpy.default_rng(20261031).choice(eligible_source_order, 600, replace=False)",
        "split": "sort sampled ids by (sha256(UTF-8 id), id); first 200 DEV; last 400 HOLDOUT",
        "dev_n": len(dev_ids),
        "dev_ids_sha256_sorted": sha256_ids(dev_ids),
        "sampled_600_ids_sha256_sorted": sha256_ids(sampled),
        "holdout_n": len(holdout_ids),
        "training_pool_n": len(train_ids),
        "training_pool_ids_sha256_sorted": sha256_ids(train_ids),
        "training_pool_definition": "eligible source rows excluding all 600 sampled ids",
        "holdout_content_used_for_training_or_evaluation": False,
    }
    manifest_path = BASE / "split_manifest.json"
    encoded = json.dumps(meta, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text() != encoded:
            raise FileExistsError("Existing split manifest differs; refusing to overwrite")
    else:
        manifest_path.write_text(encoded)
    log(f"Split reconstructed: DEV={len(dev_ids)}, HOLDOUT={len(holdout_ids)}, train pool={len(train_ids)}")
    return dev_ids, holdout_ids, train_ids, meta


def load_rows(ids: list[str]) -> list[dict]:
    if not ids:
        return []
    dataset = ds.dataset(DATA, format="parquet")
    table = dataset.to_table(
        columns=["id", "question", "type", "context", "supporting_facts"],
        filter=ds.field("id").isin(ids),
    )
    rows = table.to_pylist()
    if {str(row["id"]) for row in rows} != set(ids) or len(rows) != len(ids):
        raise ValueError("Filtered data ids do not exactly match requested non-HOLDOUT ids")
    return rows


def prepare_rows(rows: list[dict]) -> list[dict]:
    prepared = []
    for row in rows:
        context = row["context"]
        titles, sentence_groups = context["title"], context["sentences"]
        gold_titles = {normalize_title(x) for x in row["supporting_facts"]["title"]}
        if len(titles) != 10 or len(sentence_groups) != 10 or len(gold_titles) != 2:
            raise ValueError(f"Malformed eligible row {row['id']}")
        paragraphs = [
            (str(title).strip() + " " + " ".join(str(s).strip() for s in group)).strip()
            for title, group in zip(titles, sentence_groups)
        ]
        gold = {i for i, title in enumerate(titles) if normalize_title(title) in gold_titles}
        if len(gold) != 2:
            raise ValueError(f"Expected exactly two gold candidate paragraphs in {row['id']}")
        prepared.append({
            "qid": str(row["id"]),
            "question": str(row["question"]),
            "type": str(row["type"]),
            "paragraphs": paragraphs,
            "gold": sorted(gold),
        })
    return prepared


def snapshot_files(path: Path) -> list[dict]:
    result = []
    for file in sorted(path.rglob("*")):
        if file.is_file():
            result.append({"path": str(file.relative_to(path)), "bytes": file.stat().st_size, "sha256": sha256_file(file)})
    return result


def load_models():
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    log("Resolving the pinned public cross-encoder revision into the B3-local cache")
    teacher_path = Path(snapshot_download(
        repo_id=TEACHER_ID,
        revision=TEACHER_REVISION,
        cache_dir=str(BASE / ".model-cache" / "hub"),
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"],
        token=False,
    ))
    teacher_files = snapshot_files(teacher_path)
    weight_file = next((x for x in teacher_files if x["path"] == "model.safetensors"), None)
    if weight_file is None or weight_file["bytes"] > 500_000_000:
        raise ValueError("Teacher weights missing or exceed the 500 MB download limit")
    if weight_file["sha256"] != TEACHER_MODEL_SHA256:
        raise ValueError("Pinned teacher weights fail the published SHA256 check")
    student_files = snapshot_files(STUDENT_SNAPSHOT)
    student_weight = next((x for x in student_files if x["path"] == "model.safetensors"), None)
    if student_weight is None or student_weight["bytes"] > 150_000_000:
        raise ValueError("Cached MiniLM student weights are missing or larger than 150 MB")
    teacher_tok = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True, use_fast=True)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_path, local_files_only=True, use_safetensors=True
    ).to(DEVICE).eval()
    student_tok = AutoTokenizer.from_pretrained(STUDENT_SNAPSHOT, local_files_only=True, use_fast=True)
    student = AutoModel.from_pretrained(
        STUDENT_SNAPSHOT, local_files_only=True, use_safetensors=True
    ).to(DEVICE).eval()
    return teacher_tok, teacher, teacher_files, student_tok, student, student_files


def tokenizer_batch(tokenizer, texts, max_length: int) -> tuple[dict, int]:
    enc = tokenizer(texts, padding=False, truncation=False)
    features = []
    truncated = 0
    for index, ids in enumerate(enc["input_ids"]):
        token_types = enc.get("token_type_ids")
        if len(ids) > max_length:
            truncated += 1
            ids = ids[:max_length]
            if tokenizer.sep_token_id is not None:
                ids[-1] = tokenizer.sep_token_id
        feature = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if token_types is not None:
            feature["token_type_ids"] = token_types[index][:len(ids)]
        features.append(feature)
    return tokenizer.pad(features, padding=True, return_tensors="pt"), truncated


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
            check_time()
            log(f"Encoded {role}: {min(start + 32, len(texts))}/{len(texts)}")
    return np.concatenate(vectors, axis=0), truncated_total


@torch.inference_mode()
def teacher_scores(items: list[tuple[str, str]], tokenizer, model) -> tuple[np.ndarray, int]:
    scores, truncations = [], 0
    for start in range(0, len(items), 24):
        batch_items = items[start:start + 24]
        paired = [(query, passage) for query, passage in batch_items]
        batch, truncated = tokenizer_batch(tokenizer, paired, MAX_LENGTH)
        truncations += truncated
        batch = {key: value.to(DEVICE) for key, value in batch.items()}
        values = model(**batch).logits.reshape(-1).float().cpu().numpy()
        scores.extend(float(x) for x in values)
        if start and start % 1200 == 0:
            check_time()
            log(f"Teacher-scored {min(start + 24, len(items))}/{len(items)} query-passage items")
    return np.asarray(scores, dtype=np.float32), truncations


def dense_scores(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    return np.einsum("qd,qpd->qp", queries, passages, optimize=True)


def full_support_at(scores: np.ndarray, examples: list[dict], k: int) -> np.ndarray:
    result = np.zeros(len(examples), dtype=np.float32)
    for q, example in enumerate(examples):
        order = np.argsort(-scores[q], kind="stable")[:k]
        result[q] = float(set(example["gold"]).issubset(set(order.tolist())))
    return result


def bootstrap_mean(values: np.ndarray, bootstrap_indices: np.ndarray) -> tuple[float, float, float]:
    point = float(np.mean(values))
    boot = values[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def pair_orders(examples: list[dict], query_vectors: np.ndarray, passage_vectors: np.ndarray) -> dict[str, list[list[tuple[int, int]]]]:
    all_pairs = list(itertools.combinations(range(10), 2))
    result = {name: [] for name in ALLOCATIONS}
    for q, example in enumerate(examples):
        qid = example["qid"]
        rel = passage_vectors[q] @ query_vectors[q]
        rank = np.empty(10, dtype=np.int64)
        rank[np.argsort(-rel, kind="stable")] = np.arange(1, 11)
        seed_bytes = hashlib.sha256(f"{SEED}\0{qid}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
        rand_order = [all_pairs[i] for i in rng.permutation(len(all_pairs))[:max(TEACHER_BUDGETS)]]
        top_order = sorted(
            all_pairs,
            key=lambda ij: (min(rank[ij[0]], rank[ij[1]]), max(rank[ij[0]], rank[ij[1]]), ij),
        )[:max(TEACHER_BUDGETS)]
        weights = np.asarray([abs(1.0 / rank[i] - 1.0 / rank[j]) for i, j in all_pairs], dtype=np.float64)
        prd_idx = rng.choice(len(all_pairs), size=max(TEACHER_BUDGETS), replace=False, p=weights / weights.sum())
        prd_order = [all_pairs[int(i)] for i in prd_idx]
        passage_sim = passage_vectors[q] @ passage_vectors[q].T
        remaining = set(range(len(all_pairs)))
        counts = np.zeros(10, dtype=np.int32)
        set_order = []
        for _ in range(max(TEACHER_BUDGETS)):
            best_index, best_score = None, -math.inf
            for pair_index in sorted(remaining):
                i, j = all_pairs[pair_index]
                joint = 0.5 * (rel[i] + rel[j]) + 0.20 * (1.0 - float(passage_sim[i, j]))
                score = joint / math.sqrt((1 + counts[i]) * (1 + counts[j]))
                if score > best_score:
                    best_index, best_score = pair_index, score
            assert best_index is not None
            i, j = all_pairs[best_index]
            set_order.append((i, j))
            counts[i] += 1
            counts[j] += 1
            remaining.remove(best_index)
        result["random"].append(rand_order)
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
    q, p = torch.from_numpy(query_vectors), torch.from_numpy(passage_vectors)
    qidx, left, right, signs = (torch.from_numpy(x) for x in pair_rows)
    with torch.no_grad():
        if not torch.allclose(adapter(q), q, atol=1e-6, rtol=1e-6):
            raise AssertionError("Zero-initialized query adapter must reproduce the base bi-encoder")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(3):
        permutation = np.random.default_rng(seed + epoch).permutation(len(qidx))
        for start in range(0, len(permutation), 512):
            ix = torch.from_numpy(permutation[start:start + 512])
            qi, li, ri, yi = qidx[ix], left[ix], right[ix], signs[ix]
            qout = adapter(q[qi])
            score_l = (qout * p[qi, li]).sum(dim=1)
            score_r = (qout * p[qi, ri]).sum(dim=1)
            loss = F.softplus(-yi * (score_l - score_r)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return adapter.eval()


def adapter_scores(adapter: QueryAdapter, query_vectors: np.ndarray, passage_vectors: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        q = adapter(torch.from_numpy(query_vectors)).cpu().numpy()
    return dense_scores(q, passage_vectors)


def collect_pair_rows(
    orders: list[list[tuple[int, int]]], budget: int, teacher_matrix: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]:
    qidx, left, right, signs = [], [], [], []
    skipped_ties = 0
    for q, order in enumerate(orders):
        for i, j in order[:budget]:
            diff = float(teacher_matrix[q, i] - teacher_matrix[q, j])
            if not math.isfinite(diff):
                raise ValueError("Missing or non-finite teacher label on an allocated pair")
            if diff == 0.0:
                skipped_ties += 1
                continue
            qidx.append(q)
            left.append(i)
            right.append(j)
            signs.append(1.0 if diff > 0 else -1.0)
    if not qidx:
        raise ValueError("No non-tied teacher comparisons to train on")
    return (
        np.asarray(qidx, np.int64), np.asarray(left, np.int64),
        np.asarray(right, np.int64), np.asarray(signs, np.float32),
    ), skipped_ties


def write_results(metrics: list[dict], result_text: str) -> None:
    metrics_path, result_path = BASE / "metrics.jsonl", BASE / "RESULT.md"
    if metrics_path.exists() or result_path.exists():
        raise FileExistsError("Refusing to overwrite existing final outputs")
    with metrics_path.open("x") as handle:
        for record in metrics:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    metrics_hash = sha256_file(metrics_path)
    result_text = result_text.replace("METRICS_SHA256", metrics_hash)
    with result_path.open("x") as handle:
        handle.write(result_text)
    with (BASE / "SHA256SUMS").open("x") as handle:
        handle.write(f"{metrics_hash}  metrics.jsonl\n")
        handle.write(f"{sha256_file(result_path)}  RESULT.md\n")
        handle.write(f"{sha256_file(BASE / 'split_manifest.json')}  split_manifest.json\n")
        handle.write(f"{sha256_file(BASE / 'run_b3.py')}  run_b3.py\n")


def result_markdown(status: str, lines: list[str]) -> str:
    body = [
        "# B3_alloc: set-aware teacher-feedback allocation",
        "",
        f"**Status:** {status}",
        "",
        *lines,
        "",
        "- SHA256(metrics.jsonl): METRICS_SHA256",
        "- SHA256(RESULT.md): recorded in SHA256SUMS because a file cannot contain its own final hash.",
        "",
    ]
    return "\n".join(body)


def main() -> None:
    global STARTED
    STARTED = time.monotonic()
    if not DATA.is_file() or not STUDENT_SNAPSHOT.is_dir():
        raise FileNotFoundError("Authorized pilot data or cached student model is missing")
    if any((BASE / name).exists() for name in ("metrics.jsonl", "RESULT.md", "SHA256SUMS")):
        raise FileExistsError("Final artifacts already exist; refusing to overwrite")
    if torch.get_num_threads() != 2 or DEVICE.type != "cpu":
        raise RuntimeError("CPU thread/device constraint failed")
    log(f"Python {sys.version.split()[0]}, torch {torch.__version__}, CPU={platform.machine()}, threads=2")

    dev_ids, holdout_ids, train_ids, data_meta = load_split_metadata()
    dev = prepare_rows(load_rows(dev_ids))
    if len(dev) != DEV_N or {x["qid"] for x in dev} != set(dev_ids):
        raise AssertionError("DEV materialization check failed")
    log("Loading pinned local teacher and cached MiniLM student")
    teacher_tok, teacher, teacher_files, student_tok, student, student_files = load_models()
    dev_questions = [x["question"] for x in dev]
    dev_paragraphs = [p for x in dev for p in x["paragraphs"]]
    dev_qvec, trunc_q_dev = encode_texts(dev_questions, student_tok, student, "DEV queries")
    dev_pflat, trunc_p_dev = encode_texts(dev_paragraphs, student_tok, student, "DEV passages")
    dev_pvec = dev_pflat.reshape(DEV_N, 10, -1)
    student_dev_scores = dense_scores(dev_qvec, dev_pvec)
    dev_teacher_items = [(x["question"], p) for x in dev for p in x["paragraphs"]]
    teacher_dev_flat, teacher_trunc_dev = teacher_scores(dev_teacher_items, teacher_tok, teacher)
    teacher_dev_scores = teacher_dev_flat.reshape(DEV_N, 10)
    boot_idx = np.random.default_rng(0).integers(0, DEV_N, size=(BOOTSTRAPS, DEV_N))
    gate_records, gate_lines = [], [
        "## Hypothesis and setup",
        "Set-aware pair allocation should improve a MiniLM bi-encoder's multi-hop FULL-SUPPORT over random, top-ranked, and PRD-style informative-pair allocation at equal labeled-pair budgets.",
        f"HotpotQA distractor validation; {TRAIN_POOL_N} eligible non-sample training questions; DEV={DEV_N}; HOLDOUT={HOLDOUT_N} sampled rows excluded from model inputs and metrics. CPU only, 2 threads, max length {MAX_LENGTH}; no paid/remote LLM calls.",
        f"Teacher: {TEACHER_ID} revision {TEACHER_REVISION}. Student: cached all-MiniLM-L6-v2 plus a trainable 384→64→384 query adapter. Primary headroom gate: paired 1,000-resample DEV bootstrap CI for teacher minus student FS@2 must be wholly above zero.",
        "",
        "## Teacher headroom gate (DEV only)",
    ]
    for k in (2, 4):
        student_outcome = full_support_at(student_dev_scores, dev, k)
        teacher_outcome = full_support_at(teacher_dev_scores, dev, k)
        delta_stats = bootstrap_mean(teacher_outcome - student_outcome, boot_idx)
        student_stats = bootstrap_mean(student_outcome, boot_idx)
        teacher_stats = bootstrap_mean(teacher_outcome, boot_idx)
        gate_records.append({
            "record_type": "teacher_headroom",
            "metric": f"FS@{k}",
            "n_questions": DEV_N,
            "student_mean_ci95": student_stats,
            "teacher_mean_ci95": teacher_stats,
            "teacher_minus_student_ci95": delta_stats,
            "bootstrap_resamples": BOOTSTRAPS,
            "bootstrap_unit": "question",
        })
        gate_lines.append(
            f"- FS@{k}: student {student_stats[0]:.3f} [{student_stats[1]:.3f}, {student_stats[2]:.3f}]; teacher {teacher_stats[0]:.3f} [{teacher_stats[1]:.3f}, {teacher_stats[2]:.3f}]; Δ {delta_stats[0]:+.3f} [{delta_stats[1]:+.3f}, {delta_stats[2]:+.3f}]."
        )
    headroom_pass = gate_records[0]["teacher_minus_student_ci95"][1] > 0
    gate_lines.extend([
        f"- Gate: {'PASS' if headroom_pass else 'FAIL'}; FS@2 is the predeclared primary criterion.",
        f"- Data SHA256 {data_meta['source_sha256']}; teacher weight SHA256 {TEACHER_MODEL_SHA256}; DEV ids hash {data_meta['dev_ids_sha256_sorted']}; training pool ids hash {data_meta['training_pool_ids_sha256_sorted']}.",
        f"- Truncations: student DEV queries/passages {trunc_q_dev}/{trunc_p_dev}; teacher DEV pairs {teacher_trunc_dev}.",
        "- Sanity: 200 DEV rows, 10 candidates and 2 gold paragraphs each; HOLDOUT rows are excluded from model inputs and evaluation.",
        "- Limitation: one fixed 200-question DEV sample and one MS MARCO teacher; headroom conclusion does not establish a general advantage.",
    ])
    if not headroom_pass:
        gate_records[0]["gate"] = "fail_stop"
        gate_records[0]["reason"] = "teacher FS@2 paired bootstrap CI did not lie wholly above zero"
        gate_records.append({
            "record_type": "run_manifest",
            "data": data_meta,
            "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION, "files": teacher_files},
            "student": {"id": "sentence-transformers/all-MiniLM-L6-v2", "revision": STUDENT_SNAPSHOT.name, "files": student_files},
            "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pyarrow": importlib.metadata.version("pyarrow"),
                        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                        "platform": platform.platform(), "machine": platform.machine(), "cpu": platform.processor(),
                        "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(), "device": str(DEVICE)},
            "timing": {"wall_seconds": time.monotonic() - STARTED},
        })
        write_results(gate_records, result_markdown("STOPPED: teacher not clearly stronger on DEV", gate_lines))
        log("Teacher headroom gate failed; no training was performed")
        return

    log("Teacher passed the DEV headroom gate; loading the training pool")
    train = prepare_rows(load_rows(train_ids))
    if len(train) != TRAIN_POOL_N or {x["qid"] for x in train} != set(train_ids):
        raise AssertionError("Training pool materialization or id membership check failed")
    if ({x["qid"] for x in train} & set(holdout_ids)) or ({x["qid"] for x in dev} & set(holdout_ids)):
        raise AssertionError("HOLDOUT id entered model data")
    train_questions = [x["question"] for x in train]
    train_paragraphs = [p for x in train for p in x["paragraphs"]]
    train_qvec, trunc_q_train = encode_texts(train_questions, student_tok, student, "training queries")
    train_pflat, trunc_p_train = encode_texts(train_paragraphs, student_tok, student, "training passages")
    train_pvec = train_pflat.reshape(TRAIN_POOL_N, 10, -1)
    log("Constructing deterministic pair-allocation orders")
    orders = pair_orders(train, train_qvec, train_pvec)

    needed = [set() for _ in train]
    for method in ALLOCATIONS:
        for q, order in enumerate(orders[method]):
            for i, j in order[:max(TEACHER_BUDGETS)]:
                needed[q].add(i)
                needed[q].add(j)
    teacher_items, teacher_item_keys = [], []
    for q, indices in enumerate(needed):
        for candidate in sorted(indices):
            teacher_items.append((train[q]["question"], train[q]["paragraphs"][candidate]))
            teacher_item_keys.append((q, candidate))
    log(f"Teacher feedback cache contains {len(teacher_items)} unique training query-passage scores")
    train_teacher_flat, teacher_trunc_train = teacher_scores(teacher_items, teacher_tok, teacher)
    train_teacher = np.full((TRAIN_POOL_N, 10), np.nan, dtype=np.float32)
    for (q, candidate), score in zip(teacher_item_keys, train_teacher_flat):
        train_teacher[q, candidate] = score

    outcomes: dict[tuple[str, int, int, int], np.ndarray] = {}
    train_records = []
    seed_values: dict[tuple[str, int, int], list[float]] = {}
    for method in ALLOCATIONS:
        for budget in TEACHER_BUDGETS:
            pair_rows, tie_count = collect_pair_rows(orders[method], budget, train_teacher)
            log(f"Training {method}, budget={budget}: {len(pair_rows[0])} pair labels; ties skipped={tie_count}")
            for seed in TRAIN_SEEDS:
                check_time()
                fit_started = time.monotonic()
                adapter = fit_adapter(train_qvec, train_pvec, pair_rows, seed)
                scores = adapter_scores(adapter, dev_qvec, dev_pvec)
                for k in (2, 4):
                    outcomes[(method, budget, seed, k)] = full_support_at(scores, dev, k)
                    seed_values.setdefault((method, budget, k), []).append(
                        float(outcomes[(method, budget, seed, k)].mean())
                    )
                train_records.append({
                    "record_type": "seed_run",
                    "allocation": method,
                    "budget_pair_labels_per_question": budget,
                    "seed": seed,
                    "train_pair_labels": int(len(pair_rows[0])),
                    "teacher_ties_skipped": tie_count,
                    "epochs": 3,
                    "training_seconds": time.monotonic() - fit_started,
                })
                del adapter, scores
                gc.collect()

    metrics = list(gate_records)
    metrics[0]["gate"] = "pass"
    metrics.extend(train_records)
    summary_stats = {}
    for method in ALLOCATIONS:
        for budget in TEACHER_BUDGETS:
            for k in (2, 4):
                by_question = np.stack([outcomes[(method, budget, seed, k)] for seed in TRAIN_SEEDS]).mean(axis=0)
                stats = bootstrap_mean(by_question, boot_idx)
                summary_stats[(method, budget, k)] = stats
                metrics.append({
                    "record_type": "summary",
                    "allocation": method,
                    "budget_pair_labels_per_question": budget,
                    "metric": f"FS@{k}",
                    "mean_across_seeds": stats[0],
                    "ci95_question_bootstrap": [stats[1], stats[2]],
                    "bootstrap_resamples": BOOTSTRAPS,
                    "bootstrap_unit": "question; seed outcomes averaged within question",
                    "seed_means": seed_values[(method, budget, k)],
                    "train_pool_n": TRAIN_POOL_N,
                    "dev_n": DEV_N,
                })
    for budget in TEACHER_BUDGETS:
        for comparator in ("random", "topk", "prd_rrdiff"):
            for k in (2, 4):
                set_q = np.stack([outcomes[("set_aware", budget, seed, k)] for seed in TRAIN_SEEDS]).mean(axis=0)
                cmp_q = np.stack([outcomes[(comparator, budget, seed, k)] for seed in TRAIN_SEEDS]).mean(axis=0)
                stats = bootstrap_mean(set_q - cmp_q, boot_idx)
                metrics.append({
                    "record_type": "paired_contrast",
                    "allocation": "set_aware",
                    "comparator": comparator,
                    "budget_pair_labels_per_question": budget,
                    "metric": f"FS@{k}",
                    "difference": stats[0],
                    "ci95_question_bootstrap": [stats[1], stats[2]],
                    "bootstrap_resamples": BOOTSTRAPS,
                    "bootstrap_unit": "paired question; seed outcomes averaged within question",
                })

    lines = [
        "## Hypothesis and setup",
        "Set-aware allocation should improve a MiniLM bi-encoder's multi-hop FULL-SUPPORT over random, top-ranked, and PRD-style informative-pair allocation at equal teacher-pair budgets.",
        f"HotpotQA distractor validation; training pool {TRAIN_POOL_N}; DEV {DEV_N}; sampled HOLDOUT {HOLDOUT_N} excluded. CPU only, OMP/PyTorch threads=2, max length {MAX_LENGTH}. PRD comparator uses RRDiff weighted sampling without replacement; set-aware greedily rewards relevance plus passage diversity and balances endpoint reuse.",
        f"Teacher {TEACHER_ID} revision {TEACHER_REVISION}; student cached MiniLM with trainable 384→64→384 query adapter. Three seeds {list(TRAIN_SEEDS)}; 3 fixed epochs. Bootstrap: 1,000 paired question resamples, seed 0.",
        "",
        "## Teacher headroom gate (DEV)",
        *gate_lines[5:10],
        "",
        "## FULL-SUPPORT by budget: mean over three seeds [95% question-bootstrap CI]",
        "| Pair labels/q | Metric | Random | Top-k | PRD-RRDiff | Set-aware |",
        "|---:|:---|:---|:---|:---|:---|",
    ]
    for budget in TEACHER_BUDGETS:
        for k in (2, 4):
            cells = []
            for method in ALLOCATIONS:
                point, low, high = summary_stats[(method, budget, k)]
                cells.append(f"{point:.3f} [{low:.3f}, {high:.3f}]")
            lines.append(f"| {budget} | FS@{k} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Paired set-aware differences (FS@2; positive favors set-aware)"])
    lines.extend(["| Pair labels/q | vs random | vs top-k | vs PRD-RRDiff |", "|---:|:---|:---|:---|"])
    for budget in TEACHER_BUDGETS:
        diffs = []
        for comparator in ("random", "topk", "prd_rrdiff"):
            record = next(x for x in metrics if x.get("record_type") == "paired_contrast"
                          and x.get("budget_pair_labels_per_question") == budget
                          and x.get("comparator") == comparator and x.get("metric") == "FS@2")
            low, high = record["ci95_question_bootstrap"]
            diffs.append(f"{record['difference']:+.3f} [{low:+.3f}, {high:+.3f}]")
        lines.append(f"| {budget} | " + " | ".join(diffs) + " |")
    lines.extend([
        "",
        "## Sanity checks and limitations",
        f"- Eligible population 7,342; DEV 200; training pool {TRAIN_POOL_N}; unique DEV/training id hashes recorded in split_manifest.json. HOLDOUT question/paragraph rows never entered model input or metrics.",
        f"- Raw data SHA256 {data_meta['source_sha256']}; teacher weights SHA256 {TEACHER_MODEL_SHA256}; student and teacher per-file hashes are in metrics.jsonl.",
        f"- Truncations at {MAX_LENGTH}: DEV student query/passages {trunc_q_dev}/{trunc_p_dev}; train student query/passages {trunc_q_train}/{trunc_p_train}; teacher DEV/train {teacher_trunc_dev}/{teacher_trunc_train}.",
        "- Gold facts are used for eligibility and DEV evaluation, never as student targets. Teacher logits are memoized for unique query-passage pairs; each allocation still uses its exact 2/5/10 pair-label budget.",
        "- Limitations: fixed teacher/student family, adapter-only fine-tuning, one 200-question DEV sample, and heuristic set-complementarity; exploratory feasibility only.",
        f"- Runtime: Python {sys.version.split()[0]}, NumPy {np.__version__}, PyArrow {importlib.metadata.version('pyarrow')}, PyTorch {torch.__version__}, Transformers {importlib.metadata.version('transformers')}; CPU {platform.platform()} / {platform.processor()}, threads=2.",
    ])
    metrics.append({
        "record_type": "run_manifest",
        "data": data_meta,
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION, "files": teacher_files},
        "student": {"id": "sentence-transformers/all-MiniLM-L6-v2", "revision": STUDENT_SNAPSHOT.name, "files": student_files},
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pyarrow": importlib.metadata.version("pyarrow"),
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": platform.processor(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": str(DEVICE),
        },
        "model_input_truncations": {
            "student_dev_queries": trunc_q_dev,
            "student_dev_passages": trunc_p_dev,
            "student_train_queries": trunc_q_train,
            "student_train_passages": trunc_p_train,
            "teacher_dev_pairs": teacher_trunc_dev,
            "teacher_train_pairs": teacher_trunc_train,
        },
        "training": {
            "seeds": TRAIN_SEEDS,
            "epochs": 3,
            "teacher_pair_label_budgets": TEACHER_BUDGETS,
            "allocations": ALLOCATIONS,
            "physical_unique_teacher_query_passage_scores": len(teacher_items),
            "teacher_pair_labels_total": TRAIN_POOL_N * sum(TEACHER_BUDGETS) * len(ALLOCATIONS),
            "student_adapter": "residual MLP 384->64->384; frozen MiniLM text encoders; fixed passage vectors",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
        },
        "timing": {"wall_seconds": time.monotonic() - STARTED},
    })
    write_results(metrics, result_markdown("COMPLETE: teacher passed DEV gate; exploratory training/evaluation finished", lines))
    log(f"Complete in {time.monotonic() - STARTED:.1f}s; metrics.jsonl and RESULT.md written")


if __name__ == "__main__":
    main()

