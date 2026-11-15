from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import transformers
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


BASE = Path(__file__).resolve().parent
DATA_PATH = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/data/hotpot_validation.parquet")
MODEL_PATH = Path(
    "/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/.hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
SAMPLE_SEED = 20261031
TRAIN_SEED = 20261032
CROSS_SEED = 20261033
BOOTSTRAP_SEED = 0
SAMPLE_N = 600
DEV_N = 200
MAX_LENGTH = 256
THREADS = 2
BOOTSTRAPS = 1000
FRONTIER_THRESHOLDS = [round(v, 2) for v in np.arange(0.05, 1.0, 0.05)]
CROSS_TRAIN_QUERIES = 512
WALL_LIMIT_SECONDS = 3 * 60 * 60
SCREEN_COLUMNS = ["id", "type", "context.title", "context.sentences", "supporting_facts.title"]
FULL_COLUMNS = ["id", "question", "type", "context", "supporting_facts"]

torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

RUN_STARTED = time.monotonic()


def check_time() -> None:
    if time.monotonic() - RUN_STARTED > WALL_LIMIT_SECONDS:
        raise TimeoutError("Three-hour wall-clock limit reached; no result files written")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def normalize_title(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def eligible(row: dict) -> tuple[bool, str | None]:
    try:
        if row["type"] not in {"bridge", "comparison"}:
            return False, "unsupported_type"
        context = row["context"]
        titles = context["title"]
        sentence_groups = context["sentences"]
        if len(titles) != 10 or len(sentence_groups) != 10:
            return False, "candidate_count_not_10"
        gold = {normalize_title(title) for title in row["supporting_facts"]["title"]}
        if len(gold) != 2:
            return False, "unique_gold_titles_not_2"
        if sum(normalize_title(title) in gold for title in titles) != 2:
            return False, "gold_candidates_not_exactly_2"
        return True, None
    except (KeyError, TypeError):
        return False, "malformed_row"


def split_manifest() -> dict:
    parquet_file = pq.ParquetFile(DATA_PATH)
    eligible_meta: list[dict] = []
    excluded: Counter[str] = Counter()
    source_order = 0
    for batch in parquet_file.iter_batches(batch_size=128, columns=SCREEN_COLUMNS):
        for row in batch.to_pylist():
            ok, reason = eligible(row)
            if ok:
                eligible_meta.append(
                    {"id": str(row["id"]), "type": str(row["type"]), "source_order": source_order}
                )
            else:
                excluded[str(reason)] += 1
            source_order += 1
        check_time()
    eligible_ids = [row["id"] for row in eligible_meta]
    if len(eligible_ids) < SAMPLE_N or len(set(eligible_ids)) != len(eligible_ids):
        raise ValueError("Eligibility count or IDs do not match the expected sampling preconditions")

    sampled_indices = np.random.default_rng(SAMPLE_SEED).choice(
        len(eligible_meta), size=SAMPLE_N, replace=False
    )
    sampled = [eligible_meta[int(i)] for i in sampled_indices]
    sampled.sort(
        key=lambda row: (hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), row["id"])
    )
    dev_meta = sampled[:DEV_N]
    sampled_ids = {row["id"] for row in sampled}
    pool_meta = [row for row in eligible_meta if row["id"] not in sampled_ids]
    dev_ids = [row["id"] for row in dev_meta]
    pool_ids = [row["id"] for row in pool_meta]
    if len(dev_ids) != DEV_N or set(dev_ids) & set(pool_ids):
        raise AssertionError("DEV and training pool are not disjoint")
    # The 400 HOLDOUT IDs are used only to exclude them from the data scan below.
    allowed_ids = set(dev_ids) | set(pool_ids)
    if allowed_ids & (sampled_ids - set(dev_ids)):
        raise AssertionError("HOLDOUT IDs leaked into the train/DEV read set")
    stats = {
        "source_rows": source_order,
        "eligible_rows": len(eligible_meta),
        "excluded_rows": sum(excluded.values()),
        "excluded_reasons": dict(sorted(excluded.items())),
        "sample_n": SAMPLE_N,
        "dev_n": DEV_N,
        "holdout_n": SAMPLE_N - DEV_N,
        "dev_ids_sha256": ids_sha256(dev_ids),
        "sampled_600_ids_sha256": ids_sha256([row["id"] for row in sampled]),
        "training_pool_n": len(pool_ids),
        "training_pool_ids_sha256_source_order_newline_joined": ids_sha256(pool_ids),
        "dev_type_counts": dict(Counter(row["type"] for row in dev_meta)),
        "training_pool_type_counts": dict(Counter(row["type"] for row in pool_meta)),
        "holdout_full_rows_loaded": False,
        "screened_holdout_question_or_answer_fields": False,
    }
    stats["_dev_ids"] = dev_ids
    stats["_pool_ids"] = pool_ids
    stats["_allowed_ids"] = allowed_ids
    return stats


def load_train_dev_rows(allowed_ids: set[str]) -> dict[str, dict]:
    dataset = ds.dataset(str(DATA_PATH), format="parquet")
    table = dataset.to_table(
        columns=FULL_COLUMNS,
        filter=ds.field("id").isin(sorted(allowed_ids)),
    )
    rows = table.to_pylist()
    by_id = {str(row["id"]): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != allowed_ids:
        raise ValueError("Filtered train/DEV materialization did not return exactly the allowed IDs")
    return by_id


def prepare(row: dict) -> dict:
    qid = str(row["id"])
    kind = str(row["type"])
    question = str(row["question"])
    titles = row["context"]["title"]
    sentence_groups = row["context"]["sentences"]
    if kind not in {"bridge", "comparison"} or len(titles) != 10 or len(sentence_groups) != 10:
        raise ValueError(f"Malformed selected training/DEV row: {qid}")
    paragraphs = [
        (str(title).strip() + " " + " ".join(str(s).strip() for s in sentences)).strip()
        for title, sentences in zip(titles, sentence_groups)
    ]
    gold_titles = {normalize_title(x) for x in row["supporting_facts"]["title"]}
    gold = {i for i, title in enumerate(titles) if normalize_title(title) in gold_titles}
    if len(gold) != 2:
        raise ValueError(f"Selected row failed gold-paragraph integrity check: {qid}")
    return {"qid": qid, "type": kind, "question": question, "paragraphs": paragraphs, "gold": gold}


def load_dense_encoder():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Required local model snapshot is missing: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(MODEL_PATH), local_files_only=True, use_safetensors=True
    ).to("cpu")
    model.eval()
    model_files = []
    for path in sorted(p for p in MODEL_PATH.rglob("*") if p.is_file()):
        model_files.append(
            {"name": str(path.relative_to(MODEL_PATH)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return tokenizer, model, model_files


def encode_dense(texts: list[str], tokenizer, model) -> tuple[np.ndarray, int, int]:
    output: list[np.ndarray] = []
    truncated = 0
    total = len(texts)
    batch_size = 48
    for start in range(0, total, batch_size):
        check_time()
        chunk = texts[start : start + batch_size]
        raw = tokenizer(chunk, add_special_tokens=True, truncation=False, verbose=False)["input_ids"]
        truncated += sum(len(ids) > MAX_LENGTH for ids in raw)
        batch = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            verbose=False,
        )
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
            output.append(vectors.cpu().numpy().astype(np.float32, copy=False))
        if (start // batch_size + 1) % 100 == 0 or start + batch_size >= total:
            print(f"dense_encode texts={min(start + batch_size, total)}/{total}", flush=True)
    return np.concatenate(output), truncated, total


FEATURE_NAMES = [
    "k",
    "first_dense_score",
    "last_selected_score",
    "next_score",
    "boundary_gap",
    "top_mean_score",
    "top_score_std",
    "selected_pair_mean_cosine",
    "selected_pair_max_cosine",
    "selected_word_count_log1p",
    "last_passage_word_count_log1p",
    "question_word_count_log1p",
    "is_bridge",
]


def add_dense_order_and_features(items: list[dict], vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_labels = []
    for query_index, item in enumerate(items):
        start = query_index * 11
        q_vec = vectors[start]
        p_vec = vectors[start + 1 : start + 11]
        scores = p_vec @ q_vec
        order = np.argsort(-scores, kind="stable")
        item["order"] = order.tolist()
        item["prefix_labels"] = []
        ranked_scores = scores[order]
        ranked_vecs = p_vec[order]
        ranked_word_counts = np.array(
            [len(item["paragraphs"][i].split()) for i in order], dtype=np.float32
        )
        q_word_count = len(item["question"].split())
        for k in range(1, 11):
            selected_indices = set(order[:k].tolist())
            label = int(item["gold"].issubset(selected_indices))
            item["prefix_labels"].append(label)
            pair_values = []
            if k > 1:
                selected_vectors = ranked_vecs[:k]
                similarities = selected_vectors @ selected_vectors.T
                pair_values = similarities[np.triu_indices(k, k=1)].tolist()
            next_score = float(ranked_scores[k]) if k < 10 else 0.0
            features = [
                float(k),
                float(ranked_scores[0]),
                float(ranked_scores[k - 1]),
                next_score,
                float(ranked_scores[k - 1] - next_score),
                float(ranked_scores[:k].mean()),
                float(ranked_scores[:k].std()),
                float(np.mean(pair_values)) if pair_values else 0.0,
                float(np.max(pair_values)) if pair_values else 0.0,
                float(np.log1p(ranked_word_counts[:k].sum())),
                float(np.log1p(ranked_word_counts[k - 1])),
                float(np.log1p(q_word_count)),
                float(item["type"] == "bridge"),
            ]
            all_features.append(features)
            all_labels.append(label)
    return np.asarray(all_features, dtype=np.float32), np.asarray(all_labels, dtype=np.float32)


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, dev_x: np.ndarray) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = torch.from_numpy(((train_x - mean) / scale).astype(np.float32))
    y = torch.from_numpy(train_y.astype(np.float32))
    d = x.shape[1]
    weights = torch.zeros(d, dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.05)
    for step in range(400):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + 0.001 * weights.square().sum()
        loss.backward()
        optimizer.step()
        if step in {99, 199, 299, 399}:
            print(f"logistic_step={step + 1}/400 loss={float(loss.detach()):.6f}", flush=True)
    if not torch.isfinite(weights).all() or not torch.isfinite(bias):
        raise FloatingPointError("Logistic classifier produced non-finite parameters")
    x_dev = torch.from_numpy(((dev_x - mean) / scale).astype(np.float32))
    with torch.inference_mode():
        return torch.sigmoid(x_dev @ weights + bias).numpy().reshape(-1, 10)


def pair_texts(items: list[dict]) -> tuple[list[str], list[str], np.ndarray]:
    questions: list[str] = []
    contexts: list[str] = []
    labels: list[int] = []
    for item in items:
        ranked_paragraphs = [item["paragraphs"][i] for i in item["order"]]
        for k in range(1, 11):
            questions.append(item["question"])
            contexts.append(" [SEP] ".join(ranked_paragraphs[:k]))
            labels.append(item["prefix_labels"][k - 1])
    return questions, contexts, np.asarray(labels, dtype=np.int64)


def fit_cross_encoder(train_items: list[dict], tokenizer):
    torch.manual_seed(CROSS_SEED)
    rng = np.random.default_rng(TRAIN_SEED)
    n_queries = min(CROSS_TRAIN_QUERIES, len(train_items))
    chosen = np.sort(rng.choice(len(train_items), size=n_queries, replace=False))
    selected = [train_items[int(i)] for i in chosen]
    questions, contexts, labels = pair_texts(selected)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        num_labels=2,
        local_files_only=True,
        use_safetensors=True,
        ignore_mismatched_sizes=True,
    ).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    order = torch.randperm(len(labels))
    batch_size = 16
    steps = (len(labels) + batch_size - 1) // batch_size
    total_truncated = 0
    model.train()
    last_loss = None
    for step, start in enumerate(range(0, len(labels), batch_size), start=1):
        check_time()
        batch_indices = order[start : start + batch_size].tolist()
        q_batch = [questions[i] for i in batch_indices]
        c_batch = [contexts[i] for i in batch_indices]
        raw = tokenizer(
            q_batch, c_batch, add_special_tokens=True, truncation=False, padding=False, verbose=False
        )["input_ids"]
        total_truncated += sum(len(ids) > MAX_LENGTH for ids in raw)
        tokens = tokenizer(
            q_batch,
            c_batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            verbose=False,
        )
        y = torch.tensor(labels[batch_indices], dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = model(**tokens).logits
        loss = torch.nn.functional.cross_entropy(logits, y)
        last_loss = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step == steps:
            print(f"crossencoder_step={step}/{steps} loss={float(loss.detach()):.6f}", flush=True)
    model.eval()
    return model, {
        "training_queries": n_queries,
        "training_prefixes": len(labels),
        "training_truncated": total_truncated,
        "final_training_loss": last_loss,
    }


def score_cross_encoder(items: list[dict], tokenizer, model) -> tuple[np.ndarray, int]:
    questions, contexts, _ = pair_texts(items)
    probs = []
    truncated = 0
    batch_size = 24
    for start in range(0, len(questions), batch_size):
        check_time()
        q_batch = questions[start : start + batch_size]
        c_batch = contexts[start : start + batch_size]
        raw = tokenizer(
            q_batch, c_batch, add_special_tokens=True, truncation=False, padding=False, verbose=False
        )["input_ids"]
        truncated += sum(len(ids) > MAX_LENGTH for ids in raw)
        tokens = tokenizer(
            q_batch,
            c_batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            verbose=False,
        )
        with torch.inference_mode():
            logits = model(**tokens).logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        if (start // batch_size + 1) % 25 == 0 or start + batch_size >= len(questions):
            print(f"crossencoder_dev_prefixes={min(start + batch_size, len(questions))}/{len(questions)}", flush=True)
    return np.asarray(probs, dtype=np.float64).reshape(len(items), 10), truncated


def auc_binary(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(len(y), dtype=np.float64)
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def percentile_ci(values: list[float] | np.ndarray) -> list[float] | None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))]


def group_bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=(BOOTSTRAPS, n), endpoint=False)


def auc_with_ci(labels: np.ndarray, scores: np.ndarray, samples: np.ndarray) -> tuple[float | None, list[float] | None]:
    point_value = auc_binary(labels.reshape(-1), scores.reshape(-1))
    point = float(point_value) if np.isfinite(point_value) else None
    boots = [auc_binary(labels[ix].reshape(-1), scores[ix].reshape(-1)) for ix in samples]
    return point, percentile_ci(boots)


def stop_arrays(labels: np.ndarray, probs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    declared_by_k = probs >= threshold
    has_stop = declared_by_k.any(axis=1)
    first = declared_by_k.argmax(axis=1)
    chosen_k = np.where(has_stop, first + 1, 10)
    chosen_sufficient = labels[np.arange(len(labels)), chosen_k - 1].astype(bool)
    return chosen_k.astype(np.float64), has_stop, chosen_sufficient


def stop_metrics_with_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    samples: np.ndarray,
) -> dict:
    chosen_k, has_stop, sufficient = stop_arrays(labels, probs, threshold)
    declared_precision = float(sufficient[has_stop].mean()) if has_stop.any() else None
    point = {
        "precision_of_declared_sufficient_at_stop": declared_precision,
        "avg_passages_used_with_k10_fallback": float(chosen_k.mean()),
        "stop_coverage": float(has_stop.mean()),
        "sufficiency_accuracy_at_used_prefix": float(sufficient.mean()),
    }
    boot_k = chosen_k[samples].mean(axis=1)
    boot_cov = has_stop[samples].mean(axis=1)
    boot_acc = sufficient[samples].mean(axis=1)
    declared = has_stop[samples]
    true_declared = (has_stop & sufficient)[samples]
    denom = declared.sum(axis=1)
    boot_prec = np.divide(
        true_declared.sum(axis=1), denom,
        out=np.full(BOOTSTRAPS, np.nan, dtype=np.float64), where=denom > 0,
    )
    point.update(
        {
            "precision_of_declared_sufficient_at_stop_ci95": percentile_ci(boot_prec),
            "avg_passages_used_with_k10_fallback_ci95": percentile_ci(boot_k),
            "stop_coverage_ci95": percentile_ci(boot_cov),
            "sufficiency_accuracy_at_used_prefix_ci95": percentile_ci(boot_acc),
        }
    )
    return point


def fixed_k_metrics(labels: np.ndarray, k: int, samples: np.ndarray) -> dict:
    success = labels[:, k - 1].astype(np.float64)
    boots = success[samples].mean(axis=1)
    return {
        "avg_passages_used": k,
        "sufficiency_rate": float(success.mean()),
        "sufficiency_rate_ci95": percentile_ci(boots),
    }


def format_metric(value, ci, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f} [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else f"{value:.{digits}f} [NA]"


def main() -> None:
    metrics_path = BASE / "metrics.jsonl"
    result_path = BASE / "RESULT.md"
    if metrics_path.exists() or result_path.exists():
        raise FileExistsError("Refusing to overwrite an existing result artifact")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)
    if torch.get_num_threads() != THREADS:
        raise AssertionError("Torch thread cap is not two")

    source_sha = sha256_file(DATA_PATH)
    manifest = split_manifest()
    dev_ids = manifest.pop("_dev_ids")
    pool_ids = manifest.pop("_pool_ids")
    allowed_ids = manifest.pop("_allowed_ids")
    holdout_count = SAMPLE_N - DEV_N
    if len(allowed_ids) != manifest["training_pool_n"] + DEV_N:
        raise AssertionError("Unexpected data read-set size")
    rows_by_id = load_train_dev_rows(allowed_ids)
    train_items = [prepare(rows_by_id[qid]) for qid in pool_ids]
    dev_items = [prepare(rows_by_id[qid]) for qid in dev_ids]
    if len(dev_items) != DEV_N or len(train_items) != manifest["training_pool_n"]:
        raise AssertionError("Train/DEV materialization count mismatch")
    if {item["qid"] for item in train_items} & {item["qid"] for item in dev_items}:
        raise AssertionError("Train/DEV question overlap")

    tokenizer, dense_model, model_files = load_dense_encoder()
    all_items = train_items + dev_items
    flat_texts = []
    for item in all_items:
        item["vector_start"] = len(flat_texts)
        flat_texts.append(item["question"])
        flat_texts.extend(item["paragraphs"])
    vectors, dense_truncated, dense_input_count = encode_dense(flat_texts, tokenizer, dense_model)
    train_x, train_y = add_dense_order_and_features(train_items, vectors[: len(train_items) * 11])
    dev_vector_start = len(train_items) * 11
    dev_x, dev_y = add_dense_order_and_features(dev_items, vectors[dev_vector_start:])
    del dense_model, vectors, flat_texts

    logistic_probs = fit_logistic(train_x, train_y, dev_x)
    cross_model, cross_training_info = fit_cross_encoder(train_items, tokenizer)
    cross_probs, cross_dev_truncated = score_cross_encoder(dev_items, tokenizer, cross_model)
    del cross_model

    labels_by_type = {
        "overall": np.asarray([item["prefix_labels"] for item in dev_items], dtype=np.int8),
        "bridge": np.asarray([item["prefix_labels"] for item in dev_items if item["type"] == "bridge"], dtype=np.int8),
        "comparison": np.asarray([item["prefix_labels"] for item in dev_items if item["type"] == "comparison"], dtype=np.int8),
    }
    type_indices = {
        "overall": np.arange(len(dev_items)),
        "bridge": np.asarray([i for i, item in enumerate(dev_items) if item["type"] == "bridge"], dtype=np.int64),
        "comparison": np.asarray([i for i, item in enumerate(dev_items) if item["type"] == "comparison"], dtype=np.int64),
    }
    if any(len(ix) < 20 for key, ix in type_indices.items() if key != "overall"):
        raise AssertionError("Too few DEV questions in a comparison group")
    prob_matrices = {"logistic": logistic_probs, "cross_encoder": cross_probs}
    labels_matrix = labels_by_type["overall"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples_by_type = {
        group: group_bootstrap_indices(len(indices), rng)
        for group, indices in type_indices.items()
    }

    records = []
    for model_name in prob_matrices:
        for group in ("overall", "bridge", "comparison"):
            ix = type_indices[group]
            y = labels_matrix[ix]
            p = prob_matrices[model_name][ix]
            auc, auc_ci = auc_with_ci(y, p, samples_by_type[group])
            records.append(
                {
                    "record_type": "classifier_auc",
                    "model": model_name,
                    "group": group,
                    "questions": len(ix),
                    "prefix_examples": int(y.size),
                    "auroc": auc,
                    "auroc_ci95": auc_ci,
                    "bootstrap_resamples": BOOTSTRAPS,
                    "bootstrap_unit": "question",
                }
            )
            for threshold in FRONTIER_THRESHOLDS:
                stopped = stop_metrics_with_ci(y, p, threshold, samples_by_type[group])
                records.append(
                    {
                        "record_type": "stop_frontier",
                        "model": model_name,
                        "group": group,
                        "questions": len(ix),
                        "threshold": threshold,
                        **stopped,
                        "bootstrap_resamples": BOOTSTRAPS,
                        "bootstrap_unit": "question",
                    }
                )
    for group in ("overall", "bridge", "comparison"):
        y = labels_by_type[group]
        for k in range(1, 11):
            records.append(
                {
                    "record_type": "fixed_k_baseline",
                    "model": "dense_order",
                    "group": group,
                    "questions": len(y),
                    "k": k,
                    **fixed_k_metrics(y, k, samples_by_type[group]),
                    "bootstrap_resamples": BOOTSTRAPS,
                    "bootstrap_unit": "question",
                }
            )

    type_n = {group: len(indices) for group, indices in type_indices.items()}
    sanity = {
        "source_rows": manifest["source_rows"],
        "eligible_rows": manifest["eligible_rows"],
        "excluded_rows": manifest["excluded_rows"],
        "sample_n": SAMPLE_N,
        "dev_n": DEV_N,
        "holdout_n": holdout_count,
        "train_pool_n": len(train_items),
        "dev_by_type": {"bridge": type_n["bridge"], "comparison": type_n["comparison"]},
        "train_dev_disjoint": True,
        "holdout_full_rows_loaded": False,
        "holdout_used_for_training_or_metrics": False,
        "dev_exactly_200": len(dev_items) == 200,
        "all_selected_rows_have_two_gold_paragraphs": all(len(item["gold"]) == 2 for item in all_items),
        "all_dev_gold_labels_sufficient_at_k10": bool(np.all(labels_by_type["overall"][:, 9] == 1)),
        "cpu_only": True,
        "torch_threads": torch.get_num_threads(),
        "omp_threads": os.environ["OMP_NUM_THREADS"],
    }
    if not sanity["train_dev_disjoint"] or sanity["holdout_full_rows_loaded"] or sanity["holdout_used_for_training_or_metrics"]:
        raise AssertionError("Split leakage sanity check failed")
    if not sanity["all_dev_gold_labels_sufficient_at_k10"]:
        raise AssertionError("k=10 did not contain both gold paragraphs for every DEV query")

    runtime_seconds = time.monotonic() - RUN_STARTED
    metadata = {
        "record_type": "run_metadata",
        "project": "IRE Problem 5 B2_suff",
        "data_path": str(DATA_PATH),
        "source_url": DATA_URL,
        "data_sha256": source_sha,
        "data_bytes": DATA_PATH.stat().st_size,
        "sampling": {
            "eligible_source_order_rng_choice_seed": SAMPLE_SEED,
            "sample_n": SAMPLE_N,
            "sort": "(sha256(UTF-8 qid), qid); first 200 DEV, remaining 400 HOLDOUT",
            "manifest": {k: v for k, v in manifest.items() if not k.startswith("_")},
        },
        "training_pool_ids_sha256": manifest["training_pool_ids_sha256_source_order_newline_joined"],
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "path": str(MODEL_PATH),
            "max_length": MAX_LENGTH,
            "device": "cpu",
            "mps_used": False,
            "files": model_files,
        },
        "classifiers": {
            "logistic_features": FEATURE_NAMES,
            "cross_encoder": {
                **cross_training_info,
                "input": "[question, dense-ranked top-k passages joined with [SEP]]",
                "epochs": 1,
                "learning_rate": 2e-5,
                "batch_size": 16,
                "seed": CROSS_SEED,
                "subset_seed": TRAIN_SEED,
                "max_length": MAX_LENGTH,
            },
        },
        "truncation_counts": {
            "dense_query_and_passage_inputs": dense_truncated,
            "dense_query_and_passage_total": dense_input_count,
            "cross_encoder_training_prefixes": cross_training_info["training_truncated"],
            "cross_encoder_dev_prefixes": cross_dev_truncated,
        },
        "bootstrap": {"resamples": BOOTSTRAPS, "seed": BOOTSTRAP_SEED, "unit": "question", "ci": "percentile 95%"},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "numpy": np.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_threads": torch.get_num_threads(),
            "omp_threads": os.environ["OMP_NUM_THREADS"],
            "device": "cpu",
        },
        "sanity_checks": sanity,
        "runtime_seconds": runtime_seconds,
    }
    records.insert(0, metadata)
    jsonl = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records)
    metrics_sha = hashlib.sha256(jsonl.encode("utf-8")).hexdigest()
    runner_sha = sha256_file(Path(__file__))

    auc_lookup = {(r["model"], r["group"]): r for r in records if r["record_type"] == "classifier_auc"}
    frontier_lookup = {
        (r["model"], r["group"], r["threshold"]): r
        for r in records if r["record_type"] == "stop_frontier"
    }
    baseline_lookup = {
        (r["group"], r["k"]): r
        for r in records if r["record_type"] == "fixed_k_baseline"
    }
    lines = [
        "# IRE Problem 5 — B2_suff",
        "",
        "**Hypothesis.** A cheap classifier on dense-ranked evidence can recognize when both gold support paragraphs have arrived, trading stop precision against evidence size.",
        "",
        "**Setup.** HotpotQA distractor validation; eligible rows screened by the pilot protocol, 600 sampled with NumPy seed 20261031 and SHA256-qid ordered. Dense ranking uses cached all-MiniLM-L6-v2, CPU, max length 256, two threads. Logistic regression uses all training-pool prefixes; the cross-encoder is fine-tuned for one epoch on a seeded 512-query training-pool subset.",
        "",
        f"**Split.** {manifest['source_rows']} source rows; {manifest['eligible_rows']} eligible; {manifest['excluded_rows']} excluded; DEV {len(dev_items)} ({type_n['bridge']} bridge, {type_n['comparison']} comparison); HOLDOUT {holdout_count}; training pool {len(train_items)}.",
        f"Training-pool qid SHA256 (source-order IDs, newline joined): `{manifest['training_pool_ids_sha256_source_order_newline_joined']}`.",
        "Full HOLDOUT question rows were not materialized, trained on, or evaluated. To reproduce the pre-existing eligibility rule, the split scan read IDs/type plus context title/sentence-group and supporting-fact-title fields; question and answer fields were excluded.",
        "",
        "**Classifier AUROC** over the ten dense-order prefixes per DEV question; 95% percentile CI, 1,000 question-level bootstrap resamples:",
        "| Model | Group | Questions | AUROC [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for model_name in ("logistic", "cross_encoder"):
        for group in ("overall", "bridge", "comparison"):
            item = auc_lookup[(model_name, group)]
            lines.append(
                f"| {model_name} | {group} | {item['questions']} | {format_metric(item['auroc'], item['auroc_ci95'])} |"
            )
    lines += [
        "",
        "**Stop frontier.** Each cell is stop precision (95% CI) / mean passages used with k=10 fallback (95% CI). Thresholds were varied on DEV; these are tuned in-sample results.",
        "| Model | Group | t=.25 P / passages | t=.50 P / passages | t=.75 P / passages |",
        "|---|---|---:|---:|---:|",
    ]
    for model_name in ("logistic", "cross_encoder"):
        for group in ("overall", "bridge", "comparison"):
            cells = []
            for threshold in (0.25, 0.50, 0.75):
                item = frontier_lookup[(model_name, group, threshold)]
                p = format_metric(
                    item["precision_of_declared_sufficient_at_stop"],
                    item["precision_of_declared_sufficient_at_stop_ci95"],
                )
                k = format_metric(
                    item["avg_passages_used_with_k10_fallback"],
                    item["avg_passages_used_with_k10_fallback_ci95"],
                )
                cells.append(f"{p} / {k}")
            lines.append(f"| {model_name} | {group} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "**Fixed-k dense-order baselines.** Sufficiency rate with question-level bootstrap 95% CI:",
    ]
    for group in ("overall", "bridge", "comparison"):
        parts = []
        for k in (1, 2, 3, 4, 5, 10):
            item = baseline_lookup[(group, k)]
            parts.append(f"k={k}: {format_metric(item['sufficiency_rate'], item['sufficiency_rate_ci95'])}")
        lines.append(f"- {group}: " + "; ".join(parts))
    lines += [
        "",
        "**Sanity checks.** Train/DEV disjoint; exactly 200 DEV questions; two gold paragraphs per row; all DEV labels sufficient at k=10; CPU only; torch/OMP threads=2; no HOLDOUT training/evaluation; no downloads or remote calls.",
        f"**Limitations.** DEV is the only evaluation set and also selects the displayed thresholds, so frontier precision is optimistic and not an unbiased generalization estimate. Cross-encoder training used 512 of {len(train_items)} training-pool queries for one epoch; {cross_training_info['training_truncated']}/{cross_training_info['training_prefixes']} train and {cross_dev_truncated}/{DEV_N * 10} DEV prefixes exceeded 256 tokens and were truncated. The dense order is a fixed MiniLM baseline.",
        f"**Hashes.** `metrics.jsonl` SHA256 `{metrics_sha}`; runner SHA256 `{runner_sha}`; source parquet SHA256 `{source_sha}`. Result lines: pending.",
    ]
    lines[-1] = lines[-1].replace(". Result lines: pending.", ".")
    report = "\n".join(lines) + "\n"
    if len(lines) > 40:
        raise AssertionError(f"RESULT.md must be at most 40 lines; got {len(lines)}")
    if metrics_path.exists() or result_path.exists():
        raise FileExistsError("Refusing to overwrite an existing result artifact")
    with metrics_path.open("x", encoding="utf-8") as handle:
        handle.write(jsonl)
    with result_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    print(
        json.dumps(
            {
                "status": "complete",
                "metrics_path": str(metrics_path),
                "metrics_sha256": metrics_sha,
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "result_lines": len(lines),
                "records": len(records),
                "runtime_seconds": round(time.monotonic() - RUN_STARTED, 2),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
