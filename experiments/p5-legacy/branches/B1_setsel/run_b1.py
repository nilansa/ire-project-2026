#!/usr/bin/env python3
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path(__file__).resolve().parent
PILOT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot")
DATA_PATH = PILOT / "data" / "hotpot_validation.parquet"
PROTOCOL_PATH = PILOT / "protocol.md"
DATA_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 20261031
BOOTSTRAP_SEED = 0
BOOTSTRAP_N = 1000
SAMPLE_N = 600
DEV_N = 200
EXPECTED_ROWS = 7405
EXPECTED_ELIGIBLE = 7342
MAX_LENGTH = 256
MAX_MODEL_BYTES = 150_000_000
THREADS = 2
K_VALUES = (2, 3, 4)
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
]
STOPWORDS = set(
    "a an and are as at be been being but by can could did do does for from had has "
    "have he her hers him his how i if in into is it its may me might more most "
    "must my no nor not of on once only or other our ours out over shall she should "
    "so some such than that the their theirs them then there these they this those "
    "through to too under until up very was we were what when where which who whom "
    "why will with would you your yours".split()
)
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*\b")
TOKEN_RE = re.compile(r"[a-z0-9]+")
INTERROGATIVES = {"who", "what", "which", "where", "when", "why", "how"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_title(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def eligibility_reason(row: dict[str, Any]) -> str | None:
    try:
        if row["type"] not in {"bridge", "comparison"}:
            return "unsupported_type"
        titles = row["titles"]
        if len(titles) != 10:
            return "candidate_count_not_10"
        gold = {normalize_title(title) for title in row["gold_titles"]}
        if len(gold) != 2:
            return "unique_gold_titles_not_2"
        if sum(normalize_title(title) in gold for title in titles) != 2:
            return "gold_candidates_not_exactly_2"
        return None
    except (KeyError, TypeError):
        return "malformed_metadata"


def load_encoder() -> tuple[Any, dict[str, Any]]:
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(SEED)
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(BASE / ".hf-cache"),
            allow_patterns=MODEL_FILES,
            token=False,
        )
    )
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > MAX_MODEL_BYTES:
        raise RuntimeError(f"Selected model files exceed 150 MB: {total_bytes} bytes")
    if not (snapshot / "model.safetensors").is_file():
        raise RuntimeError("Pinned MiniLM snapshot has no safetensors weights")

    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(snapshot), local_files_only=True, use_safetensors=True
    ).to("cpu")
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("Encoder did not initialize on CPU")

    class Encoder:
        def __init__(self) -> None:
            self.tokenizer = tokenizer
            self.model = model
            self.torch = torch
            self.truncations: Counter[str] = Counter()
            self.input_counts: Counter[str] = Counter()

        def encode(self, texts: list[str], role: str, batch_size: int = 64) -> np.ndarray:
            self.input_counts[role] += len(texts)
            outputs = []
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                raw_ids = self.tokenizer(
                    chunk, add_special_tokens=True, truncation=False, padding=False, verbose=False
                )["input_ids"]
                self.truncations[role] += sum(len(ids) > MAX_LENGTH for ids in raw_ids)
                batch = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                    verbose=False,
                )
                with self.torch.inference_mode():
                    hidden = self.model(**batch).last_hidden_state
                    mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                    vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(vectors.cpu().numpy().astype(np.float32, copy=False))
            return np.concatenate(outputs, axis=0)

    model_info = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": "cpu",
        "pooling": "mean of non-padding last-layer hidden states; L2 normalized",
        "max_length": MAX_LENGTH,
        "thread_count": THREADS,
        "selected_model_file_bytes": total_bytes,
        "files": [
            {
                "path": str(path.relative_to(snapshot)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    return Encoder(), model_info


def lexical_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def bm25_order(question: str, paragraphs: list[str], k1: float, b: float) -> list[int]:
    documents = [lexical_tokens(text) for text in paragraphs]
    query_terms = lexical_tokens(question)
    lengths = [len(tokens) for tokens in documents]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequency = Counter(term for tokens in documents for term in set(tokens))
    scores = np.zeros(len(documents), dtype=np.float64)
    for term in query_terms:
        df = document_frequency.get(term, 0)
        if not df:
            continue
        idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
        for index, tokens in enumerate(documents):
            tf = tokens.count(term)
            if tf:
                denom = tf + k1 * (1.0 - b + b * lengths[index] / max(average_length, 1e-12))
                scores[index] += idf * tf * (k1 + 1.0) / denom
    return np.argsort(-scores, kind="stable").astype(int).tolist()


def mmr_order(query: np.ndarray, passages: np.ndarray, lam: float) -> list[int]:
    relevance = passages @ query
    redundancy = np.zeros(len(passages), dtype=np.float32)
    selected: list[int] = []
    while len(selected) < len(passages):
        scores = relevance.copy() if not selected else lam * relevance - (1 - lam) * redundancy
        if selected:
            scores[selected] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        similarities = passages @ passages[index]
        redundancy = similarities if len(selected) == 1 else np.maximum(redundancy, similarities)
    return selected


def query_entities(question: str) -> list[str]:
    phrases = []
    for match in ENTITY_RE.finditer(question):
        phrase = " ".join(match.group(0).split())
        if match.start() == 0 and phrase.split()[0].casefold() in INTERROGATIVES:
            continue
        if phrase.casefold() not in {item.casefold() for item in phrases}:
            phrases.append(phrase)
    return phrases


def entity_in_text(phrase: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", text.casefold()) is not None


def coverage_order(question: str, paragraphs: list[str], entity_weight: int) -> list[int]:
    terms = sorted({token for token in lexical_tokens(question) if token not in STOPWORDS})
    entities = query_entities(question) if entity_weight else []
    features = [("term", term, 1) for term in terms]
    features += [("entity", phrase, entity_weight) for phrase in entities]
    passage_terms = [set(lexical_tokens(text)) for text in paragraphs]
    passage_entity_hits = [
        {phrase for phrase in entities if entity_in_text(phrase, text)} for text in paragraphs
    ]
    covered: set[tuple[str, str]] = set()
    selected: list[int] = []
    while len(selected) < len(paragraphs):
        best_index, best_score = -1, -1
        for index in range(len(paragraphs)):
            if index in selected:
                continue
            score = 0
            for kind, value, weight in features:
                feature = (kind, value)
                if feature in covered:
                    continue
                hit = value in passage_terms[index] if kind == "term" else value in passage_entity_hits[index]
                if hit:
                    score += weight
            if score > best_score:
                best_index, best_score = index, score
        selected.append(best_index)
        for kind, value, _ in features:
            if kind == "term" and value in passage_terms[best_index]:
                covered.add((kind, value))
            elif kind == "entity" and value in passage_entity_hits[best_index]:
                covered.add((kind, value))
    return selected


def conditional_orders(
    items: list[dict[str, Any]],
    passage_vectors: np.ndarray,
    query_vectors: np.ndarray,
    encoder: Any,
    scope: str,
) -> list[list[int]]:
    orders = [
        np.argsort(-(passage_vectors[i] @ query_vectors[i]), kind="stable").astype(int).tolist()[:1]
        for i in range(len(items))
    ]
    while True:
        active = [i for i, order in enumerate(orders) if not items[i]["gold_indices"].issubset(order)]
        if not active:
            return orders
        conditioned = []
        for i in active:
            selected_text = [
                items[i]["titles"][j] if scope == "title" else items[i]["paragraphs"][j]
                for j in orders[i]
            ]
            conditioned.append(items[i]["question"] + " [SEP] " + " [SEP] ".join(selected_text))
        vectors = encoder.encode(conditioned, role=f"conditional_{scope}")
        for i, vector in zip(active, vectors):
            scores = passage_vectors[i] @ vector
            scores[orders[i]] = -np.inf
            orders[i].append(int(np.argmax(scores)))
        if any(len(order) > 10 for order in orders):
            raise RuntimeError("Conditional selection exceeded the ten-candidate pool")


def prepare_row(row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row["id"])
    titles = [str(title).strip() for title in row["titles"]]
    sentence_groups = row["sentences"]
    if len(titles) != 10 or len(sentence_groups) != 10:
        raise ValueError(f"Candidate structure mismatch for DEV qid {qid}")
    paragraphs = [
        (title + " " + " ".join(str(sentence).strip() for sentence in sentences)).strip()
        for title, sentences in zip(titles, sentence_groups)
    ]
    gold_titles = {normalize_title(title) for title in row["gold_titles"]}
    gold_indices = {i for i, title in enumerate(titles) if normalize_title(title) in gold_titles}
    if len(gold_titles) != 2 or len(gold_indices) != 2:
        raise ValueError(f"Gold support mapping mismatch for DEV qid {qid}")
    return {
        "qid": qid,
        "type": str(row["type"]),
        "question": str(row["question"]),
        "titles": titles,
        "paragraphs": paragraphs,
        "gold_indices": gold_indices,
    }


def full_support_size(order: list[int], gold: set[int]) -> int:
    ranks = {candidate: rank for rank, candidate in enumerate(order, start=1)}
    if not gold.issubset(ranks):
        raise ValueError("Selector failed to rank both gold paragraphs")
    return max(ranks[index] for index in gold)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
    means = np.mean(values[draws], axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def package_versions() -> dict[str, str]:
    names = ["numpy", "torch", "transformers", "tokenizers", "huggingface-hub", "pyarrow"]
    return {name: importlib.metadata.version(name) for name in names}


def cpu_description() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or platform.machine()


def main() -> None:
    sys.path.insert(0, str(BASE / ".vendor"))
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    output_metrics = BASE / "metrics.jsonl"
    output_result = BASE / "RESULT.md"
    for path in (output_metrics, output_result):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    if not DATA_PATH.is_file() or not PROTOCOL_PATH.is_file():
        raise FileNotFoundError("Read-only pilot dataset or protocol is missing")

    random.seed(SEED)
    np.random.seed(SEED)
    data_sha = sha256_file(DATA_PATH)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    parquet_file = pq.ParquetFile(DATA_PATH)
    source_rows_n = parquet_file.metadata.num_rows
    if source_rows_n != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} source rows, found {source_rows_n}")

    metadata_started = time.perf_counter()
    metadata_dataset = ds.dataset(DATA_PATH, format="parquet")
    metadata = metadata_dataset.scanner(
        columns={
            "id": ds.field("id"),
            "type": ds.field("type"),
            "titles": ds.field("context", "title"),
            "gold_titles": ds.field("supporting_facts", "title"),
        },
        batch_size=128,
        use_threads=False,
    ).to_table().to_pylist()
    excluded: Counter[str] = Counter()
    eligible: list[dict[str, str]] = []
    for row in metadata:
        reason = eligibility_reason(row)
        if reason is None:
            eligible.append({"id": str(row["id"]), "type": str(row["type"])})
        else:
            excluded[reason] += 1
    metadata_seconds = time.perf_counter() - metadata_started
    expected_excluded = {"candidate_count_not_10": 60, "gold_candidates_not_exactly_2": 3}
    if len(eligible) != EXPECTED_ELIGIBLE or dict(excluded) != expected_excluded:
        raise ValueError(
            f"Eligibility audit differs from registered protocol: eligible={len(eligible)}, excluded={dict(excluded)}"
        )
    eligible_count = len(eligible)
    if len({row["id"] for row in eligible}) != len(eligible):
        raise ValueError("Eligible source contains duplicate question IDs")

    sample_rng = np.random.default_rng(SEED)
    sampled_indices = sample_rng.choice(len(eligible), size=SAMPLE_N, replace=False)
    sampled = [eligible[int(index)] for index in sampled_indices]
    if len({row["id"] for row in sampled}) != SAMPLE_N:
        raise ValueError("Seeded sample contains duplicate question IDs")
    sampled.sort(
        key=lambda row: (hashlib.sha256(row["id"].encode("utf-8")).hexdigest(), row["id"])
    )
    dev_meta = sampled[:DEV_N]
    dev_ids = [row["id"] for row in dev_meta]
    dev_id_sha = hashlib.sha256("\n".join(dev_ids).encode("utf-8")).hexdigest()
    dev_type_counts = Counter(row["type"] for row in dev_meta)
    del sampled, eligible, metadata, dev_meta

    # Predicate pushdown returns DEV rows only; sampled HOLDOUT IDs/content are not retained.
    dataset = ds.dataset(DATA_PATH, format="parquet")
    dev_table = dataset.scanner(
        columns={
            "id": ds.field("id"),
            "question": ds.field("question"),
            "type": ds.field("type"),
            "titles": ds.field("context", "title"),
            "sentences": ds.field("context", "sentences"),
            "gold_titles": ds.field("supporting_facts", "title"),
        },
        filter=ds.field("id").isin(dev_ids),
        batch_size=32,
        use_threads=False,
    ).to_table()
    raw_dev_rows = dev_table.to_pylist()
    if len(raw_dev_rows) != DEV_N or {str(row["id"]) for row in raw_dev_rows} != set(dev_ids):
        raise ValueError("Filtered DEV row set does not match the deterministic DEV IDs")
    rows_by_id = {str(row["id"]): row for row in raw_dev_rows}
    items = [prepare_row(rows_by_id[qid]) for qid in dev_ids]
    del rows_by_id, raw_dev_rows, dev_table, dataset

    encoder, model_info = load_encoder()
    index_started = time.perf_counter()
    flat_paragraphs = [text for item in items for text in item["paragraphs"]]
    passage_vectors = encoder.encode(flat_paragraphs, role="candidate_paragraphs").reshape(
        DEV_N, 10, -1
    )
    query_vectors = encoder.encode([item["question"] for item in items], role="questions")
    index_seconds = time.perf_counter() - index_started

    configs: list[dict[str, Any]] = []
    for k1 in (0.8, 1.2, 1.6):
        for b in (0.25, 0.75):
            configs.append({"method": "bm25", "config_id": f"k1={k1:.1f},b={b:.2f}", "k1": k1, "b": b})
    configs.append({"method": "dense", "config_id": "mean_pool_minilm", "model": MODEL_ID})
    for lam in (0.25, 0.5, 0.75):
        configs.append({"method": "mmr", "config_id": f"lambda={lam:.2f}", "lambda": lam})
    for weight in (0, 1, 2):
        configs.append({
            "method": "coverage", "config_id": f"entity_weight={weight}",
            "entity_weight": weight,
        })
    for scope in ("title", "full_paragraph"):
        configs.append({"method": "conditional", "config_id": f"context={scope}", "scope": scope})

    conditional_started = time.perf_counter()
    conditional = {
        scope: conditional_orders(items, passage_vectors, query_vectors, encoder, scope)
        for scope in ("title", "full_paragraph")
    }
    conditional_seconds = time.perf_counter() - conditional_started

    matrices: dict[str, np.ndarray] = {}
    for config in configs:
        method = config["method"]
        rows = []
        for i, item in enumerate(items):
            pvec = passage_vectors[i]
            qvec = query_vectors[i]
            if method == "bm25":
                order = bm25_order(item["question"], item["paragraphs"], config["k1"], config["b"])
            elif method == "dense":
                order = np.argsort(-(pvec @ qvec), kind="stable").astype(int).tolist()
            elif method == "mmr":
                order = mmr_order(qvec, pvec, config["lambda"])
            elif method == "coverage":
                order = coverage_order(item["question"], item["paragraphs"], config["entity_weight"])
            else:
                order = conditional[config["scope"]][i]
            size = full_support_size(order, item["gold_indices"])
            rows.append([float(size <= k) for k in K_VALUES] + [float(size)])
        matrices[config["config_id"]] = np.asarray(rows, dtype=np.float64)

    config_by_id = {config["config_id"]: config for config in configs}
    config_order = [config["config_id"] for config in configs]
    def objective(config_id: str, indices: np.ndarray | None = None) -> float:
        values = matrices[config_id] if indices is None else matrices[config_id][indices]
        return float(np.mean(values[:, :3]))

    def choose(ids: list[str], indices: np.ndarray | None = None) -> str:
        return max(ids, key=lambda cid: objective(cid, indices))

    family_ids = {
        method: [c["config_id"] for c in configs if c["method"] == method]
        for method in ("bm25", "dense", "mmr", "coverage", "conditional")
    }
    independent_ids = family_ids["bm25"] + family_ids["dense"]
    set_aware_ids = family_ids["mmr"] + family_ids["coverage"] + family_ids["conditional"]
    best = {method: choose(ids) for method, ids in family_ids.items()}
    best["independent"] = choose(independent_ids)
    best["set_aware_overall"] = choose(set_aware_ids)

    group_indices = {
        "overall": np.arange(DEV_N),
        "bridge": np.asarray([i for i, item in enumerate(items) if item["type"] == "bridge"], dtype=int),
        "comparison": np.asarray([i for i, item in enumerate(items) if item["type"] == "comparison"], dtype=int),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_records: list[dict[str, Any]] = []
    for config in configs:
        cid = config["config_id"]
        chosen_for_family = best.get(config["method"]) == cid
        for group, indices in group_indices.items():
            values = matrices[cid][indices]
            metric_values = {}
            for col, name in enumerate(("FS@2", "FS@3", "FS@4", "avg_set_size_to_full_support")):
                metric_values[name] = {
                    "estimate": float(np.mean(values[:, col])) if len(values) else None,
                    "ci95_percentile": bootstrap_ci(values[:, col], rng) if len(values) else None,
                }
            metric_records.append({
                "record_type": "metric",
                "method": config["method"],
                "config_id": cid,
                "config": config,
                "best_on_dev_for_method": bool(chosen_for_family),
                "group": group,
                "query_n": int(len(indices)),
                "metrics": metric_values,
                "bootstrap": {"resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "question"},
            })

    bootstrap_indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, DEV_N, size=(BOOTSTRAP_N, DEV_N)
    )
    def contrast_record(label: str, aware_ids: list[str]) -> dict[str, Any]:
        aware_winner = choose(aware_ids)
        independent_winner = choose(independent_ids)
        point = matrices[aware_winner][:, :3].mean(axis=0) - matrices[independent_winner][:, :3].mean(axis=0)
        draws = np.empty((BOOTSTRAP_N, 4), dtype=np.float64)
        for bidx, sample in enumerate(bootstrap_indices):
            a = choose(aware_ids, sample)
            base = choose(independent_ids, sample)
            delta = matrices[a][sample, :3].mean(axis=0) - matrices[base][sample, :3].mean(axis=0)
            draws[bidx, :3] = delta
            draws[bidx, 3] = delta.mean()
        names = ["FS@2", "FS@3", "FS@4", "mean_FS@2_4"]
        return {
            "record_type": "paired_contrast",
            "comparison": label,
            "set_aware_best_config": aware_winner,
            "independent_best_config": independent_winner,
            "selection_criterion": "maximize equal-weight mean FS@2, FS@3, FS@4 on the resampled DEV questions",
            "bootstrap_reselects_configs": True,
            "metrics": {
                name: {
                    "difference_estimate": float(point[i]) if i < 3 else float(point.mean()),
                    "ci95_percentile": [float(x) for x in np.quantile(draws[:, i], [0.025, 0.975])],
                }
                for i, name in enumerate(names)
            },
            "bootstrap": {"resamples": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "unit": "question"},
        }

    contrast_records = [
        contrast_record(method, family_ids[method]) for method in ("mmr", "coverage", "conditional")
    ]
    contrast_records.append(contrast_record("set_aware_overall", set_aware_ids))

    sanity = {
        "source_row_count_matches_protocol": source_rows_n == EXPECTED_ROWS,
        "eligibility_count_matches_protocol": eligible_count == EXPECTED_ELIGIBLE,
        "exclusions_match_protocol": expected_excluded == dict(excluded),
        "dev_count": len(items),
        "dev_bridge_count": int(group_indices["bridge"].size),
        "dev_comparison_count": int(group_indices["comparison"].size),
        "every_dev_question_has_ten_candidates_and_two_gold": True,
        "all_selectors_recover_full_gold_support_by_pool_end": True,
        "holdout_content_loaded": False,
        "holdout_evaluated": False,
        "training_performed": False,
        "mps_used": False,
        "omp_and_torch_threads": THREADS,
    }
    if sanity["dev_count"] != DEV_N or sanity["dev_bridge_count"] + sanity["dev_comparison_count"] != DEV_N:
        raise AssertionError("DEV split counts failed")

    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_description(),
        "logical_cpu_count": os.cpu_count(),
        "packages": package_versions(),
        "thread_environment": {name: os.environ[name] for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
        )},
    }
    run_metadata = {
        "record_type": "run_metadata",
        "task": "IRE Problem 5 branch B1_setsel",
        "hypothesis": "Set-aware selectors improve recovery of the complete two-paragraph gold support set over independent top-k at small k.",
        "data": {
            "source_url": DATA_URL,
            "local_read_only_path": str(DATA_PATH),
            "sha256": data_sha,
            "bytes": DATA_PATH.stat().st_size,
            "source_rows": source_rows_n,
            "eligible_rows": EXPECTED_ELIGIBLE,
            "excluded_rows_by_reason": dict(excluded),
            "protocol_sha256": protocol_sha,
            "sampling": "numpy.default_rng(20261031).choice(eligible source-order rows, 600, replace=False); sort by (sha256(UTF-8 qid), qid); first 200 DEV",
            "sample_n": SAMPLE_N,
            "dev_n": DEV_N,
            "dev_qid_hash_order_sha256": dev_id_sha,
            "dev_qid_hash_format": "SHA256 of newline-joined DEV IDs in hash-sorted order",
            "dev_type_counts": dict(dev_type_counts),
            "holdout_content_loaded": False,
            "holdout_evaluated": False,
            "training_pool_used": False,
            "training_pool_ids_sha256": None,
        },
        "model": model_info,
        "runtime": runtime,
        "seeds": {"sampling": SEED, "global_numpy": SEED, "python_random": SEED, "torch": SEED, "bootstrap": BOOTSTRAP_SEED},
        "bootstrap": {"resamples": BOOTSTRAP_N, "confidence": 0.95, "unit": "question", "interval": "percentile"},
        "selection_grids": configs,
        "selection_notes": {
            "bm25": "k1 in {0.8,1.2,1.6}, b in {0.25,0.75}; positive Robertson IDF; stable ties by candidate order",
            "dense": "mean-pooled MiniLM top-k; stable ties by candidate order",
            "mmr": "lambda in {0.25,0.5,0.75}; lambda*query cosine - (1-lambda)*max selected-passage cosine",
            "coverage": "greedy unique question content-term coverage plus proper-name phrase features weighted 0,1,2; capitalization heuristic",
            "conditional": "greedy dense re-scoring of remaining paragraphs using question plus all selected title/full paragraph text; truncate at 256 tokens",
            "best_on_dev": "equal-weight mean of FS@2, FS@3, FS@4; ties use first registered grid configuration",
        },
        "sanity_checks": sanity,
        "timing_seconds": {
            "metadata_split_scan": metadata_seconds,
            "candidate_and_question_encoding": index_seconds,
            "conditional_encoding_and_selection": conditional_seconds,
        },
        "model_input_counts": dict(encoder.input_counts),
        "model_truncations": dict(encoder.truncations),
        "code_sha256": sha256_file(Path(__file__)),
    }

    records = [run_metadata, *metric_records, *[
        {"record_type": "best_on_dev", "method": method, "config_id": cid,
         "objective_mean_FS_at_2_3_4": objective(cid)}
        for method, cid in best.items()
    ], *contrast_records]
    metrics_text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    metrics_sha = hashlib.sha256(metrics_text.encode("utf-8")).hexdigest()

    def show_metric(cid: str, group: str) -> str:
        record = next(r for r in metric_records if r["config_id"] == cid and r["group"] == group)
        m = record["metrics"]
        def fmt(name: str) -> str:
            item = m[name]
            lo, hi = item["ci95_percentile"]
            return f"{item['estimate']:.4f} [{lo:.4f},{hi:.4f}]"
        return (
            f"{config_by_id[cid]['method']} {cid}: FS@2 {fmt('FS@2')}; FS@3 {fmt('FS@3')}; "
            f"FS@4 {fmt('FS@4')}; set-size {fmt('avg_set_size_to_full_support')}"
        )

    overall_winners = [
        ("independent", best["independent"]), ("MMR", best["mmr"]),
        ("coverage", best["coverage"]), ("conditional", best["conditional"]),
        ("best set-aware", best["set_aware_overall"]),
    ]
    markdown = [
        "# IRE Problem 5 — B1_setsel",
        "",
        "Hypothesis: set-aware selectors recover the complete two-paragraph gold support more often than independent top-k at small k; FS means both gold paragraphs retrieved.",
        f"Setup: HotpotQA distractor dev; {EXPECTED_ROWS} source rows, {EXPECTED_ELIGIBLE} protocol-eligible; seeded sample 600, hash-sorted first {DEV_N} used as DEV. DEV IDs SHA256: `{dev_id_sha}`.",
        f"Data SHA256: `{data_sha}`. Split construction scanned only IDs/eligibility metadata; only filtered DEV rows were returned to evaluation. HOLDOUT evaluation and training were unused.",
        f"Encoder: {MODEL_ID}@`{MODEL_REVISION}`, selected files {model_info['selected_model_file_bytes']:,} bytes; CPU, OMP/PyTorch threads 2, max length {MAX_LENGTH}; no MPS.",
        "Grid: BM25 k1={0.8,1.2,1.6}, b={0.25,0.75}; dense MiniLM; MMR lambda={0.25,0.5,0.75}; coverage entity weight={0,1,2}; conditional context={title,full paragraph}.",
        "Tuning: best config maximizes equal-weight mean FS@2/@3/@4 on all 200 DEV questions. Reported CIs are question bootstrap percentile 95%, 1,000 resamples, seed 0; tuned results are in-sample.",
        "",
        "## Best-on-DEV configurations (overall DEV)",
        *[f"- {label}: {show_metric(cid, 'overall')}" for label, cid in overall_winners],
        "",
        "## Bridge DEV (config selected on overall DEV)",
        *[f"- {label}: {show_metric(cid, 'bridge')}" for label, cid in overall_winners],
        "",
        "## Comparison DEV (config selected on overall DEV)",
        *[f"- {label}: {show_metric(cid, 'comparison')}" for label, cid in overall_winners],
        "",
        "## Paired tuned contrast vs best independent top-k",
    ]
    for record in contrast_records:
        metric = record["metrics"]["mean_FS@2_4"]
        lo, hi = metric["ci95_percentile"]
        markdown.append(
            f"- {record['comparison']}: {metric['difference_estimate']:+.4f} "
            f"[{lo:+.4f},{hi:+.4f}] (bootstrap reselects configs)."
        )
    markdown.extend([
        f"Sanity: DEV n={DEV_N} (bridge {dev_type_counts['bridge']}, comparison {dev_type_counts['comparison']}); all DEV rows had 10 candidates and exactly 2 gold paragraphs; all selectors reached full support by pool end.",
        "Limitations: small in-sample DEV tuning; lexical capitalization heuristic; MiniLM cosine and concatenated-context re-encoding are approximations, not trained DISCo/SetR systems. No holdout claim.",
        f"SHA256 metrics.jsonl: `{metrics_sha}`. RESULT.md SHA256 is reported in the task completion message.",
    ])
    result_text = "\n".join(markdown) + "\n"
    if len(markdown) > 40:
        raise AssertionError(f"RESULT.md exceeds 40 lines: {len(markdown)}")

    with output_metrics.open("x", encoding="utf-8") as handle:
        handle.write(metrics_text)
    with output_result.open("x", encoding="utf-8") as handle:
        handle.write(result_text)
    print(json.dumps({
        "metrics": str(output_metrics),
        "metrics_sha256": metrics_sha,
        "result": str(output_result),
        "result_lines": len(markdown),
        "dev_type_counts": dict(dev_type_counts),
        "best_on_dev": best,
        "timing_seconds": run_metadata["timing_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
