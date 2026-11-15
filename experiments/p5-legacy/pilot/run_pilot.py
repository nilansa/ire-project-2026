from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "hotpot_validation.parquet"
DATA_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SEED = 20261031
SAMPLE_N = 600
DEV_N = 200
HOLDOUT_N = 400
MAX_K = 4
MAX_MODEL_BYTES = 150_000_000
MODEL_MAX_LENGTH = 256
THREADS = 8
METHODS = [
    "bm25",
    "dense",
    "mmr",
    "disco_approx_cost_norm",
    "disco_approx_no_cost_norm",
    "mdr_style",
    "random",
    "gold_oracle",
]
SINGLE_SHOT = [
    "bm25",
    "dense",
    "mmr",
    "disco_approx_cost_norm",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_title(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def score_selection(selected: list[int], gold_indices: set[int], paragraphs: list[str]) -> dict[str, Any]:
    selected_set = set(selected)
    recall = len(selected_set & gold_indices) / len(gold_indices)
    return {
        "sp_paragraph_recall": float(recall),
        "full_support": bool(gold_indices <= selected_set),
        "evidence_tokens": int(sum(len(paragraphs[i].split()) for i in selected)),
    }


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def bm25_order(question: str, paragraphs: list[str]) -> list[int]:
    documents = [lexical_tokens(text) for text in paragraphs]
    query_terms = lexical_tokens(question)
    lengths = [len(tokens) for tokens in documents]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequency = Counter(term for tokens in documents for term in set(tokens))
    scores = np.zeros(len(documents), dtype=np.float64)
    k1, b = 1.2, 0.75
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


def mmr_order(
    question_vector: np.ndarray,
    paragraph_vectors: np.ndarray,
    max_k: int = MAX_K,
    base_elapsed: float = 0.0,
) -> tuple[list[int], list[float]]:
    start = time.perf_counter()
    relevance = paragraph_vectors @ question_vector
    redundancy = np.zeros(len(paragraph_vectors), dtype=np.float32)
    selected: list[int] = []
    timings: list[float] = []
    for _ in range(max_k):
        scores = 0.5 * relevance - 0.5 * redundancy if selected else relevance.copy()
        if selected:
            scores[selected] = -np.inf
        index = int(np.argmax(scores))
        selected.append(index)
        sims = paragraph_vectors @ paragraph_vectors[index]
        redundancy = sims if len(selected) == 1 else np.maximum(redundancy, sims)
        timings.append(base_elapsed + time.perf_counter() - start)
    return selected, timings


def facility_coverages(
    query_tokens: np.ndarray, paragraph_tokens: list[np.ndarray]
) -> tuple[np.ndarray, list[int]]:
    rows = []
    costs = []
    for tokens in paragraph_tokens:
        if len(tokens):
            rows.append(np.max(query_tokens @ tokens.T, axis=1))
            costs.append(len(tokens))
        else:
            rows.append(np.zeros(len(query_tokens), dtype=np.float32))
            costs.append(1)
    return np.stack(rows), costs


def greedy_facility_from_coverages(
    coverages: np.ndarray,
    token_costs: list[int],
    cost_normalized: bool,
    max_k: int = MAX_K,
    base_elapsed: float = 0.0,
) -> tuple[list[int], list[float]]:
    start = time.perf_counter()
    selected: list[int] = []
    timings: list[float] = []
    current_max: np.ndarray | None = None
    for _ in range(max_k):
        best_index = -1
        best_score = -np.inf
        best_coverage = None
        for index, coverage in enumerate(coverages):
            if index in selected:
                continue
            if current_max is None:
                new_max = coverage
                gain = float(np.sum(new_max))
            else:
                new_max = np.maximum(current_max, coverage)
                gain = float(np.sum(new_max - current_max))
            score = gain / max(1, token_costs[index]) if cost_normalized else gain
            if score > best_score:
                best_index, best_score, best_coverage = index, score, new_max
        selected.append(best_index)
        current_max = np.array(best_coverage, copy=True)
        timings.append(base_elapsed + time.perf_counter() - start)
    return selected, timings


def greedy_facility_order(
    query_tokens: np.ndarray,
    paragraph_tokens: list[np.ndarray],
    cost_normalized: bool,
    max_k: int = MAX_K,
    started_at: float | None = None,
) -> tuple[list[int], list[float]]:
    start = time.perf_counter() if started_at is None else started_at
    coverages, costs = facility_coverages(query_tokens, paragraph_tokens)
    elapsed_before_selection = time.perf_counter() - start
    return greedy_facility_from_coverages(
        coverages,
        costs,
        cost_normalized,
        max_k,
        base_elapsed=elapsed_before_selection,
    )

def extract_query_entities(question: str) -> list[str]:
    pattern = re.compile(r"\b[A-Z][A-Za-z0-9'’.-]*(?:\s+[A-Z][A-Za-z0-9'’.-]*)*\b")
    interrogatives = {"who", "what", "which", "where", "when", "why", "how"}
    phrases = []
    for match in pattern.finditer(question):
        phrase = " ".join(match.group(0).split())
        first = phrase.split()[0].casefold()
        if match.start() == 0 and first in interrogatives:
            continue
        if phrase and phrase.casefold() not in {p.casefold() for p in phrases}:
            phrases.append(phrase)
    return phrases


def entity_in_text(phrase: str, text: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", text.casefold()) is not None


def load_encoder() -> tuple[Any, Any, dict[str, Any]]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    cache_dir = BASE / ".hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    import torch
    from transformers import AutoModel, AutoTokenizer

    allowed_files = [
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
    local_snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(cache_dir),
            allow_patterns=allowed_files,
        )
    )
    model_files = sorted(path for path in local_snapshot.rglob("*") if path.is_file())
    total_bytes = sum(path.stat().st_size for path in model_files)
    if total_bytes > MAX_MODEL_BYTES:
        raise RuntimeError(f"Selected model files exceed 150 MB: {total_bytes} bytes")
    weight_files = [path for path in model_files if path.name == "model.safetensors"]
    if len(weight_files) != 1 or weight_files[0].stat().st_size > MAX_MODEL_BYTES:
        raise RuntimeError("Expected one sentence encoder weight file no larger than 150 MB")

    torch.set_num_threads(THREADS)
    tokenizer = AutoTokenizer.from_pretrained(str(local_snapshot), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(local_snapshot), local_files_only=True, use_safetensors=True
    ).to("cpu")
    model.eval()
    model_info = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_length": MODEL_MAX_LENGTH,
        "device": "cpu",
        "thread_count": THREADS,
        "selected_model_file_bytes": total_bytes,
        "files": [
            {
                "path": str(path.relative_to(BASE)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in model_files
        ],
    }
    return Encoder(tokenizer, model, torch), local_snapshot, model_info


class Encoder:
    def __init__(self, tokenizer: Any, model: Any, torch: Any):
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch
        self.truncations: Counter[str] = Counter()
        self.input_counts: Counter[str] = Counter()

    def encode(
        self, texts: list[str], role: str, need_tokens: bool = False
    ) -> tuple[np.ndarray, list[np.ndarray], int]:
        raw = self.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False, verbose=False
        )["input_ids"]
        truncated = sum(len(tokens) > MODEL_MAX_LENGTH for tokens in raw)
        self.truncations[role] += truncated
        self.input_counts[role] += len(texts)
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MODEL_MAX_LENGTH,
            return_tensors="pt",
            return_special_tokens_mask=True,
            verbose=False,
        )
        special_mask = batch.pop("special_tokens_mask")
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vector_array = vectors.cpu().numpy().astype(np.float32, copy=False)
            token_arrays: list[np.ndarray] = []
            if need_tokens:
                valid = batch["attention_mask"].bool() & ~special_mask.bool()
                for index in range(hidden.shape[0]):
                    token_vectors = hidden[index][valid[index]]
                    token_vectors = self.torch.nn.functional.normalize(token_vectors, p=2, dim=1)
                    values = token_vectors.cpu().numpy().astype(np.float32, copy=False)
                    if len(values) == 0:
                        values = np.zeros((1, hidden.shape[-1]), dtype=np.float32)
                    token_arrays.append(values)
        return vector_array, token_arrays, truncated


def build_methods(
    encoder: Encoder,
    qid: str,
    question: str,
    paragraphs: list[str],
    paragraph_vectors: np.ndarray,
    paragraph_tokens: list[np.ndarray],
    gold_indices: set[int],
) -> tuple[dict[str, tuple[list[int], list[float]]], list[int]]:
    results: dict[str, tuple[list[int], list[float]]] = {}

    query_started = time.perf_counter()
    query_vectors, query_token_arrays, _ = encoder.encode(
        [question], role="query_shared", need_tokens=True
    )
    query_encode_seconds = time.perf_counter() - query_started
    query_vector = query_vectors[0]
    query_token_vectors = query_token_arrays[0]

    start = time.perf_counter()
    dense_scores = paragraph_vectors @ query_vector
    dense_order = np.argsort(-dense_scores, kind="stable").astype(int).tolist()
    dense_elapsed = query_encode_seconds + time.perf_counter() - start
    results["dense"] = (dense_order[:MAX_K], [dense_elapsed] * MAX_K)

    results["mmr"] = mmr_order(
        query_vector, paragraph_vectors, MAX_K, base_elapsed=query_encode_seconds
    )

    coverage_started = time.perf_counter()
    coverages, token_costs = facility_coverages(query_token_vectors, paragraph_tokens)
    coverage_seconds = time.perf_counter() - coverage_started
    for name, normalize_cost in (
        ("disco_approx_cost_norm", True),
        ("disco_approx_no_cost_norm", False),
    ):
        results[name] = greedy_facility_from_coverages(
            coverages,
            token_costs,
            normalize_cost,
            MAX_K,
            base_elapsed=query_encode_seconds + coverage_seconds,
        )

    start = time.perf_counter()
    selected: list[int] = []
    mdr_times: list[float] = []
    for step in range(MAX_K):
        if step == 0:
            mdr_query_vector = query_vector
        else:
            mdr_query = question + " " + " ".join(paragraphs[index] for index in selected)
            mdr_query_vector = encoder.encode(
                [mdr_query], role="query_mdr_conditioned"
            )[0][0]
        scores = paragraph_vectors @ mdr_query_vector
        if selected:
            scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
        mdr_times.append(query_encode_seconds + time.perf_counter() - start)
    results["mdr_style"] = (selected, mdr_times)

    seed_material = hashlib.sha256(f"{SEED}\0{qid}".encode("utf-8")).digest()[:8]
    query_seed = int.from_bytes(seed_material, "big")
    start = time.perf_counter()
    random_order = np.random.default_rng(query_seed).permutation(len(paragraphs)).astype(int).tolist()
    random_times = [time.perf_counter() - start for _ in range(MAX_K)]
    results["random"] = (random_order[:MAX_K], random_times)

    start = time.perf_counter()
    oracle_order = sorted(gold_indices) + [
        index for index in range(len(paragraphs)) if index not in gold_indices
    ]
    oracle_times = [time.perf_counter() - start for _ in range(MAX_K)]
    results["gold_oracle"] = (oracle_order[:MAX_K], oracle_times)

    start = time.perf_counter()
    bm25 = bm25_order(question, paragraphs)
    bm25_times = [time.perf_counter() - start for _ in range(MAX_K)]
    results["bm25"] = (bm25[:MAX_K], bm25_times)
    return results, dense_order

def eligibility_reason(row: dict[str, Any]) -> str | None:
    try:
        if row["type"] not in {"bridge", "comparison"}:
            return "unsupported_type"
        context = row["context"]
        titles = context["title"]
        sentence_groups = context["sentences"]
        if len(titles) != 10 or len(sentence_groups) != 10:
            return "candidate_count_not_10"
        gold = {normalize_title(title) for title in row["supporting_facts"]["title"]}
        if len(gold) != 2:
            return "unique_gold_titles_not_2"
        matched = sum(normalize_title(title) in gold for title in titles)
        if matched != 2:
            return "gold_candidates_not_exactly_2"
        return None
    except (KeyError, TypeError):
        return "malformed_row"


def prepare_rows(row: dict[str, Any]) -> tuple[str, str, list[str], set[int]]:
    qid = str(row["id"])
    question = str(row["question"])
    kind = str(row["type"])
    if kind not in {"bridge", "comparison"}:
        raise ValueError(f"Unexpected question type for {qid}: {kind}")
    context = row["context"]
    titles = [str(title) for title in context["title"]]
    sentences = context["sentences"]
    if len(titles) != 10 or len(sentences) != 10:
        raise ValueError(f"Expected ten candidate paragraphs for {qid}")
    paragraphs = []
    for title, sentence_group in zip(titles, sentences):
        body = " ".join(str(sentence).strip() for sentence in sentence_group)
        paragraphs.append((title.strip() + " " + body).strip())
    gold_titles = {normalize_title(title) for title in row["supporting_facts"]["title"]}
    if len(gold_titles) != 2:
        raise ValueError(f"Expected two unique supporting-fact titles for {qid}")
    gold_indices = {
        index for index, title in enumerate(titles) if normalize_title(title) in gold_titles
    }
    if len(gold_indices) != 2:
        raise ValueError(f"Gold paragraphs do not map to two candidates for {qid}")
    return qid, question, paragraphs, gold_indices


def aggregate_metrics(
    rows: list[dict[str, Any]], splits: list[str]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        output[split] = {}
        for group in ("overall", "bridge", "comparison"):
            group_rows = (
                split_rows
                if group == "overall"
                else [row for row in split_rows if row["type"] == group]
            )
            output[split][group] = {"query_n": len({row["qid"] for row in group_rows})}
            for method in METHODS:
                output[split][group][method] = {}
                for k in range(1, MAX_K + 1):
                    cell = [row for row in group_rows if row["method"] == method and row["k"] == k]
                    if not cell:
                        output[split][group][method][str(k)] = None
                        continue
                    output[split][group][method][str(k)] = {
                        "sp_paragraph_recall": float(np.mean([r["sp_paragraph_recall"] for r in cell])),
                        "full_support_rate": float(np.mean([r["full_support"] for r in cell])),
                        "evidence_tokens_mean": float(np.mean([r["evidence_tokens"] for r in cell])),
                        "latency_seconds_mean": float(np.mean([r["latency_seconds"] for r in cell])),
                    }
    return output


def failure_slices(
    rows: list[dict[str, Any]], query_info: list[dict[str, Any]]
) -> dict[str, Any]:
    info = {item["qid"]: item for item in query_info}
    bridge = [item for item in query_info if item["split"] == "holdout" and item["type"] == "bridge"]
    by_method: dict[str, Any] = {}
    for method in METHODS:
        cells = [
            row
            for row in rows
            if row["split"] == "holdout"
            and row["type"] == "bridge"
            and row["method"] == method
            and row["k"] == 2
        ]
        gap_queries = 0
        failed_queries = 0
        high_entity_hits = 0
        entity_eligible = 0
        for row in cells:
            meta = info[row["qid"]]
            selected = set(row["selected"])
            missed = set(meta["gold_indices"]) - selected
            if missed:
                failed_queries += 1
            if any(meta["dense_rank_by_candidate"][index] >= 3 for index in missed):
                gap_queries += 1
            entities = meta["query_entities"]
            if entities:
                entity_eligible += 1
                hit = any(
                    index not in meta["gold_indices"]
                    and meta["dense_rank_by_candidate"][index] <= 2
                    and any(entity_in_text(entity, meta["paragraphs"][index]) for entity in entities)
                    for index in selected
                )
                high_entity_hits += int(hit)
        n = len(bridge)
        by_method[method] = {
            "bridge_queries_n": n,
            "bridging_hop_gap_queries_n": gap_queries,
            "bridging_hop_gap_fraction_all_bridge": gap_queries / n if n else None,
            "queries_not_full_support_n": failed_queries,
            "bridging_hop_gap_fraction_among_failures": (
                gap_queries / failed_queries if failed_queries else None
            ),
            "entity_eligible_queries_n": entity_eligible,
            "high_similarity_entity_distractor_queries_n": high_entity_hits,
            "high_similarity_entity_distractor_fraction_entity_eligible": (
                high_entity_hits / entity_eligible if entity_eligible else None
            ),
        }
    return by_method


def calculate_gates(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    bridge = metrics["holdout"]["bridge"]
    scores = {
        method: bridge[method]["2"]["full_support_rate"] for method in SINGLE_SHOT
    }
    best_method = max(SINGLE_SHOT, key=lambda method: scores[method])
    best_score = scores[best_method]
    mdr_score = bridge["mdr_style"]["2"]["full_support_rate"]
    delta = mdr_score - best_score

    bridge_qids = sorted(
        {
            row["qid"]
            for row in rows
            if row["split"] == "holdout" and row["type"] == "bridge" and row["k"] == 2
        }
    )
    values = {qid: {} for qid in bridge_qids}
    for row in rows:
        if row["split"] != "holdout" or row["type"] != "bridge" or row["k"] != 2:
            continue
        values[row["qid"]][row["method"]] = float(row["full_support"])
    matrix = np.asarray(
        [[values[qid][method] for method in SINGLE_SHOT] for qid in bridge_qids],
        dtype=np.float64,
    )
    mdr_values = np.asarray([values[qid]["mdr_style"] for qid in bridge_qids], dtype=np.float64)
    rng = np.random.default_rng(0)
    deltas = np.empty(1000, dtype=np.float64)
    for index in range(len(deltas)):
        sample = rng.integers(0, len(bridge_qids), size=len(bridge_qids))
        deltas[index] = float(mdr_values[sample].mean() - matrix[sample].mean(axis=0).max())
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975]).tolist()
    ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    disco_minus_dense = (
        bridge["disco_approx_cost_norm"]["2"]["full_support_rate"]
        - bridge["dense"]["2"]["full_support_rate"]
    )
    return {
        "G1": {
            "name": "headroom",
            "best_single_shot_method": best_method,
            "best_full_support_rate": best_score,
            "threshold": 0.75,
            "pass": bool(best_score <= 0.75),
        },
        "G2": {
            "name": "set-conditioning value",
            "mdr_full_support_rate": mdr_score,
            "best_single_shot_method": best_method,
            "best_single_shot_full_support_rate": best_score,
            "difference_absolute": delta,
            "bootstrap_ci_95_percentile": [float(ci_low), float(ci_high)],
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 0,
            "bootstrap_reselects_best_single_shot": True,
            "ci_excludes_zero": ci_excludes_zero,
            "minimum_difference": 0.08,
            "pass": bool(delta >= 0.08 and ci_excludes_zero),
        },
        "G3": {
            "name": "coverage != support",
            "disco_cost_norm_minus_dense_full_support": disco_minus_dense,
            "maximum_allowed_difference": 0.03,
            "pass": bool(disco_minus_dense <= 0.03),
        },
    }


def package_versions() -> dict[str, str]:
    packages = ["numpy", "torch", "transformers", "tokenizers", "huggingface-hub", "pyarrow"]
    return {name: importlib.metadata.version(name) for name in packages}


def cpu_description() -> str:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return platform.processor() or platform.machine()


def disco_repo_info() -> dict[str, Any]:
    repo = BASE / "DISCo"
    if not repo.exists():
        return {"status": "missing"}
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    working_bytes = sum(
        path.stat().st_size
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    all_bytes = sum(path.stat().st_size for path in repo.rglob("*") if path.is_file())
    du = subprocess.check_output(["du", "-sh", str(repo)], text=True).split()[0]
    def manifest_lines(name: str) -> list[str]:
        return [
            line.strip()
            for line in (repo / name).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#") and not line.startswith("--")
        ]
    cpu_lines = (repo / "ColBERT" / "conda_env_cpu.yml").read_text().splitlines()
    cpu_dependencies = []
    in_dependencies = False
    for line in cpu_lines:
        if line.strip() == "dependencies:":
            in_dependencies = True
            continue
        if not in_dependencies:
            continue
        top_level = re.match(r"^  - (?!pip:)(.+)$", line)
        pip_level = re.match(r"^    - (.+)$", line)
        if top_level:
            cpu_dependencies.append(top_level.group(1))
        elif pip_level:
            cpu_dependencies.append(pip_level.group(1))
    dependency_packages = {
        "python_3_10_requirements": manifest_lines("disco_requirements_py3_10.txt"),
        "torch_requirements": manifest_lines("disco_requirements_torch.txt"),
        "cpu_conda_environment": cpu_dependencies,
    }
    return {
        "url": "https://github.com/structlearning/DISCo",
        "shallow": True,
        "commit": commit,
        "du_size_including_git": du,
        "working_tree_file_bytes_excluding_git": working_bytes,
        "all_file_bytes_including_git": all_bytes,
        "dependency_manifests": [
            "disco_requirements_py3_10.txt",
            "disco_requirements_torch.txt",
            "ColBERT/conda_env_cpu.yml",
        ],
        "dependency_packages": dependency_packages,
        "cpu_assessment": (
            "A CPU Conda environment is documented (faiss-cpu and pytorch-cpu 1.13), "
            "but the recommended install script also installs requirements that pin "
            "faiss-gpu-cu12 and CUDA PyTorch, and expects separate WARP/submodlib checkouts. "
            "CPU use is possible in principle but not turnkey on this macOS ARM host."
        ),
        "cost_estimate": (
            "Not run. Planning estimate: 30-120 minutes wall time on this 16-core M4 Max "
            "for a 4,000-paragraph holdout index plus 400 queries, excluding environment "
            "setup and any dataset/model download; low confidence because the repository "
            "documents GPU use as preferred and its CPU path was not benchmarked here."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# IRE Problem 5 pilot results",
        "",
        "Primary result: the 400-question holdout. The 200-question dev split is shown only as a fixed-pipeline sanity check; no tuning was performed.",
        "",
        "## Provenance and runtime",
        "",
        f"- Data: {result['data']['source_url']}",
        f"- Raw Parquet SHA256: {result['data']['sha256']} ({result['data']['bytes']:,} bytes)",
        f"- Eligibility: {result['data']['eligible_rows_in_source']:,}/{result['data']['rows_in_source']:,} rows retained; excluded {result['data']['excluded_rows_by_reason']}.",
        f"- Sample: {result['sampling']['sample_n']} questions from eligible source rows, NumPy seed {result['sampling']['seed']}; qid-hash sorted split ({result['sampling']['dev_n']} dev, {result['sampling']['holdout_n']} holdout).",
        "- Protocol deviation: the first interrupted attempt found malformed rows after transiently scoring preceding rows; no aggregates/gates or result artifacts were produced. Final analysis applies the documented schema filter before sampling.",
        f"- Encoder: {result['model']['id']} at revision {result['model']['revision']}; CPU, {result['model']['selected_model_file_bytes']:,} model-file bytes.",
        f"- Runtime: Python {result['runtime']['python']}; NumPy {result['runtime']['packages']['numpy']}; PyTorch {result['runtime']['packages']['torch']}; Transformers {result['runtime']['packages']['transformers']}; PyArrow {result['runtime']['packages']['pyarrow']}.",
        f"- CPU: {result['runtime']['cpu']}; {result['runtime']['cpu_count']} logical CPUs; {result['model']['thread_count']} PyTorch threads.",
        f"- Candidate-index build: {result['timing']['candidate_index_build_seconds']:.2f} s; full run: {result['timing']['total_wall_seconds']:.2f} s.",
        "",
        "## Holdout metrics",
        "",
        "Cells report FULL-SUPPORT rate, SP paragraph recall, mean whitespace evidence tokens, and mean query latency in seconds. All k values are included.",
        "",
    ]
    for group in ("overall", "bridge", "comparison"):
        lines.extend(
            [
                f"### {group.title()}",
                "",
                "| Method | k | FULL-SUPPORT | SP recall | Evidence tokens | Latency (s/query) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            for k in range(1, MAX_K + 1):
                cell = result["metrics"]["holdout"][group][method][str(k)]
                if cell is None:
                    continue
                lines.append(
                    f"| {method} | {k} | {cell['full_support_rate']:.3f} | "
                    f"{cell['sp_paragraph_recall']:.3f} | {cell['evidence_tokens_mean']:.1f} | "
                    f"{cell['latency_seconds_mean']:.6f} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Pre-registered gates (holdout bridge, k=2)",
            "",
        ]
    )
    for key, gate in result["gates"].items():
        if key == "G2":
            ci = gate["bootstrap_ci_95_percentile"]
            detail = (
                f"delta={gate['difference_absolute']:+.3f}; paired bootstrap 95% CI "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            )
        elif key == "G1":
            detail = (
                f"best={gate['best_single_shot_method']} at "
                f"{gate['best_full_support_rate']:.3f} (threshold <=0.750)"
            )
        else:
            detail = (
                f"cost-normalized DISCo approximation minus dense="
                f"{gate['disco_cost_norm_minus_dense_full_support']:+.3f} "
                f"(threshold <=+0.030)"
            )
        lines.append(f"- **{key} {gate['name']}: {'PASS' if gate['pass'] else 'FAIL'}** — {detail}.")
    lines.extend(
        [
            "",
            "## k=2 bridge failure slices",
            "",
            "| Method | Bridging-hop gap / bridge n | Gap among failures | Entity distractor / eligible n |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, item in result["failure_slices_holdout_bridge_k2"].items():
        frac_all = item["bridging_hop_gap_fraction_all_bridge"]
        frac_fail = item["bridging_hop_gap_fraction_among_failures"]
        frac_entity = item["high_similarity_entity_distractor_fraction_entity_eligible"]
        show = lambda value: "NA" if value is None else f"{value:.3f}"
        lines.append(
            f"| {method} | {item['bridging_hop_gap_queries_n']}/{item['bridge_queries_n']} "
            f"({show(frac_all)}) | "
            f"{show(frac_fail)} | {item['high_similarity_entity_distractor_queries_n']}/"
            f"{item['entity_eligible_queries_n']} ({show(frac_entity)}) |"
        )
    lines.extend(
        [
            "",
            "## DISCo repository feasibility",
            "",
            f"- Shallow clone: commit {result['disco_repo']['commit']}, "
            f"{result['disco_repo']['du_size_including_git']} including .git; "
            f"{result['disco_repo']['working_tree_file_bytes_excluding_git']:,} checked-out file bytes.",
            f"- CPU: {result['disco_repo']['cpu_assessment']}",
            f"- Python 3.10 dependencies: {', '.join(result['disco_repo']['dependency_packages']['python_3_10_requirements'])}.",
            f"- PyTorch pins: {', '.join(result['disco_repo']['dependency_packages']['torch_requirements'])}; CPU environment packages: {', '.join(result['disco_repo']['dependency_packages']['cpu_conda_environment'])}.",
            f"- Estimated run cost: {result['disco_repo']['cost_estimate']}",
            "- No DISCo training or heavy run was started.",
            "",
            "## Dev sanity check",
            "",
            "The fixed methods were also evaluated on the 200 dev questions; these values are secondary and were not used for tuning or gate decisions.",
            "",
            "| Method | FULL-SUPPORT@2 overall | SP recall@2 overall | Mean evidence tokens@2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        cell = result["metrics"]["dev"]["overall"][method]["2"]
        lines.append(
            f"| {method} | {cell['full_support_rate']:.3f} | "
            f"{cell['sp_paragraph_recall']:.3f} | {cell['evidence_tokens_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Model input truncation",
            "",
            f"- Inputs encoded by role: {json.dumps(result['model_input_counts'], sort_keys=True)}",
            f"- Inputs truncated by role: {json.dumps(result['model_truncations'], sort_keys=True)}",
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Method lineage references",
            "",
            "- MDR prior art: Xiong et al., Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval (https://arxiv.org/abs/2009.12756). This pilot uses a simplified, untrained evidence-conditioned reference, not a reproduction.",
            "- DISCo source repository: https://github.com/structlearning/DISCo; associated ICLR 2026 paper: https://openreview.net/pdf?id=cUdODCFjUM. The coverage method here is explicitly an approximation.",
            "- Encoder model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    output_json = BASE / "results.json"
    output_md = BASE / "results.md"
    for path in (output_json, output_md):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset file is missing: {DATA_PATH}")

    run_started = time.perf_counter()
    raw_sha = sha256_file(DATA_PATH)
    table_started = time.perf_counter()
    import pyarrow.parquet as parquet

    source_rows = parquet.read_table(DATA_PATH).to_pylist()
    table_load_seconds = time.perf_counter() - table_started
    excluded_rows: Counter[str] = Counter()
    eligible_rows = []
    for row in source_rows:
        reason = eligibility_reason(row)
        if reason is None:
            eligible_rows.append(row)
        else:
            excluded_rows[reason] += 1
    if len(eligible_rows) < SAMPLE_N:
        raise ValueError(f"Dataset has only {len(eligible_rows)} eligible rows; need {SAMPLE_N}")

    sample_rng = np.random.default_rng(SEED)
    sampled_indices = sample_rng.choice(len(eligible_rows), size=SAMPLE_N, replace=False)
    sampled = [eligible_rows[int(index)] for index in sampled_indices]
    if len({str(row["id"]) for row in sampled}) != SAMPLE_N:
        raise ValueError("Sample contains duplicate question IDs")
    sampled.sort(
        key=lambda row: (
            hashlib.sha256(str(row["id"]).encode("utf-8")).hexdigest(),
            str(row["id"]),
        )
    )
    split_rows = [
        (row, "dev" if index < DEV_N else "holdout")
        for index, row in enumerate(sampled)
    ]

    encoder, model_snapshot, model_info = load_encoder()
    paragraph_index_seconds = 0.0
    result_rows: list[dict[str, Any]] = []
    query_info: list[dict[str, Any]] = []
    type_counts = Counter()
    prepared = []
    for row, split in split_rows:
        qid, question, paragraphs, gold_indices = prepare_rows(row)
        kind = str(row["type"])
        type_counts[(split, kind)] += 1
        prepared.append(
            {
                "qid": qid,
                "question": question,
                "paragraphs": paragraphs,
                "gold_indices": gold_indices,
                "split": split,
                "type": kind,
            }
        )

    question_batch_size = 10
    for group_start in range(0, len(prepared), question_batch_size):
        group = prepared[group_start : group_start + question_batch_size]
        flat_paragraphs = [
            paragraph for item in group for paragraph in item["paragraphs"]
        ]
        index_started = time.perf_counter()
        flat_vectors, flat_tokens, _ = encoder.encode(
            flat_paragraphs, role="candidate_paragraphs", need_tokens=True
        )
        paragraph_index_seconds += time.perf_counter() - index_started
        for group_index, item in enumerate(group):
            first = group_index * 10
            paragraph_vectors = flat_vectors[first : first + 10]
            paragraph_tokens = flat_tokens[first : first + 10]
            method_results, dense_order = build_methods(
                encoder,
                item["qid"],
                item["question"],
                item["paragraphs"],
                paragraph_vectors,
                paragraph_tokens,
                item["gold_indices"],
            )
            dense_rank = [0] * len(dense_order)
            for rank, candidate_index in enumerate(dense_order, start=1):
                dense_rank[candidate_index] = rank
            query_info.append(
                {
                    "qid": item["qid"],
                    "split": item["split"],
                    "type": item["type"],
                    "gold_indices": sorted(item["gold_indices"]),
                    "dense_rank_by_candidate": dense_rank,
                    "query_entities": extract_query_entities(item["question"]),
                    "paragraphs": item["paragraphs"],
                }
            )
            for method, (order, timings) in method_results.items():
                for k in range(1, MAX_K + 1):
                    selection = order[:k]
                    metric = score_selection(
                        selection, item["gold_indices"], item["paragraphs"]
                    )
                    result_rows.append(
                        {
                            "qid": item["qid"],
                            "split": item["split"],
                            "type": item["type"],
                            "method": method,
                            "k": k,
                            "selected": selection,
                            **metric,
                            "latency_seconds": float(timings[k - 1]),
                        }
                    )
            sample_index = group_start + group_index + 1
            if sample_index % 50 == 0 or sample_index == SAMPLE_N:
                print(f"Processed {sample_index}/{SAMPLE_N} questions", flush=True)

    metrics = aggregate_metrics(result_rows, ["dev", "holdout"])
    slices = failure_slices(result_rows, query_info)
    gates = calculate_gates(metrics, result_rows)
    disco_repo = disco_repo_info()
    model_repo_files = [
        {
            "path": str(path.relative_to(model_snapshot)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(model_snapshot.rglob("*"))
        if path.is_file()
    ]
    result = {
        "protocol": "protocol.md",
        "data": {
            "source_url": DATA_URL,
            "sha256": raw_sha,
            "bytes": DATA_PATH.stat().st_size,
            "rows_in_source": len(source_rows),
            "eligible_rows_in_source": len(eligible_rows),
            "excluded_rows_by_reason": dict(excluded_rows),
            "format": "Parquet, HotpotQA distractor validation",
            "load_seconds": table_load_seconds,
        },
        "sampling": {
            "sample_n": SAMPLE_N,
            "seed": SEED,
            "eligible_population_n": len(eligible_rows),
            "selection": "numpy.default_rng(seed).choice(eligible_source_rows, 600, replace=False)",
            "split": "sort sampled rows by (sha256(qid), qid); first 200 dev, remaining 400 holdout",
            "dev_n": DEV_N,
            "holdout_n": HOLDOUT_N,
            "type_counts": {
                split: {
                    kind: int(type_counts[(split, kind)])
                    for kind in ("bridge", "comparison")
                }
                for split in ("dev", "holdout")
            },
            "sampled_qids_in_hash_order": [str(row["id"]) for row in sampled],
        },
        "model": model_info,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": cpu_description(),
            "cpu_count": os.cpu_count(),
            "packages": package_versions(),
        },
        "model_repo_file_hashes": model_repo_files,
        "model_input_counts": dict(encoder.input_counts),
        "model_truncations": dict(encoder.truncations),
        "timing": {
            "candidate_index_build_seconds": paragraph_index_seconds,
            "total_wall_seconds": time.perf_counter() - run_started,
            "latency_scope": (
                "per-query method latency includes query encoding and selection; excludes "
                "model load, raw data parsing, and one-time candidate paragraph encoding"
            ),
        },
        "methods": METHODS,
        "metrics": metrics,
        "failure_slices_holdout_bridge_k2": slices,
        "gates": gates,
        "disco_repo": disco_repo,
        "per_query": result_rows,
        "query_info": [
            {key: value for key, value in item.items() if key != "paragraphs"}
            for item in query_info
        ],
    }
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with output_md.open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(result))
    print(f"Wrote {output_json}", flush=True)
    print(f"Wrote {output_md}", flush=True)
    print(
        "Gates: "
        + ", ".join(f"{name}={'PASS' if gate['pass'] else 'FAIL'}" for name, gate in gates.items()),
        flush=True,
    )


if __name__ == "__main__":
    main()
