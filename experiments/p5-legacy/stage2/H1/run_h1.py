from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "data"
DATASETS = ("hotpotqa", "2wiki", "musique")
KS = (2, 4, 6, 10)
MAX_K = max(KS)
THREADS = 3
MAX_LENGTH = 256
MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
BGE_ID = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
METHODS = (
    "bm25",
    "dense_minilm",
    "dense_bge",
    "mmr_minilm",
    "disco_approx_cost_norm",
    "disco_approx_no_cost_norm",
    "mdr_untrained_approx",
    "beam_2hop_approx",
)
SINGLE_SHOT = METHODS[:6]
CONDITIONING = METHODS[6:]

os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
os.environ["HF_HOME"] = str(BASE / ".model-cache")
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def ready_hashes(value: Any, prefix: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(value, dict):
        explicit_path = value.get("path", value.get("file"))
        explicit_hash = value.get("sha256", value.get("hash"))
        if isinstance(explicit_path, str) and isinstance(explicit_hash, str):
            found[explicit_path.replace("\\", "/")] = explicit_hash.lower()
        for key, item in value.items():
            path = f"{prefix}/{key}".strip("/")
            if isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item):
                found[path.replace("\\", "/")] = item.lower()
            else:
                found.update(ready_hashes(item, path))
    elif isinstance(value, list):
        for item in value:
            found.update(ready_hashes(item, prefix))
    return found


def validate_ready() -> dict[str, str]:
    ready_path = DATA / "READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"required readiness marker missing: {ready_path}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    listed = ready_hashes(ready)
    actual: dict[str, str] = {}
    for ds in DATASETS:
        for name in ("corpus.jsonl", "questions.jsonl", "emb_minilm.f16.npy"):
            path = DATA / ds / name
            if not path.is_file():
                raise FileNotFoundError(f"required data file missing: {path}")
            rel = f"{ds}/{name}"
            matches = [v for key, v in listed.items() if key.rstrip("/").endswith(rel)]
            digest = sha256_file(path)
            if matches and digest not in matches:
                raise ValueError(f"READY SHA256 mismatch for {rel}")
            if not matches and digest not in listed.values():
                raise ValueError(f"READY.json does not contain a matching SHA256 for {rel}")
            actual[rel] = digest
    return actual


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def passage_text(row: dict[str, Any]) -> str:
    return (str(row["title"]).strip() + " " + str(row["text"]).strip()).strip()


def load_dataset(ds: str) -> dict[str, Any]:
    folder = DATA / ds
    corpus = read_jsonl(folder / "corpus.jsonl")
    questions = read_jsonl(folder / "questions.jsonl")
    if not corpus:
        raise ValueError(f"empty corpus for {ds}")
    pids = [str(row["pid"]) for row in corpus]
    if len(set(pids)) != len(pids):
        raise ValueError(f"duplicate pid in {ds}")
    pid_to_row = {pid: i for i, pid in enumerate(pids)}
    passages = [passage_text(row) for row in corpus]
    tokens = [len(text.split()) for text in passages]
    vectors = np.load(folder / "emb_minilm.f16.npy", mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[0] != len(corpus):
        raise ValueError(f"MiniLM embedding shape is not row-aligned for {ds}: {vectors.shape}")
    probe = np.asarray(vectors[::max(1, len(vectors) // 1024)], dtype=np.float32)
    norms = np.linalg.norm(probe, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(np.abs(norms - 1.0) > 0.03):
        raise ValueError(f"MiniLM embeddings are not normalized for {ds}")
    seen_qids: set[str] = set()
    for q in questions:
        qid = str(q["qid"])
        if qid in seen_qids:
            raise ValueError(f"duplicate qid in {ds}: {qid}")
        seen_qids.add(qid)
        if q.get("split") not in {"train", "tune", "holdout"}:
            raise ValueError(f"invalid split in {ds}/{qid}")
        if not isinstance(q.get("type"), str) or not q["type"]:
            raise ValueError(f"missing question type in {ds}/{qid}")
        gold = {str(pid) for pid in q.get("gold_pids", [])}
        if not gold or not gold.issubset(pid_to_row):
            raise ValueError(f"missing or unresolved gold_pids in {ds}/{qid}")
    if not any(q["split"] == "tune" for q in questions) or not any(
        q["split"] == "holdout" for q in questions
    ):
        raise ValueError(f"{ds} must contain both tune and holdout questions")
    return {
        "corpus": corpus,
        "questions": questions,
        "pids": pids,
        "pid_to_row": pid_to_row,
        "passages": passages,
        "token_counts": tokens,
        "minilm": vectors,
    }


class Encoder:
    def __init__(self, model_id: str, revision: str | None, role: str):
        from huggingface_hub import snapshot_download
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(THREADS)
        try:
            torch.set_num_interop_threads(THREADS)
        except RuntimeError:
            pass
        kwargs = {"repo_id": model_id, "cache_dir": str(BASE / ".model-cache" / "hub")}
        if revision:
            kwargs["revision"] = revision
        kwargs["allow_patterns"] = [
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
        self.snapshot = Path(snapshot_download(**kwargs))
        self.model_id = model_id
        self.revision = self.snapshot.name
        self.role = role
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True)
        self.model = AutoModel.from_pretrained(
            str(self.snapshot), local_files_only=True, use_safetensors=True
        ).to("cpu")
        self.model.eval()
        self.max_length = min(MAX_LENGTH, int(self.model.config.max_position_embeddings))
        self.stats: Counter[str] = Counter()

    @property
    def dimension(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(
        self, texts: list[str], role: str, need_tokens: bool = False, batch_size: int = 32
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        pooled_parts: list[np.ndarray] = []
        token_parts: list[np.ndarray] = []
        for offset in range(0, len(texts), batch_size):
            batch_texts = texts[offset : offset + batch_size]
            raw_ids = self.tokenizer(
                batch_texts, add_special_tokens=True, truncation=False, padding=False
            )["input_ids"]
            self.stats[f"{role}.inputs"] += len(batch_texts)
            self.stats[f"{role}.truncated"] += sum(len(ids) > self.max_length for ids in raw_ids)
            self.stats[f"{role}.dropped_wordpieces"] += sum(
                max(0, len(ids) - self.max_length) for ids in raw_ids
            )
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_special_tokens_mask=need_tokens,
            )
            special = batch.pop("special_tokens_mask") if need_tokens else None
            with self.torch.inference_mode():
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                pooled_parts.append(pooled.cpu().numpy().astype(np.float32, copy=False))
                if need_tokens:
                    valid = batch["attention_mask"].bool() & ~special.bool()
                    for i in range(hidden.shape[0]):
                        values = hidden[i][valid[i]]
                        values = self.torch.nn.functional.normalize(values, p=2, dim=1)
                        arr = values.cpu().numpy().astype(np.float32, copy=False)
                        if len(arr) == 0:
                            arr = np.zeros((1, self.dimension), dtype=np.float32)
                        token_parts.append(arr)
        pooled_out = np.concatenate(pooled_parts, axis=0) if pooled_parts else np.empty((0, self.dimension), np.float32)
        return pooled_out, token_parts if need_tokens else None

    def info(self) -> dict[str, Any]:
        files = [p for p in self.snapshot.rglob("*") if p.is_file()]
        weights = [p for p in files if p.name.endswith((".safetensors", ".bin"))]
        return {
            "model_id": self.model_id,
            "resolved_revision": self.revision,
            "max_length": self.max_length,
            "weight_files": [
                {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                for p in weights
            ],
        }


def build_bge_cache(ds: str, data: dict[str, Any], encoder: Encoder) -> np.memmap:
    cache_dir = BASE / "cache"
    cache_dir.mkdir(exist_ok=True)
    target = cache_dir / f"{ds}_bge.f16.npy"
    meta_path = cache_dir / f"{ds}_bge.json"
    corpus_sha = sha256_file(DATA / ds / "corpus.jsonl")
    expected = {
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "corpus_sha256": corpus_sha,
        "rows": len(data["corpus"]),
        "dimension": encoder.dimension,
        "max_length": encoder.max_length,
    }
    if target.exists() or meta_path.exists():
        if not target.is_file() or not meta_path.is_file():
            raise FileExistsError(f"incomplete existing BGE cache for {ds}; refusing to overwrite")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"existing BGE cache metadata mismatch for {ds}")
        vectors = np.load(target, mmap_mode="r")
        if vectors.shape != (expected["rows"], expected["dimension"]):
            raise ValueError(f"existing BGE cache shape mismatch for {ds}")
        if metadata.get("vectors_sha256") != sha256_file(target):
            raise ValueError(f"existing BGE cache checksum mismatch for {ds}")
        encoder.stats[f"{ds}.corpus.inputs"] += int(metadata.get("corpus_inputs", 0))
        encoder.stats[f"{ds}.corpus.truncated"] += int(metadata.get("corpus_truncated", 0))
        encoder.stats[f"{ds}.corpus.dropped_wordpieces"] += int(
            metadata.get("corpus_dropped_wordpieces", 0)
        )
        return vectors

    temp = target.with_name(target.name + ".tmp.npy")
    if temp.exists():
        raise FileExistsError(f"partial BGE cache exists at {temp}; refusing to overwrite")
    vectors = np.lib.format.open_memmap(
        temp, mode="w+", dtype=np.float16, shape=(expected["rows"], expected["dimension"])
    )
    stats_before = encoder.stats.copy()
    for start in range(0, expected["rows"], 32):
        stop = min(start + 32, expected["rows"])
        batch = encoder.encode(data["passages"][start:stop], f"{ds}.corpus")[0]
        vectors[start:stop] = batch.astype(np.float16)
        if start and start % 4096 == 0:
            print(f"{ds}: BGE corpus encoded {start}/{expected['rows']}", flush=True)
    vectors.flush()
    del vectors
    os.replace(temp, target)
    expected["vectors_sha256"] = sha256_file(target)
    expected["corpus_inputs"] = encoder.stats[f"{ds}.corpus.inputs"] - stats_before[f"{ds}.corpus.inputs"]
    expected["corpus_truncated"] = encoder.stats[f"{ds}.corpus.truncated"] - stats_before[f"{ds}.corpus.truncated"]
    expected["corpus_dropped_wordpieces"] = encoder.stats[f"{ds}.corpus.dropped_wordpieces"] - stats_before[f"{ds}.corpus.dropped_wordpieces"]
    write_json(meta_path, expected)
    return np.load(target, mmap_mode="r")


def lexical_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def build_bm25(passages: list[str]) -> Any:
    from rank_bm25 import BM25Okapi

    return BM25Okapi(
        [lexical_tokens(text) for text in passages], k1=1.2, b=0.75
    )


def stable_order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def exact_scores(vectors: np.ndarray, query: np.ndarray, block: int = 65536) -> np.ndarray:
    scores = np.empty(vectors.shape[0], dtype=np.float32)
    for start in range(0, vectors.shape[0], block):
        stop = min(start + block, vectors.shape[0])
        scores[start:stop] = np.asarray(vectors[start:stop], dtype=np.float32) @ query
    return scores


def mmr_order(
    query: np.ndarray, top_vectors: np.ndarray, dense_indices: np.ndarray, max_k: int = MAX_K
) -> list[int]:
    selected_local: list[int] = []
    vectors = np.asarray(top_vectors, dtype=np.float32)
    relevance = vectors @ query
    redundancy = np.zeros(len(dense_indices), dtype=np.float32)
    selected_local.append(0)
    redundancy = vectors @ vectors[0]
    while len(selected_local) < min(max_k, len(dense_indices)):
        scores = 0.5 * relevance - 0.5 * redundancy
        scores[selected_local] = -np.inf
        idx = int(np.argmax(scores))
        selected_local.append(idx)
        sims = vectors @ vectors[idx]
        redundancy = np.maximum(redundancy, sims)
    return [int(dense_indices[i]) for i in selected_local]


def facility_coverages(query_tokens: np.ndarray, passage_tokens: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    rows: list[np.ndarray] = []
    costs: list[int] = []
    for tokens in passage_tokens:
        if len(tokens):
            rows.append(np.max(query_tokens @ tokens.T, axis=1))
            costs.append(len(tokens))
        else:
            rows.append(np.zeros(len(query_tokens), dtype=np.float32))
            costs.append(1)
    return np.stack(rows), costs


def greedy_coverage_order(
    coverages: np.ndarray, costs: list[int], normalize_cost: bool, max_k: int = MAX_K
) -> list[int]:
    selected: list[int] = []
    current: np.ndarray | None = None
    while len(selected) < min(max_k, len(costs)):
        best_idx = -1
        best_score = -np.inf
        best_current = None
        for idx, coverage in enumerate(coverages):
            if idx in selected:
                continue
            new_current = coverage if current is None else np.maximum(current, coverage)
            gain = float(np.sum(new_current)) if current is None else float(np.sum(new_current - current))
            score = gain / max(1, costs[idx]) if normalize_cost else gain
            if score > best_score:
                best_idx, best_score, best_current = idx, score, new_current
        selected.append(best_idx)
        current = np.array(best_current, copy=True)
    return selected


def selection_metrics(order: list[int], gold: set[int], token_counts: list[int], k: int) -> dict[str, Any]:
    chosen = order[:k]
    hits = len(set(chosen) & gold)
    return {
        "full_support": bool(gold.issubset(chosen)),
        "sp_recall": hits / len(gold),
        "evidence_tokens": int(sum(token_counts[i] for i in chosen)),
    }


def method_orders(
    question: str,
    data: dict[str, Any],
    bm25: Any,
    bge_vectors: np.ndarray,
    minilm_encoder: Encoder,
    bge_encoder: Encoder,
    split: str,
) -> dict[str, tuple[list[int], list[float]]]:
    passages = data["passages"]
    n = len(passages)
    times: dict[str, tuple[list[int], list[float]]] = {}

    start = time.perf_counter()
    bm_scores = bm25.get_scores(lexical_tokens(question))
    bm_order = stable_order(np.asarray(bm_scores))
    bm_elapsed = time.perf_counter() - start
    times["bm25"] = (bm_order[:MAX_K].astype(int).tolist(), [bm_elapsed] * len(KS))

    q_start = time.perf_counter()
    q_minilm, q_token_list = minilm_encoder.encode([question], f"{split}.query", need_tokens=True)
    minilm_q_elapsed = time.perf_counter() - q_start
    qv = q_minilm[0]
    min_scores = exact_scores(data["minilm"], qv)
    min_order = stable_order(min_scores)
    dense_time = time.perf_counter() - q_start
    times["dense_minilm"] = (min_order[:MAX_K].astype(int).tolist(), [dense_time] * len(KS))

    bq_start = time.perf_counter()
    q_bge = bge_encoder.encode([BGE_QUERY_PREFIX + question], f"{split}.query")[0][0]
    bge_scores = exact_scores(bge_vectors, q_bge)
    bge_order = stable_order(bge_scores)
    times["dense_bge"] = (bge_order[:MAX_K].astype(int).tolist(), [time.perf_counter() - bq_start] * len(KS))

    mm_start = time.perf_counter()
    top100 = min_order[: min(100, n)]
    mm_order = mmr_order(qv, np.asarray(data["minilm"][top100], dtype=np.float32), top100)
    mm_elapsed = dense_time + time.perf_counter() - mm_start
    times["mmr_minilm"] = (mm_order, [mm_elapsed] * len(KS))

    disco_start = time.perf_counter()
    top50 = min_order[: min(50, n)]
    _, passage_token_vectors = minilm_encoder.encode(
        [passages[i] for i in top50], f"{split}.disco_candidates", need_tokens=True
    )
    coverages, costs = facility_coverages(q_token_list[0], passage_token_vectors or [])
    cover_seconds = time.perf_counter() - disco_start
    for name, norm in (("disco_approx_cost_norm", True), ("disco_approx_no_cost_norm", False)):
        select_start = time.perf_counter()
        local = greedy_coverage_order(coverages, costs, norm)
        elapsed = dense_time + cover_seconds + time.perf_counter() - select_start
        times[name] = ([int(top50[i]) for i in local], [elapsed] * len(KS))

    selected: list[int] = []
    mdr_steps: list[float] = []
    mdr_started = time.perf_counter()
    for step in range(MAX_K):
        if step == 0:
            scores, order = min_scores, min_order
        else:
            conditioned = question + " " + " ".join(passages[i] for i in selected)
            cv = minilm_encoder.encode([conditioned], f"{split}.mdr_conditioned")[0][0]
            scores = exact_scores(data["minilm"], cv)
            order = stable_order(scores)
        for candidate in order:
            idx = int(candidate)
            if idx not in selected:
                selected.append(idx)
                break
        mdr_steps.append(dense_time + time.perf_counter() - mdr_started)
    times["mdr_untrained_approx"] = (selected, [mdr_steps[k - 1] for k in KS])

    beam_started = time.perf_counter()
    best_score = -np.inf
    best_pair: tuple[int, int] | None = None
    best_second_order: np.ndarray | None = None
    for first_value in min_order[: min(5, n)]:
        first = int(first_value)
        conditioned = question + " " + passages[first]
        cv = minilm_encoder.encode([conditioned], f"{split}.beam_conditioned")[0][0]
        second_scores = exact_scores(data["minilm"], cv)
        second_scores[first] = -np.inf
        second_order = stable_order(second_scores)
        second = int(second_order[0])
        pair_score = float(min_scores[first] + second_scores[second])
        if pair_score > best_score:
            best_score = pair_score
            best_pair = (first, second)
            best_second_order = second_order
    if best_pair is None or best_second_order is None:
        raise ValueError("beam search needs at least two corpus passages")
    beam_order = list(best_pair) + [
        int(i) for i in best_second_order if int(i) not in best_pair
    ]
    beam_elapsed = dense_time + time.perf_counter() - beam_started
    times["beam_2hop_approx"] = (beam_order[:MAX_K], [beam_elapsed] * len(KS))
    return times


def paired_bootstrap_ci(a: list[float], b: list[float], seed: int = 0, reps: int = 1000) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        raise ValueError("paired bootstrap requires equal nonempty samples")
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        means[i] = delta[rng.integers(0, len(delta), size=len(delta))].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": len(delta),
        "delta": float(delta.mean()),
        "ci95": [float(low), float(high)],
        "bootstrap_reps": reps,
        "seed": seed,
        "excludes_zero": bool(low > 0 or high < 0),
    }


def summarize(per_q: dict[str, dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[str]] = {"overall": list(per_q)}
    for qid, row in per_q.items():
        groups.setdefault(str(row["type"]), []).append(qid)
    output: dict[str, Any] = {}
    for group, qids in groups.items():
        cell: dict[str, Any] = {"n": len(qids)}
        for method in METHODS:
            cell[method] = {}
            for k in KS:
                vals = [per_q[qid]["methods"][method][str(k)] for qid in qids]
                cell[method][str(k)] = {
                    "full_support": float(np.mean([v["full_support"] for v in vals])),
                    "sp_recall": float(np.mean([v["sp_recall"] for v in vals])),
                    "evidence_tokens_mean": float(np.mean([v["evidence_tokens"] for v in vals])),
                    "latency_seconds_mean": float(np.mean([v["latency_seconds"] for v in vals])),
                    "n": len(vals),
                }
        output[group] = cell
    return output


def evaluate_split(
    ds: str,
    split: str,
    data: dict[str, Any],
    bm25: Any,
    bge_vectors: np.ndarray,
    minilm_encoder: Encoder,
    bge_encoder: Encoder,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    questions = [q for q in data["questions"] if q["split"] == split]
    for index, question in enumerate(questions, 1):
        qid = str(question["qid"])
        gold = {data["pid_to_row"][str(pid)] for pid in question["gold_pids"]}
        orders = method_orders(
            str(question["question"]), data, bm25, bge_vectors,
            minilm_encoder, bge_encoder, split,
        )
        methods: dict[str, Any] = {}
        for method, (order, latency_by_k) in orders.items():
            methods[method] = {}
            for k, latency in zip(KS, latency_by_k):
                methods[method][str(k)] = {
                    **selection_metrics(order, gold, data["token_counts"], k),
                    "latency_seconds": float(latency),
                }
        result[qid] = {"type": str(question["type"]), "methods": methods}
        if index % 250 == 0:
            print(f"{ds}/{split}: evaluated {index}/{len(questions)}", flush=True)
    return result


def tune_selection(tune_per_q: dict[str, Any]) -> dict[str, str]:
    def best(methods: tuple[str, ...]) -> str:
        scores = {
            method: float(np.mean([
                row["methods"][method]["4"]["full_support"]
                for row in tune_per_q.values()
            ]))
            for method in methods
        }
        return max(methods, key=lambda method: scores[method])

    return {
        "best_conditioning_method": best(CONDITIONING),
        "best_single_shot_method": best(SINGLE_SHOT),
        "selection_split": "hotpotqa/tune",
        "selection_metric": "overall FULL-SUPPORT@4",
    }


def gates(holdout: dict[str, Any], frozen: dict[str, str]) -> dict[str, Any]:
    per_q = holdout["hotpotqa"]
    qids = list(per_q)
    disco = [per_q[q]["methods"]["disco_approx_cost_norm"]["4"]["full_support"] for q in qids]
    dense = [per_q[q]["methods"]["dense_minilm"]["4"]["full_support"] for q in qids]
    g3 = paired_bootstrap_ci(disco, dense)
    g3["method_a"] = "disco_approx_cost_norm"
    g3["method_b"] = "dense_minilm"
    g3["k"] = 4
    g3["pass"] = g3["delta"] > 0.03 and g3["excludes_zero"]
    ca = frozen["best_conditioning_method"]
    sb = frozen["best_single_shot_method"]
    cond = [per_q[q]["methods"][ca]["4"]["full_support"] for q in qids]
    single = [per_q[q]["methods"][sb]["4"]["full_support"] for q in qids]
    g2 = paired_bootstrap_ci(cond, single)
    g2["method_a"] = ca
    g2["method_b"] = sb
    g2["k"] = 4
    g2["pass"] = g2["delta"] >= 0.05 and g2["excludes_zero"]
    return {"G3_prime": g3, "G2_prime": g2}


def write_report(result: dict[str, Any]) -> None:
    lines = ["# H1 open-pool replication", ""]
    if result.get("status") == "awaiting_data":
        lines += ["Status: awaiting shared data readiness.", "", "No metrics have been computed."]
    elif result.get("status") == "tune_complete":
        lines += [
            "Status: tune complete; holdout has not been run.",
            "",
            f"Frozen conditioning method: {result['frozen_config']['best_conditioning_method']}.",
            f"Frozen single-shot method: {result['frozen_config']['best_single_shot_method']}.",
            "",
            "No holdout metrics have been computed.",
        ]
    elif result.get("status") != "complete":
        lines += [f"Status: {result.get('status')}", "", result.get("message", "No metrics computed.")]
    else:
        lines += ["## Holdout gates", ""]
        for key, gate in result["gates"].items():
            low, high = gate["ci95"]
            lines.append(
                f"- {key}: {'PASS' if gate['pass'] else 'FAIL'}; "
                f"{gate['method_a']} − {gate['method_b']} FULL-SUPPORT@4 "
                f"= {gate['delta']:+.4f}, 95% paired bootstrap CI "
                f"[{low:+.4f}, {high:+.4f}], n={gate['n']}."
            )
        lines += ["", "## Scope and provenance", ""]
        lines.append("See `results.json` for metrics by method, k, dataset, and question type, plus hashes and truncation counts.")
        lines.append(f"Frozen on tune: conditioning={result['frozen_config']['best_conditioning_method']}; single-shot={result['frozen_config']['best_single_shot_method']}.")
        if result.get("deviations"):
            lines.append("Declared deviations: " + "; ".join(result["deviations"]))
        else:
            lines.append("Declared deviations: none.")
    (BASE / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    excluded = {"manifest.sha256"}
    lines = []
    for path in sorted(BASE.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        if any(part in {".model-cache", ".uv-cache", ".venv", "__pycache__"} for part in path.parts):
            continue
        rel = path.relative_to(BASE).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")
    (BASE / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage(stage: str) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    results_path = BASE / "results.json"
    log_path = BASE / "holdout_runs.log"
    current = json.loads(results_path.read_text()) if results_path.exists() else {"status": "running"}
    if stage == "holdout":
        if log_path.exists() and "START " in log_path.read_text(encoding="utf-8"):
            raise RuntimeError("holdout already started; one-shot policy forbids rerun")
        if not (BASE / "frozen_config.json").is_file():
            raise FileNotFoundError("run --stage tune and freeze config before holdout")
        frozen = json.loads((BASE / "frozen_config.json").read_text(encoding="utf-8"))
    else:
        if "tune" in current.get("per_query", {}):
            raise FileExistsError("tune results already exist; refusing accidental rerun")
        frozen = {}

    input_hashes = validate_ready()
    minilm_encoder = Encoder(MINILM_ID, MINILM_REVISION, "minilm")
    bge_encoder = Encoder(BGE_ID, None, "bge")
    result = current
    result.update({
        "status": "running",
        "protocol": "protocol.md",
        "input_sha256": input_hashes,
        "methods": list(METHODS),
        "k": list(KS),
        "cpu_threads": THREADS,
    })
    per_query = result.setdefault("per_query", {})
    split_key = "tune" if stage == "tune" else "holdout"
    holdout_claimed = False
    for ds in DATASETS:
        data = load_dataset(ds)
        bge_vectors = build_bge_cache(ds, data, bge_encoder)
        bm25 = build_bm25(data["passages"])
        if stage == "holdout" and not holdout_claimed:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"START {time.strftime('%Y-%m-%dT%H:%M:%S%z')} config={sha256_file(BASE / 'frozen_config.json')}\n")
            holdout_claimed = True
        per_query.setdefault(split_key, {})[ds] = evaluate_split(
            ds, split_key, data, bm25, bge_vectors, minilm_encoder, bge_encoder
        )
        result["metrics"] = {
            split: {dataset: summarize(rows) for dataset, rows in datasets.items()}
            for split, datasets in per_query.items()
        }
        write_json(results_path, result)
        del data, bge_vectors, bm25

    result["encoder_info"] = {"minilm": minilm_encoder.info(), "bge": bge_encoder.info()}
    result.setdefault("encoder_input_stats_by_stage", {})[split_key] = {
        "minilm": dict(minilm_encoder.stats), "bge": dict(bge_encoder.stats)
    }
    result["software"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("rank_bm25", "torch", "transformers", "huggingface_hub")
        },
    }
    if stage == "tune":
        result["frozen_config"] = tune_selection(per_query["tune"]["hotpotqa"])
        write_json(BASE / "frozen_config.json", result["frozen_config"])
        result["status"] = "tune_complete"
    else:
        result["frozen_config"] = frozen
        result["gates"] = gates(per_query["holdout"], frozen)
        result["status"] = "complete"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"DONE {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    write_json(results_path, result)
    write_report(result)
    write_manifest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("tune", "holdout"), required=True)
    args = parser.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
