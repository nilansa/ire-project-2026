#!/usr/bin/env python3
"""H3 open-scale sufficiency experiment. See protocol.md before running."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
DATA = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/stage2/data")
MODELS = BASE / ".models"
os.environ.setdefault("HF_HOME", str(BASE / ".hf-cache"))
os.environ.setdefault("UV_CACHE_DIR", str(BASE / ".uv-cache"))
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["NUMEXPR_NUM_THREADS"] = "3"

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(3)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

DATASETS = ("hotpotqa", "2wiki", "musique")
RETRIEVERS = ("dense", "disco_approx")
DETECTORS = ("min_score", "sum_score", "logistic", "nli")
K_VALUES = tuple(range(2, 11))
SEED = 20261031
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 0
MAX_PER_SPLIT = 200
POOL_SIZE = 256
MAX_LENGTH = 256
NLI_MAX_LENGTH = 512
MAX_DOWNLOAD_BYTES = 1_000_000_000
MAX_NLI_BYTES = 600_000_000
STALL_SECONDS = 120
QUERY_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "whom", "why",
}
ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|\d{4})\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def ready_hashes(node: Any, prefix: str = "") -> dict[str, str]:
    """Accept flat path->hash, nested dataset maps, or path/hash records."""
    found: dict[str, str] = {}
    if isinstance(node, list):
        for item in node:
            found.update(ready_hashes(item, prefix))
    elif isinstance(node, dict):
        path = next((node[k] for k in ("path", "file", "filename", "name") if isinstance(node.get(k), str)), None)
        digest = next((node[k] for k in ("sha256", "hash", "digest") if _is_sha(node.get(k))), None)
        if path and digest:
            found[path] = digest.lower()
        for key, value in node.items():
            if key in {"path", "file", "filename", "name", "sha256", "hash", "digest"}:
                continue
            next_prefix = f"{prefix}/{key}".strip("/")
            if _is_sha(value):
                found[next_prefix] = value.lower()
            elif isinstance(value, (dict, list)):
                found.update(ready_hashes(value, next_prefix))
    return found


def verify_ready(data_root: Path = DATA) -> dict[str, Any]:
    marker = data_root / "READY.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Data are not ready: {marker} is absent")
    ready_bytes = marker.read_bytes()
    manifest = json.loads(ready_bytes)
    expected = ready_hashes(manifest)
    actual_paths = sorted(p for p in data_root.rglob("*") if p.is_file() and p != marker)
    if not expected:
        raise ValueError("READY.json contains no per-file SHA-256 entries")
    matched: dict[Path, str] = {}
    for raw_path, digest in expected.items():
        normalized = raw_path.replace("\\", "/").lstrip("/")
        candidates = [p for p in actual_paths if normalized == p.relative_to(data_root).as_posix() or normalized.endswith("/" + p.relative_to(data_root).as_posix())]
        if len(candidates) == 1:
            matched[candidates[0]] = digest
    actual_set = set(actual_paths)
    if set(matched) != actual_set:
        missing = sorted(str(p.relative_to(data_root)) for p in actual_set - set(matched))[:8]
        unknown = sorted(k for k in expected if not any(k.replace("\\", "/").endswith("/" + p.relative_to(data_root).as_posix()) or k.replace("\\", "/") == p.relative_to(data_root).as_posix() for p in actual_paths))[:8]
        raise ValueError(f"READY.json file coverage mismatch; unhashed={missing}, unmatched_entries={unknown}")
    hashes: dict[str, str] = {}
    for path, expected_digest in matched.items():
        actual_digest = sha256_file(path)
        if actual_digest != expected_digest:
            raise ValueError(f"READY.json SHA-256 mismatch: {path.relative_to(data_root)}")
        hashes[path.relative_to(data_root).as_posix()] = actual_digest
    required = {f"{ds}/{name}" for ds in DATASETS for name in ("corpus.jsonl", "questions.jsonl", "emb_minilm.f16.npy")}
    if not required.issubset(hashes):
        raise ValueError(f"Missing required data files: {sorted(required - hashes.keys())}")
    return {"ready_sha256": hashlib.sha256(ready_bytes).hexdigest(), "files": hashes}


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sample_split(questions: list[dict[str, Any]], limit: int = MAX_PER_SPLIT) -> list[dict[str, Any]]:
    """Deterministic approximately proportional sampling within question type."""
    if len(questions) <= limit:
        return sorted(questions, key=lambda q: q["qid"])
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        buckets[q["type"]].append(q)
    total = len(questions)
    allocations = {t: min(len(v), max(1, int(limit * len(v) / total))) for t, v in buckets.items()}
    while sum(allocations.values()) > limit:
        t = max((t for t in allocations if allocations[t] > 1), key=lambda x: allocations[x], default=None)
        if t is None:
            break
        allocations[t] -= 1
    while sum(allocations.values()) < limit:
        candidates = [t for t, v in buckets.items() if allocations[t] < len(v)]
        if not candidates:
            break
        t = max(candidates, key=lambda x: limit * len(buckets[x]) / total - allocations[x])
        allocations[t] += 1
    out = []
    for t, bucket in buckets.items():
        ordered = sorted(bucket, key=lambda q: hashlib.sha256(f"{SEED}:{q['qid']}".encode()).hexdigest())
        out.extend(ordered[:allocations[t]])
    return sorted(out, key=lambda q: q["qid"])


def load_questions(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("split") != split:
                continue
            # Do not copy or expose the answer field to any model or feature.
            rows.append({
                "qid": str(obj["qid"]),
                "question": str(obj["question"]),
                "type": str(obj.get("type", "unknown")),
                "gold_pids": [str(x) for x in obj["gold_pids"]],
                "split": split,
            })
    if not rows:
        raise ValueError(f"No {split} rows found in {path}")
    sample = sample_split(rows)
    meta = {
        "population_n": len(rows), "sample_n": len(sample),
        "population_by_type": dict(sorted(Counter(q["type"] for q in rows).items())),
        "sample_by_type": dict(sorted(Counter(q["type"] for q in sample).items())),
    }
    return sample, meta


class Corpus:
    """Memory-map vectors; keep only IDs/byte offsets in RAM and fetch text lazily."""
    def __init__(self, corpus_path: Path, embedding_path: Path):
        self.path = corpus_path
        self.emb = np.load(embedding_path, mmap_mode="r")
        if self.emb.ndim != 2 or self.emb.dtype.kind != "f":
            raise ValueError(f"Expected a floating-point 2D embedding matrix: {embedding_path}")
        self.pids: list[str] = []
        self.offsets: list[int] = []
        self.by_pid: dict[str, int] = {}
        with corpus_path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                row = json.loads(line)
                pid = str(row["pid"])
                if pid in self.by_pid:
                    raise ValueError(f"Duplicate corpus pid: {pid}")
                self.by_pid[pid] = len(self.pids)
                self.pids.append(pid)
                self.offsets.append(offset)
        if len(self.pids) != self.emb.shape[0]:
            raise ValueError(f"Corpus/embedding row mismatch: {len(self.pids)} != {self.emb.shape[0]}")
        self.handle = corpus_path.open("rb")

    def close(self) -> None:
        self.handle.close()

    def fetch(self, index: int) -> dict[str, Any]:
        self.handle.seek(self.offsets[index])
        return json.loads(self.handle.readline())

    def top_candidates(self, query: np.ndarray, limit: int = POOL_SIZE, chunk_rows: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
        best_i = np.empty(0, dtype=np.int64)
        best_s = np.empty(0, dtype=np.float32)
        for start in range(0, self.emb.shape[0], chunk_rows):
            end = min(start + chunk_rows, self.emb.shape[0])
            block = np.asarray(self.emb[start:end], dtype=np.float32)
            scores = block @ query
            take = min(limit, len(scores))
            local = np.argpartition(scores, len(scores) - take)[-take:]
            idx = local.astype(np.int64) + start
            sc = scores[local].astype(np.float32, copy=False)
            all_i = np.concatenate((best_i, idx))
            all_s = np.concatenate((best_s, sc))
            order = np.lexsort((all_i, -all_s))[:limit]
            best_i, best_s = all_i[order], all_s[order]
        order = np.lexsort((best_i, -best_s))
        return best_i[order], best_s[order]


def normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    return vec / denom if denom else vec


def disco_greedy(candidate_indices: np.ndarray, candidate_scores: np.ndarray, token_vectors: np.ndarray, corpus_vectors: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Greedy positive MaxSim coverage over pooled passage vectors (APPROX)."""
    docs = np.asarray(corpus_vectors[candidate_indices], dtype=np.float32)
    sims = np.maximum(token_vectors @ docs.T, 0.0)
    covered = np.zeros(sims.shape[0], dtype=np.float32)
    chosen: list[int] = []
    gains: list[float] = []
    available = np.ones(len(candidate_indices), dtype=bool)
    for _ in range(min(k, len(candidate_indices))):
        marginal = np.maximum(sims - covered[:, None], 0).sum(axis=0)
        marginal[~available] = -np.inf
        local = int(np.argmax(marginal))
        available[local] = False
        covered = np.maximum(covered, sims[:, local])
        chosen.append(int(candidate_indices[local]))
        gains.append(float(marginal[local]))
    return np.asarray(chosen, dtype=np.int64), np.asarray(gains, dtype=np.float32)


def whitespace_tokens(title: str, text: str) -> int:
    return len((title + " " + text).split())


def support_metrics(gold_pids: list[str] | set[str], retrieved_pids: list[str] | set[str]) -> tuple[int, float]:
    gold, retrieved = set(gold_pids), set(retrieved_pids)
    if not gold:
        raise ValueError("gold_pids must not be empty")
    return int(gold.issubset(retrieved)), len(gold & retrieved) / len(gold)


def false_sufficient_rate(labels: np.ndarray, predicted_sufficient: np.ndarray) -> float | None:
    insufficient = np.asarray(labels) == 0
    if not insufficient.any():
        return None
    return float((np.asarray(predicted_sufficient, dtype=bool) & insufficient).sum() / insufficient.sum())


def entity_set(text: str) -> set[str]:
    return {" ".join(x.casefold().split()) for x in ENTITY_RE.findall(text)}


def entity_overlap(passages: list[str]) -> float:
    entities = [entity_set(x) for x in passages]
    pairs = [(entities[i], entities[j]) for i in range(len(entities)) for j in range(i + 1, len(entities))]
    if not pairs:
        return 0.0
    vals = [len(a & b) / len(a | b) if a | b else 0.0 for a, b in pairs]
    return float(np.mean(vals))


def build_features(row: dict[str, Any], vocab: list[str]) -> list[float]:
    scores = np.asarray(row["ce_scores"], dtype=np.float32)
    rank_scores = np.asarray(row["rank_scores"], dtype=np.float32)
    gaps = np.diff(rank_scores)
    qtokens = len(row["question"].split())
    base = [
        float(row["k"]), float(scores[0]), float(scores[-1]), float(scores.min()),
        float(scores.max()), float(scores.mean()), float(scores.std()), float(scores.sum()),
        float(rank_scores[0] - rank_scores[-1]),
        float(rank_scores[-1] - row["next_rank_score"]),
        float(np.abs(gaps).mean()) if len(gaps) else 0.0,
        float(row["entity_overlap"]), math.log1p(row["tokens"]), math.log1p(qtokens),
    ]
    base.extend([1.0 if row["type"] == t else 0.0 for t in vocab])
    base.append(1.0 if row["type"] not in vocab else 0.0)
    return base


def select_threshold(labels: np.ndarray, scores: np.ndarray, precision_target: float = 0.90) -> dict[str, Any]:
    candidates = np.unique(scores[np.isfinite(scores)])
    best = None
    for threshold in candidates:
        pred = scores >= threshold
        positives = int(pred.sum())
        if not positives:
            continue
        precision = float(labels[pred].mean())
        recall = float(labels[pred].sum() / max(1, labels.sum()))
        if precision >= precision_target:
            item = (recall, positives, -float(threshold), float(threshold), precision)
            if best is None or item[:3] > best[:3]:
                best = item
    if best is None:
        return {"threshold": None, "reached": False, "tune_precision": None, "tune_recall": 0.0, "tune_calls": 0}
    return {"threshold": best[3], "reached": True, "tune_precision": best[4], "tune_recall": best[0], "tune_calls": best[1]}


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def bootstrap_ci(values: np.ndarray, stat, seed: int = BOOTSTRAP_SEED) -> tuple[list[float] | None, int]:
    values = np.asarray(values)
    if not len(values):
        return None, 0
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAPS):
        idx = rng.integers(0, len(values), size=len(values))
        val = stat(values[idx])
        if val is not None and np.isfinite(val):
            samples.append(float(val))
    if not samples:
        return None, 0
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))], len(samples)


def mean_with_ci(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    ci, valid = bootstrap_ci(x, lambda a: float(np.mean(a)))
    return {"mean": float(np.mean(x)) if len(x) else None, "ci95": ci, "n": int(len(x)), "bootstrap_valid": valid}


def ratio_with_ci(numer: np.ndarray, denom: np.ndarray) -> dict[str, Any]:
    pair = np.column_stack((numer, denom)).astype(float)
    def stat(a: np.ndarray) -> float | None:
        d = a[:, 1].sum()
        return float(a[:, 0].sum() / d) if d else None
    ci, valid = bootstrap_ci(pair, stat)
    value = stat(pair)
    return {"mean": value, "ci95": ci, "n": int(len(pair)), "bootstrap_valid": valid}


def auc_with_ci(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    pair = np.column_stack((labels, scores))
    ci, valid = bootstrap_ci(pair, lambda a: safe_auc(a[:, 0].astype(int), a[:, 1]))
    return {"mean": safe_auc(labels, scores), "ci95": ci, "n": int(len(labels)), "bootstrap_valid": valid}


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None) -> dict[str, Any]:
    pred = scores >= threshold if threshold is not None else np.zeros(len(labels), dtype=bool)
    tp = pred & (labels == 1)
    fp = pred & (labels == 0)
    result = {
        "auroc": auc_with_ci(labels, scores),
        "precision": ratio_with_ci(tp.astype(float), pred.astype(float)),
        "recall": ratio_with_ci(tp.astype(float), (labels == 1).astype(float)),
        "fsr": ratio_with_ci(fp.astype(float), (labels == 0).astype(float)),
        "predicted_sufficient_rate": mean_with_ci(pred.astype(float)),
    }
    return result


def paired_difference_ci(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    return mean_with_ci(x)


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def download_repo(repo_id: str, local_dir: Path, hf_transfer: bool, stall_seconds: int = STALL_SECONDS) -> dict[str, Any]:
    local_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(BASE / ".hf-cache")
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1" if hf_transfer else "0"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo_id!r}, local_dir={str(local_dir)!r}, token=False)"
    )
    log_path = BASE / ".download_attempts.log"
    start = time.monotonic()
    before = dir_bytes(local_dir)
    last_bytes, last_growth = before, start
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"{timestamp()} START repo={repo_id} hf_transfer={hf_transfer}\n")
        log.flush()
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=log, stderr=log, env=env)
        stalled = False
        while proc.poll() is None:
            time.sleep(10)
            size = dir_bytes(local_dir)
            if size > last_bytes:
                last_bytes, last_growth = size, time.monotonic()
            elif time.monotonic() - last_growth >= stall_seconds:
                stalled = True
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
        code_exit = proc.wait()
        status = "stalled" if stalled else ("ok" if code_exit == 0 else "failed")
        log.write(f"{timestamp()} END repo={repo_id} status={status} exit={code_exit} bytes={dir_bytes(local_dir)} seconds={time.monotonic()-start:.1f}\n")
    return {"repo": repo_id, "hf_transfer": hf_transfer, "status": status, "exit_code": code_exit, "bytes": dir_bytes(local_dir), "seconds": round(time.monotonic() - start, 1)}


def ensure_model(repo_id: str, folder: str, attempts: list[dict[str, Any]], cross_encoder: bool = False) -> Path:
    local = MODELS / folder
    if (local / "config.json").is_file() and any((local / x).is_file() for x in ("model.safetensors", "pytorch_model.bin")):
        if folder == "nli-deberta-v3-small" and dir_bytes(local) > MAX_NLI_BYTES:
            raise RuntimeError(f"NLI model exceeds 600 MB: {dir_bytes(local)} bytes")
        total_downloads = dir_bytes(MODELS) + dir_bytes(BASE / ".uv-cache")
        if total_downloads > MAX_DOWNLOAD_BYTES:
            raise RuntimeError(f"Model and dependency downloads exceed 1 GB: {total_downloads} bytes")
        return local
    modes = [True, False] if cross_encoder else [False]
    for mode in modes:
        result = download_repo(repo_id, local, mode)
        attempts.append(result)
        if result["status"] == "ok":
            break
    else:
        if not cross_encoder:
            raise RuntimeError(f"Model download failed: {repo_id}; attempts={attempts}")
        fallback = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        fallback_dir = MODELS / "ms-marco-MiniLM-L-12-v2"
        result = download_repo(fallback, fallback_dir, False)
        attempts.append(result)
        if result["status"] != "ok":
            raise RuntimeError(f"Primary and fallback cross-encoder downloads failed: {attempts}")
        local = fallback_dir
    if folder == "nli-deberta-v3-small" and dir_bytes(local) > MAX_NLI_BYTES:
        raise RuntimeError(f"NLI model exceeds 600 MB: {dir_bytes(local)} bytes")
    total_downloads = dir_bytes(MODELS) + dir_bytes(BASE / ".uv-cache")
    if total_downloads > MAX_DOWNLOAD_BYTES:
        raise RuntimeError(f"Model and dependency downloads exceed 1 GB: {total_downloads} bytes")
    return local


def load_encoder(path: Path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).to("cpu").eval()
    return tok, model


def encode_questions(questions: list[dict[str, Any]], tok, model) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    started = time.perf_counter()
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for start in range(0, len(questions), 32):
        batchq = questions[start:start + 32]
        batch = tok([q["question"] for q in batchq], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
        masks = batch["attention_mask"]
        for j, q in enumerate(batchq):
            mask = masks[j].bool()
            pooled = normalize(hidden[j][mask].mean(dim=0).cpu().numpy().astype(np.float32))
            ids = batch["input_ids"][j][mask].tolist()
            toks = tok.convert_ids_to_tokens(ids)
            keep = [n for n, token in enumerate(toks) if token not in tok.all_special_tokens and token.replace("##", "").casefold() not in QUERY_STOP]
            if not keep:
                keep = list(range(len(ids)))
            token_vecs = hidden[j][mask][keep].cpu().numpy().astype(np.float32)
            token_vecs = np.stack([normalize(x) for x in token_vecs])
            out[q["qid"]] = (pooled, token_vecs)
    return out, (time.perf_counter() - started) * 1000 / max(1, len(questions))


def predict_pairs(text_pairs: list[tuple[str, str]], tok, model, max_length: int) -> tuple[list[float], float]:
    if not text_pairs:
        return [], 0.0
    start = time.perf_counter()
    scores: list[float] = []
    for offset in range(0, len(text_pairs), 32):
        chunk = text_pairs[offset:offset + 32]
        batch = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**batch).logits
        if logits.shape[-1] != 1:
            raise ValueError(f"Expected scalar relevance logits, got {tuple(logits.shape)}")
        scores.extend(float(x) for x in logits[:, 0].cpu().tolist())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return scores, elapsed_ms / len(text_pairs)


def predict_nli(pairs: list[tuple[str, str]], tok, model) -> tuple[list[float], float]:
    if not pairs:
        return [], 0.0
    entailment = next((int(i) for i, label in model.config.id2label.items() if "entail" in str(label).casefold()), None)
    if entailment is None:
        raise ValueError(f"NLI checkpoint has no entailment label: {model.config.id2label}")
    start = time.perf_counter()
    result: list[float] = []
    for offset in range(0, len(pairs), 16):
        chunk = pairs[offset:offset + 16]
        batch = tok([a for a, _ in chunk], [b for _, b in chunk], padding=True, truncation=True, max_length=NLI_MAX_LENGTH, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)[:, entailment]
        result.extend(float(x) for x in probs.cpu().tolist())
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms / len(pairs)


def prepare_split(dataset: str, split: str, corpus: Corpus, questions: list[dict[str, Any]], encoder_tok, encoder, ce_tok, ce_model, nli_tok, nli_model, feature_vocab: list[str] | None = None) -> tuple[list[dict[str, Any]], list[str], float]:
    if feature_vocab is None:
        feature_vocab = sorted({q["type"] for q in questions})
    query_vecs, encode_ms = encode_questions(questions, encoder_tok, encoder)
    retrieval_start = time.perf_counter()
    query_work = []
    for q in questions:
        pooled, token_vecs = query_vecs[q["qid"]]
        qstart = time.perf_counter()
        cand_i, cand_s = corpus.top_candidates(pooled)
        dense_ms = (time.perf_counter() - qstart) * 1000
        dstart = time.perf_counter()
        disco_i, disco_g = disco_greedy(cand_i, cand_s, token_vecs, corpus.emb, 10)
        disco_ms = (time.perf_counter() - dstart) * 1000
        dense10_i, dense10_s = cand_i[:10], cand_s[:10]
        query_work.append({"q": q, "cand_i": cand_i, "cand_s": cand_s, "dense_i": dense10_i, "dense_s": dense10_s, "disco_i": disco_i, "disco_g": disco_g, "pool": pooled, "tokens": token_vecs, "dense_ms": dense_ms, "disco_ms": disco_ms})
    retrieval_ms_total = (time.perf_counter() - retrieval_start) * 1000
    _ = retrieval_start, retrieval_ms_total

    ce_jobs: list[tuple[str, str, str, str]] = []
    texts: dict[tuple[str, str], dict[str, Any]] = {}
    for item in query_work:
        q = item["q"]
        for ret, indices in (("dense", item["dense_i"]), ("disco_approx", item["disco_i"])):
            for idx in indices:
                doc = corpus.fetch(int(idx))
                pid = str(doc["pid"])
                texts[(q["qid"], pid)] = doc
                ce_jobs.append((ret, q["qid"], pid, str(doc.get("title", "")) + " " + str(doc.get("text", ""))))
    question_by_id = {item["q"]["qid"]: item["q"]["question"] for item in query_work}
    ce_pairs = [(question_by_id[qid], text) for _, qid, _, text in ce_jobs]
    ce_values, ce_pair_ms = predict_pairs(ce_pairs, ce_tok, ce_model, MAX_LENGTH)
    ce_by: dict[tuple[str, str, str], float] = {}
    for job, value in zip(ce_jobs, ce_values):
        ret, qid, pid, _ = job
        ce_by[(ret, qid, pid)] = value

    rows: list[dict[str, Any]] = []
    nli_jobs: list[tuple[int, str, str]] = []
    for item in query_work:
        q = item["q"]
        qid = q["qid"]
        gold = set(q["gold_pids"])
        if not gold:
            raise ValueError(f"Empty gold_pids for {qid}")
        for ret in RETRIEVERS:
            indices = item["dense_i"] if ret == "dense" else item["disco_i"]
            rank_scores = item["dense_s"] if ret == "dense" else item["disco_g"]
            pids = [corpus.pids[int(i)] for i in indices]
            docs = [corpus.fetch(int(i)) for i in indices]
            passages = [str(d.get("title", "")) + " " + str(d.get("text", "")) for d in docs]
            ce_scores = [ce_by[(ret, qid, pid)] for pid in pids]
            tokens_by_k: dict[int, int] = {}
            for k in K_VALUES:
                if len(pids) < k:
                    continue
                chosen = pids[:k]
                label, recall = support_metrics(gold, chosen)
                token_count = sum(whitespace_tokens(str(docs[j].get("title", "")), str(docs[j].get("text", ""))) for j in range(k))
                tokens_by_k[k] = token_count
                row = {
                    "dataset": dataset, "split": split, "qid": qid, "type": q["type"], "question": q["question"],
                    "retriever": ret, "k": k, "label": label, "full_support": float(label), "sp_recall": float(recall),
                    "tokens": int(token_count), "gold_n": len(gold), "retrieved_pids": chosen,
                    "ce_scores": ce_scores[:k], "rank_scores": rank_scores[:k],
                    "next_rank_score": float(rank_scores[k]) if len(rank_scores) > k else float(rank_scores[k - 1]),
                    "entity_overlap": entity_overlap(passages[:k]),
                    "retrieval_ms": float(encode_ms + item["dense_ms"] + (item["disco_ms"] if ret == "disco_approx" else 0.0)),
                    "ce_pair_ms": float(ce_pair_ms), "nli_pair_ms": 0.0,
                }
                row["features"] = build_features(row, feature_vocab)
                premise = "\n\n".join(passages[:k])
                hyp = f"The question '{q['question'].rstrip('?')}' is answerable from the provided passages."
                nli_jobs.append((len(rows), premise, hyp))
                rows.append(row)
    nli_pairs = [(premise, hypothesis) for _, premise, hypothesis in nli_jobs]
    nli_values, nli_pair_ms = predict_nli(nli_pairs, nli_tok, nli_model)
    for (row_idx, _, _), value in zip(nli_jobs, nli_values):
        rows[row_idx]["nli_score"] = float(value)
        rows[row_idx]["nli_pair_ms"] = float(nli_pair_ms)
    # Cross-encoder predictions for each retrieved prefix are already represented by min/sum summaries.
    return rows, feature_vocab, float(np.mean([r["retrieval_ms"] for r in rows]))


def fit_logistic(rows: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray([r["features"] for r in rows], dtype=np.float64)
    y = np.asarray([r["label"] for r in rows], dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        p = np.full(len(y), float(classes[0]), dtype=float)
        return p, {"constant": float(classes[0]), "n": len(y)}
    min_class = int(counts.min())
    oof = np.zeros(len(y), dtype=float)
    folds = min(5, min_class)
    if folds >= 2:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        for train_idx, test_idx in cv.split(x, y):
            scaler = StandardScaler().fit(x[train_idx])
            model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)
            model.fit(scaler.transform(x[train_idx]), y[train_idx])
            oof[test_idx] = model.predict_proba(scaler.transform(x[test_idx]))[:, list(model.classes_).index(1)]
    else:
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED).fit(scaler.transform(x), y)
        oof = model.predict_proba(scaler.transform(x))[:, list(model.classes_).index(1)]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED).fit(scaler.transform(x), y)
    final = {"constant": None, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(), "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "classes": model.classes_.tolist(), "n": len(y), "oof_folds": folds}
    return oof, final


def logistic_predict(rows: list[dict[str, Any]], model: dict[str, Any]) -> np.ndarray:
    if model["constant"] is not None:
        return np.full(len(rows), float(model["constant"]))
    x = np.asarray([r["features"] for r in rows], dtype=float)
    z = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(model["coef"]) + model["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def row_scores(row: dict[str, Any]) -> dict[str, float]:
    return {"min_score": float(min(row["ce_scores"])), "sum_score": float(sum(row["ce_scores"])), "nli": float(row["nli_score"])}


def fit_policies(tune_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policies: dict[str, Any] = {}
    for dataset in DATASETS:
        for ret in RETRIEVERS:
            for k in K_VALUES:
                group = [r for r in tune_rows if r["dataset"] == dataset and r["retriever"] == ret and r["k"] == k]
                if not group:
                    continue
                y = np.asarray([r["label"] for r in group], dtype=int)
                base_scores = {name: np.asarray([row_scores(r)[name] for r in group]) for name in ("min_score", "sum_score", "nli")}
                oof, lr = fit_logistic(group)
                pred_start = time.perf_counter()
                logistic_predict(group, lr)
                lr_ms = (time.perf_counter() - pred_start) * 1000 / max(1, len(group))
                base_scores["logistic"] = oof
                key = f"{dataset}|{ret}|{k}"
                policies[key] = {"logistic_model": lr, "thresholds": {name: select_threshold(y, scores) for name, scores in base_scores.items()}}
                for r, score in zip(group, oof):
                    r["detector_scores"] = {**row_scores(r), "logistic": float(score)}
                    r["logistic_ms"] = lr_ms
    return policies, tune_rows


def add_holdout_scores(rows: list[dict[str, Any]], policies: dict[str, Any]) -> None:
    for dataset in DATASETS:
        for ret in RETRIEVERS:
            for k in K_VALUES:
                group = [r for r in rows if r["dataset"] == dataset and r["retriever"] == ret and r["k"] == k]
                if not group:
                    continue
                model = policies[f"{dataset}|{ret}|{k}"]["logistic_model"]
                pred_start = time.perf_counter()
                lp = logistic_predict(group, model)
                lr_ms = (time.perf_counter() - pred_start) * 1000 / max(1, len(group))
                for r, score in zip(group, lp):
                    r["detector_scores"] = {**row_scores(r), "logistic": float(score)}
                    r["logistic_ms"] = lr_ms


def groups(rows: list[dict[str, Any]], keys: tuple[str, ...]):
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def metric_summary(rows: list[dict[str, Any]], policies: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for (dataset, split, ret, k), group in groups(rows, ("dataset", "split", "retriever", "k")).items():
        typed = [(key[0], value) for key, value in groups(group, ("type",)).items()]
        type_groups = [("ALL", group)] + sorted(typed, key=lambda x: x[0])
        for typ, subset in type_groups:
            labels = np.asarray([r["label"] for r in subset], dtype=int)
            entry = {"dataset": dataset, "split": split, "retriever": ret, "k": k, "type": typ, "n": len(subset), "positives": int(labels.sum()), "full_support": mean_with_ci(np.asarray([r["full_support"] for r in subset])), "sp_recall": mean_with_ci(np.asarray([r["sp_recall"] for r in subset])), "evidence_tokens": mean_with_ci(np.asarray([r["tokens"] for r in subset])), "retrieval_ms_query": mean_with_ci(np.asarray([r["retrieval_ms"] for r in subset]))}
            for detector in DETECTORS:
                scores = np.asarray([r["detector_scores"][detector] for r in subset], dtype=float)
                threshold = policies[f"{dataset}|{ret}|{k}"]["thresholds"][detector]["threshold"]
                cls = classification_metrics(labels, scores, threshold)
                latency = []
                for r in subset:
                    if detector in ("min_score", "sum_score", "logistic"):
                        detector_ms = r["ce_pair_ms"] * k
                        if detector == "logistic":
                            detector_ms += r.get("logistic_ms", 0.0)
                    else:
                        detector_ms = r["nli_pair_ms"]
                    latency.append(r["retrieval_ms"] + detector_ms)
                entry[detector] = {**cls, "threshold": threshold, "tune_threshold_reached": policies[f"{dataset}|{ret}|{k}"]["thresholds"][detector]["reached"], "latency_ms_query": mean_with_ci(np.asarray(latency))}
            output.append(entry)
    return output


def adaptive_by_q(rows: list[dict[str, Any]], policies: dict[str, Any], dataset: str, ret: str, detector: str) -> dict[str, dict[str, Any]]:
    result = {}
    subset = [r for r in rows if r["dataset"] == dataset and r["retriever"] == ret]
    for qid, qrows in groups(subset, ("qid",)).items():
        qrows = sorted(qrows, key=lambda r: r["k"])
        chosen = qrows[-1]
        called = False
        for r in qrows:
            threshold = policies[f"{dataset}|{ret}|{r['k']}"]["thresholds"][detector]["threshold"]
            score = r["detector_scores"][detector]
            if threshold is not None and score >= threshold:
                chosen, called = r, True
                break
        result[qid[0]] = {"row": chosen, "called": called}
    return result


def cost_match(adapt: dict[str, dict[str, Any]], fixed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    qids = sorted(adapt)
    target_cost = float(np.mean([adapt[q]["row"]["tokens"] for q in qids]))
    by_q_k = {(r["qid"], r["k"]): r for r in fixed_rows}
    fixed_means = {k: float(np.mean([by_q_k[(q, k)]["tokens"] for q in qids])) for k in K_VALUES}
    low = max((k for k in K_VALUES if fixed_means[k] <= target_cost), default=K_VALUES[0])
    high = min((k for k in K_VALUES if fixed_means[k] >= target_cost), default=K_VALUES[-1])
    if low == high:
        weight = 0.0
    else:
        weight = (target_cost - fixed_means[low]) / (fixed_means[high] - fixed_means[low])
    adapt_fs, fixed_fs, cost_diff = [], [], []
    for q in qids:
        ar = adapt[q]["row"]
        lo, hi = by_q_k[(q, low)], by_q_k[(q, high)]
        fixed_fs.append((1 - weight) * lo["full_support"] + weight * hi["full_support"])
        adapt_fs.append(ar["full_support"])
        cost_diff.append(ar["tokens"] - ((1 - weight) * lo["tokens"] + weight * hi["tokens"]))
    diff = np.asarray(adapt_fs) - np.asarray(fixed_fs)
    diff_ci = paired_difference_ci(diff)
    cost_ci = paired_difference_ci(np.asarray(cost_diff, dtype=float))
    return {"n": len(qids), "adaptive_mean_tokens": target_cost, "fixed_mix": {"k_low": low, "k_high": high, "high_weight": float(weight)}, "matched_fixed_mean_tokens": float(np.mean([fixed_means[low] * (1 - weight) + fixed_means[high] * weight])), "token_gap_adaptive_minus_fixed": cost_ci, "adaptive_full_support": mean_with_ci(np.asarray(adapt_fs)), "matched_fixed_full_support": mean_with_ci(np.asarray(fixed_fs)), "delta_full_support": diff_ci}


def adaptive_metrics(rows: list[dict[str, Any]], policies: dict[str, Any], dataset: str, ret: str, detector: str) -> dict[str, Any]:
    adapt = adaptive_by_q(rows, policies, dataset, ret, detector)
    chosen = [v["row"] for v in adapt.values()]
    labels = np.asarray([r["label"] for r in chosen], dtype=int)
    calls = np.asarray([v["called"] for v in adapt.values()], dtype=bool)
    fsr = ratio_with_ci((calls & (labels == 0)).astype(float), (labels == 0).astype(float))
    precision = ratio_with_ci((calls & (labels == 1)).astype(float), calls.astype(float))
    fixed_rows = [r for r in rows if r["dataset"] == dataset and r["retriever"] == ret]
    cost = cost_match(adapt, fixed_rows)
    return {"dataset": dataset, "retriever": ret, "detector": detector, "n": len(chosen), "adaptive_full_support": mean_with_ci(np.asarray([r["full_support"] for r in chosen])), "sp_recall": mean_with_ci(np.asarray([r["sp_recall"] for r in chosen])), "evidence_tokens": mean_with_ci(np.asarray([r["tokens"] for r in chosen])), "fsr": fsr, "precision": precision, "predicted_sufficient_rate": mean_with_ci(calls.astype(float)), "cost_matched_fixed": cost}


def choose_gate_policy(tune_rows: list[dict[str, Any]], policies: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for ret in RETRIEVERS:
        for detector in DETECTORS:
            curve = adaptive_metrics(tune_rows, policies, "hotpotqa", ret, detector)
            latency = float(np.mean([r["retrieval_ms"] + (r["nli_pair_ms"] if detector == "nli" else r["ce_pair_ms"] * r["k"]) for r in tune_rows if r["dataset"] == "hotpotqa" and r["retriever"] == ret]))
            candidates.append({"retriever": ret, "detector": detector, "delta": curve["cost_matched_fixed"]["delta_full_support"]["mean"], "latency_ms_query": latency, "tune_curve": curve})
    candidates.sort(key=lambda x: (-(x["delta"] if x["delta"] is not None else -1e9), x["latency_ms_query"], x["retriever"], x["detector"]))
    if not candidates:
        raise ValueError("No HotpotQA tune adaptive candidates")
    return candidates[0]


def collect_metrics(rows: list[dict[str, Any]], policies: dict[str, Any]) -> dict[str, Any]:
    return {"fixed": metric_summary(rows, policies), "adaptive": [adaptive_metrics(rows, policies, ds, ret, det) for ds in DATASETS for ret in RETRIEVERS for det in DETECTORS]}


def load_models(attempts: list[dict[str, Any]]):
    MODELS.mkdir(parents=True, exist_ok=True)
    encoder_dir = ensure_model("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2", attempts)
    ce_dir = ensure_model("cross-encoder/ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-6-v2", attempts, cross_encoder=True)
    nli_dir = ensure_model("cross-encoder/nli-deberta-v3-small", "nli-deberta-v3-small", attempts)
    encoder_tok, encoder = load_encoder(encoder_dir)
    ce_tok = AutoTokenizer.from_pretrained(ce_dir, local_files_only=True)
    ce_model = AutoModelForSequenceClassification.from_pretrained(ce_dir, local_files_only=True).to("cpu").eval()
    nli_tok = AutoTokenizer.from_pretrained(nli_dir, local_files_only=True)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_dir, local_files_only=True).to("cpu").eval()
    ce_id = "cross-encoder/ms-marco-MiniLM-L-12-v2" if ce_dir.name.endswith("L-12-v2") else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    return (encoder_tok, encoder, ce_tok, ce_model, nli_tok, nli_model), {"encoder": "sentence-transformers/all-MiniLM-L6-v2", "cross_encoder": ce_id, "nli": "cross-encoder/nli-deberta-v3-small", "local_model_bytes": dir_bytes(MODELS)}


def save_json(path: Path, obj: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    started = time.monotonic()
    ready = verify_ready()
    output_files = [BASE / "results.json", BASE / "RESULT.md", BASE / "manifest.sha256", BASE / "holdout_runs.log", BASE / "frozen_config.json"]
    existing = [str(p) for p in output_files if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing H3 outputs: {existing}")
    attempts: list[dict[str, Any]] = []
    models, model_info = load_models(attempts)
    encoder_tok, encoder, ce_tok, ce_model, nli_tok, nli_model = models
    data_meta = {"ready": ready, "sample_cap_per_split": MAX_PER_SPLIT, "sample_seed": SEED}
    all_tune: list[dict[str, Any]] = []
    vocab_by_ds = {}
    populations = {}
    corpora = {}
    for dsname in DATASETS:
        root = DATA / dsname
        corpus = Corpus(root / "corpus.jsonl", root / "emb_minilm.f16.npy")
        corpora[dsname] = corpus
        questions, qmeta = load_questions(root / "questions.jsonl", "tune")
        populations[f"{dsname}:tune"] = qmeta
        rows, vocab, _ = prepare_split(dsname, "tune", corpus, questions, encoder_tok, encoder, ce_tok, ce_model, nli_tok, nli_model)
        vocab_by_ds[dsname] = vocab
        all_tune.extend(rows)
        print(f"TUNE_READY dataset={dsname} n={len(questions)} rows={len(rows)}", flush=True)
    policies, all_tune = fit_policies(all_tune)
    gate_choice = choose_gate_policy(all_tune, policies)
    tune_results = collect_metrics(all_tune, policies)
    frozen = {
        "protocol_sha256": sha256_file(BASE / "protocol.md"), "data_ready_sha256": ready["ready_sha256"],
        "sample_cap_per_split": MAX_PER_SPLIT, "sample_seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED, "models": model_info, "model_download_attempts": attempts,
        "feature_vocab": vocab_by_ds, "policies": policies, "gate_policy": {k: v for k, v in gate_choice.items() if k != "tune_curve"},
        "cpu_threads": 3, "retrievers": list(RETRIEVERS), "detectors": list(DETECTORS), "k_values": list(K_VALUES),
    }
    save_json(BASE / "frozen_config.json", frozen)
    holdout_log = BASE / "holdout_runs.log"
    if holdout_log.exists():
        raise FileExistsError(f"Refusing a second holdout run: {holdout_log}")
    config_hash = sha256_file(BASE / "frozen_config.json")
    with holdout_log.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"time": timestamp(), "status": "STARTED", "frozen_config_sha256": config_hash, "ready_sha256": ready["ready_sha256"]}) + "\n")
        f.flush()
        os.fsync(f.fileno())
    holdout_rows: list[dict[str, Any]] = []
    try:
        for dsname in DATASETS:
            root = DATA / dsname
            questions, qmeta = load_questions(root / "questions.jsonl", "holdout")
            populations[f"{dsname}:holdout"] = qmeta
            rows, _, _ = prepare_split(dsname, "holdout", corpora[dsname], questions, encoder_tok, encoder, ce_tok, ce_model, nli_tok, nli_model, vocab_by_ds[dsname])
            add_holdout_scores(rows, policies)
            holdout_rows.extend(rows)
            print(f"HOLDOUT_DONE dataset={dsname} n={len(questions)} rows={len(rows)}", flush=True)
        holdout_results = collect_metrics(holdout_rows, policies)
        with holdout_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": timestamp(), "status": "COMPLETED", "n_rows": len(holdout_rows)}) + "\n")
    except BaseException as exc:
        with holdout_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time": timestamp(), "status": "FAILED", "error": type(exc).__name__}) + "\n")
        raise
    gate_eval = adaptive_metrics(holdout_rows, policies, "hotpotqa", gate_choice["retriever"], gate_choice["detector"])
    hp_rows = [r for r in holdout_rows if r["dataset"] == "hotpotqa"]
    all_auc = []
    for (ret, k), group in groups(hp_rows, ("retriever", "k")).items():
        for detector in DETECTORS:
            labels = np.asarray([r["label"] for r in group], dtype=int)
            scores = np.asarray([r["detector_scores"][detector] for r in group], dtype=float)
            all_auc.append({"retriever": ret, "k": k, "detector": detector, "auroc": auc_with_ci(labels, scores)})
    valid_auc = [x for x in all_auc if x["auroc"]["mean"] is not None]
    best_auc = max(valid_auc, key=lambda x: x["auroc"]["mean"]) if valid_auc else None
    s1_pass = bool(best_auc and best_auc["auroc"]["mean"] >= .80)
    delta = gate_eval["cost_matched_fixed"]["delta_full_support"]
    delta_ci = delta["ci95"]
    s2_pass = bool(delta["mean"] is not None and delta["mean"] >= .03 and delta_ci and (delta_ci[0] > 0 or delta_ci[1] < 0))
    # A positive improvement requires the entire interval above zero; a negative interval cannot pass.
    s2_pass = bool(delta["mean"] is not None and delta["mean"] >= .03 and delta_ci and delta_ci[0] > 0)
    results = {
        "data": data_meta, "populations": populations, "models": model_info, "download_attempts": attempts,
        "frozen_config_sha256": config_hash, "tune": tune_results, "holdout": holdout_results,
        "gate": {"S1_prime": {"pass": s1_pass, "best_nonleaking_detector": best_auc, "all_holdout_auc": all_auc, "threshold": .80},
                 "S2_prime": {"pass": s2_pass, "selected_on_tune": {k: v for k, v in gate_choice.items() if k != "tune_curve"}, "holdout": gate_eval, "threshold_delta_full_support": .03}},
        "runtime_seconds": round(time.monotonic() - started, 2),
        "limitations": ["Question samples capped at 200 per dataset/split.", "DISCo-style APPROX uses pooled passage vectors and an exact dense top-256 candidate pool; it is not the DISCo index."],
    }
    save_json(BASE / "results.json", results)
    verdict = "useful detector" if s1_pass else "cheap detectors insufficient"
    delta_txt = "NA" if delta["mean"] is None else f"{delta['mean']:+.3f} [{delta_ci[0]:+.3f}, {delta_ci[1]:+.3f}], n={delta['n']}"
    auc_txt = "NA" if best_auc is None else f"{best_auc['auroc']['mean']:.3f} {best_auc['auroc']['ci95']} n={best_auc['auroc']['n']} ({best_auc['retriever']}, k={best_auc['k']}, {best_auc['detector']})"
    report = (f"# H3 result\n\nS1′: **{'PASS' if s1_pass else 'FAIL'}** — {verdict}; best HotpotQA holdout AUROC {auc_txt}.\n\n"
              f"S2′: **{'PASS' if s2_pass else 'FAIL'}** — tune-selected {gate_choice['retriever']} / {gate_choice['detector']}; matched-token FULL-SUPPORT change {delta_txt}. "
              f"Holdout FSR {gate_eval['fsr']['mean']} (95% CI {gate_eval['fsr']['ci95']}).\n\n"
              f"Scope: at most {MAX_PER_SPLIT} tune and holdout questions per dataset; DISCo-style APPROX is pooled-vector greedy coverage over dense top-256. Runtime {results['runtime_seconds']:.0f}s. Full metrics and subgroup CIs: `results.json`.\n")
    if (BASE / "RESULT.md").exists():
        raise FileExistsError("Refusing to overwrite RESULT.md")
    (BASE / "RESULT.md").write_text(report, encoding="utf-8")
    manifest_paths = [BASE / "protocol.md", BASE / "requirements.txt", BASE / "h3.py", BASE / "wait_ready.py", BASE / "test_h3.py", BASE / "frozen_config.json", BASE / "holdout_runs.log", BASE / "results.json", BASE / "RESULT.md"]
    lines = [f"{sha256_file(p)}  {p.name}" for p in manifest_paths if p.is_file()]
    (BASE / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for corpus in corpora.values():
        corpus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
