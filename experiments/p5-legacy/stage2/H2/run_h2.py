from __future__ import annotations

import os

THREADS = 4
os.environ["OMP_NUM_THREADS"] = str(THREADS)
os.environ["MKL_NUM_THREADS"] = str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(THREADS)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

import argparse
import copy
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATASETS = ("hotpotqa", "2wiki", "musique")
MAX_TRAIN_QUESTIONS = 8000
BATCH_SIZE = 64
MAX_LENGTH = 192
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.05
SEED = 0
MAX_K = 4
BEAM_WIDTH = 5
DISCO_SHORTLIST = 100
BOOTSTRAP_RESAMPLES = 1000
COMPARATORS = (
    "bm25",
    "frozen_dense",
    "trained_single_shot",
    "frozen_mmr",
    "disco_cost_norm_approx",
    "disco_no_cost_norm_approx",
    "frozen_iterative_conditioning",
)
METHODS = COMPARATORS + ("trained_mdr",)

HF_HOME = ROOT / ".hf-cache"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_HOME / "hub")
DEADLINE_EPOCH = float(os.environ.get("H2_DEADLINE_EPOCH", "0"))


def check_deadline() -> None:
    if DEADLINE_EPOCH and time.time() >= DEADLINE_EPOCH:
        raise TimeoutError("registered 100-minute task wall limit reached")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_topk(scores: np.ndarray, k: int) -> list[int]:
    """Return score-descending indices, resolving ties by corpus row order."""
    values = np.asarray(scores).reshape(-1)
    if k <= 0 or values.size == 0:
        return []
    k = min(k, values.size)
    if k == values.size:
        candidates = np.arange(values.size)
    else:
        cutoff = np.partition(values, values.size - k)[values.size - k]
        above = np.flatnonzero(values > cutoff)
        tied = np.flatnonzero(values == cutoff)[: max(0, k - above.size)]
        candidates = np.concatenate((above, tied))
    return sorted(candidates.tolist(), key=lambda i: (-float(values[i]), int(i)))[:k]


def metric_values(selected: list[int], gold: set[int], texts: list[str], k: int) -> dict[str, float]:
    chosen = set(selected[:k])
    return {
        "full_support": float(gold <= chosen),
        "sp_recall": len(gold & chosen) / max(1, len(gold)),
        "evidence_tokens": float(sum(len(texts[i].split()) for i in selected[:k])),
    }


def paired_bootstrap(
    left: Iterable[float], right: Iterable[float], resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> dict[str, float | int]:
    """Paired percentile CI for mean(left - right), resampling question rows."""
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        raise ValueError("paired bootstrap inputs must be non-empty, equal-length vectors")
    delta = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, delta.size, size=(resamples, delta.size))
    means = delta[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(delta.size),
        "difference": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _ready_entries(value: Any, prefix: tuple[str, ...] = ()) -> dict[str, str]:
    """Read common READY.json path->digest and list-of-file-record encodings."""
    found: dict[str, str] = {}
    if isinstance(value, dict):
        path = value.get("path", value.get("file", value.get("name", value.get("relative_path"))))
        digest = value.get("sha256", value.get("digest"))
        if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
            found[path] = digest.lower()
        for key, child in value.items():
            if key in {"sha256", "digest", "path", "file", "name", "relative_path", "generated_at", "created_at"}:
                continue
            if isinstance(child, str) and len(child) == 64 and all(c in "0123456789abcdefABCDEF" for c in child):
                found["/".join(prefix + (key,))] = child.lower()
            elif isinstance(child, dict):
                child_digest = child.get("sha256", child.get("digest"))
                if isinstance(child_digest, str) and len(child_digest) == 64:
                    found["/".join(prefix + (key,))] = child_digest.lower()
                found.update(_ready_entries(child, prefix + (key,)))
            elif isinstance(child, list):
                found.update(_ready_entries(child, prefix + (key,)))
    elif isinstance(value, list):
        for child in value:
            found.update(_ready_entries(child, prefix))
    return found


def verify_ready() -> tuple[dict[str, str], str]:
    ready_path = DATA_ROOT / "READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"waiting for {ready_path}")
    ready = json.loads(ready_path.read_text())
    entries = _ready_entries(ready)
    required = [
        DATA_ROOT / ds / filename
        for ds in DATASETS
        for filename in ("corpus.jsonl", "questions.jsonl", "emb_minilm.f16.npy")
    ]
    resolved: dict[str, str] = {}
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"READY exists but required file is missing: {path}")
        matches = [
            digest for key, digest in entries.items()
            if Path(key).as_posix().endswith(f"{path.parent.name}/{path.name}")
            or Path(key).name == path.name and path.parent.name in Path(key).parts
        ]
        if not matches:
            raise ValueError(f"READY.json has no digest entry for {path.relative_to(DATA_ROOT)}")
        actual = sha256_file(path)
        if actual not in matches:
            raise ValueError(f"READY digest mismatch for {path.relative_to(DATA_ROOT)}")
        resolved[str(path.relative_to(DATA_ROOT))] = actual
    return resolved, sha256_file(ready_path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: expected an object")
                rows.append(row)
    return rows


@dataclass
class Corpus:
    dataset: str
    rows: list[dict[str, Any]]
    pids: list[str]
    pid_to_index: dict[str, int]
    texts: list[str]
    encoder_texts: list[str]
    base_embeddings: np.ndarray
    corpus_sha256: str


def load_corpus(dataset: str) -> Corpus:
    folder = DATA_ROOT / dataset
    rows = read_jsonl(folder / "corpus.jsonl")
    pids = [str(row["pid"]) for row in rows]
    if len(pids) != len(set(pids)):
        raise ValueError(f"duplicate corpus pid in {dataset}")
    for row in rows:
        if not isinstance(row.get("title"), str) or not isinstance(row.get("text"), str):
            raise ValueError(f"invalid corpus row in {dataset}: title/text must be strings")
    matrix = np.load(folder / "emb_minilm.f16.npy", mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape != (len(rows), 384):
        raise ValueError(f"embedding shape mismatch for {dataset}: {matrix.shape}")
    # Convert once to a contiguous float32 matrix for predictable CPU BLAS scoring.
    base = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(base).all():
        raise ValueError(f"non-finite registered embeddings in {dataset}")
    norms = np.linalg.norm(base[: min(256, len(base))], axis=1)
    if norms.size and not np.allclose(norms, 1.0, atol=0.03):
        raise ValueError(f"registered embeddings are not normalized in {dataset}")
    return Corpus(
        dataset=dataset,
        rows=rows,
        pids=pids,
        pid_to_index={pid: i for i, pid in enumerate(pids)},
        texts=[row["text"] for row in rows],
        encoder_texts=[f"{row['title']} [SEP] {row['text']}" for row in rows],
        base_embeddings=base,
        corpus_sha256=sha256_file(folder / "corpus.jsonl"),
    )


def load_questions(dataset: str, split: str) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(DATA_ROOT / dataset / "questions.jsonl") if row.get("split") == split]
    rows.sort(key=lambda row: str(row["qid"]))
    for row in rows:
        if not isinstance(row.get("question"), str) or not isinstance(row.get("gold_pids"), list):
            raise ValueError(f"invalid question row in {dataset}: question/gold_pids schema")
    return rows


def passage_text(row: dict[str, Any]) -> str:
    return f"{row['title']} [SEP] {row['text']}"


class MiniEncoder:
    def __init__(self, tokenizer: Any, model: Any, torch: Any):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch
        self.truncations: Counter[str] = Counter()
        self.device = "cpu"
        self.model.to(self.device)
        if next(self.model.parameters()).device.type != "cpu":
            raise RuntimeError("encoder must remain on CPU")

    def _tokenize(self, texts: list[str], role: str) -> dict[str, Any]:
        raw = self.tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        self.truncations[role] += sum(len(ids) > MAX_LENGTH for ids in raw["input_ids"])
        tokens = self.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        return {key: value.to("cpu") for key, value in tokens.items()}

    def gradient_encode(self, texts: list[str], role: str) -> Any:
        tokens = self._tokenize(texts, role)
        special = tokens.pop("special_tokens_mask")
        hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.torch.nn.functional.normalize(pooled, p=2, dim=1)

    def encode_texts(
        self,
        texts: list[str],
        role: str,
        need_tokens: bool = False,
        batch_size: int = 64,
    ) -> tuple[np.ndarray, list[np.ndarray] | None]:
        vectors: list[np.ndarray] = []
        token_vectors: list[np.ndarray] = []
        self.model.eval()
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                check_deadline()
                tokens = self._tokenize(texts[start : start + batch_size], role)
                special = tokens.pop("special_tokens_mask").numpy().astype(bool)
                hidden = self.model(**tokens).last_hidden_state
                attention = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(min=1)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().numpy().astype(np.float32, copy=False))
                if need_tokens:
                    hidden_np = hidden.cpu().numpy()
                    attention_np = tokens["attention_mask"].numpy().astype(bool)
                    for i in range(hidden_np.shape[0]):
                        valid = attention_np[i] & ~special[i]
                        row = hidden_np[i, valid]
                        if not row.size:
                            row = hidden_np[i, :1]
                        row = row / np.maximum(np.linalg.norm(row, axis=1, keepdims=True), 1e-12)
                        token_vectors.append(row.astype(np.float32, copy=False))
        matrix = np.stack(vectors) if vectors else np.empty((0, 384), dtype=np.float32)
        return matrix, token_vectors if need_tokens else None


def load_encoder() -> MiniEncoder:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install the pinned project dependencies with `uv sync` first") from exc
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=str(HF_HOME / "hub"),
            allow_patterns=(
                "config.json", "model.safetensors", "pytorch_model.bin", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                "added_tokens.json", "modules.json", "sentence_bert_config.json",
            ),
            max_workers=2,
            token=False,
        )
    )
    model_bytes = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
    if model_bytes > 150_000_000:
        raise RuntimeError(f"model snapshot exceeds 150 MB allowance: {model_bytes} bytes")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32)
    model.eval()
    return MiniEncoder(tokenizer, model, torch)


def unique_valid_gold(row: dict[str, Any], corpus: Corpus) -> list[str]:
    result: list[str] = []
    for value in row["gold_pids"]:
        pid = str(value)
        if pid in corpus.pid_to_index and pid not in result:
            result.append(pid)
    return result


def build_training_rows(corpus: Corpus) -> list[dict[str, Any]]:
    selected = []
    for row in load_questions("hotpotqa", "train"):
        gold = unique_valid_gold(row, corpus)
        if len(gold) >= 2:
            selected.append({**row, "_gold": gold})
    selected.sort(key=lambda row: str(row["qid"]))
    selected = selected[:MAX_TRAIN_QUESTIONS]
    if not selected:
        raise ValueError("no HotpotQA train questions with two valid gold PIDs")
    return selected


def mine_hard_negatives(
    rows: list[dict[str, Any]], corpus: Corpus, encoder: MiniEncoder
) -> tuple[list[str], list[str]]:
    hop1_queries = [row["question"] for row in rows]
    hop2_queries = [
        f"{row['question']} [SEP] {passage_text(corpus.rows[corpus.pid_to_index[row['_gold'][0]]])}"
        for row in rows
    ]

    def mine(query_texts: list[str], role: str) -> list[str]:
        found: list[str] = []
        for start in range(0, len(query_texts), BATCH_SIZE):
            check_deadline()
            end = min(len(query_texts), start + BATCH_SIZE)
            qvecs, _ = encoder.encode_texts(query_texts[start:end], role, batch_size=BATCH_SIZE)
            scores = corpus.base_embeddings @ qvecs.T
            for local, row in enumerate(rows[start:end]):
                for pid in row["_gold"]:
                    scores[corpus.pid_to_index[pid], local] = -np.inf
                idx = int(np.argmax(scores[:, local]))
                if not np.isfinite(scores[idx, local]):
                    raise ValueError(f"no corpus hard negative for qid={row['qid']}")
                found.append(corpus.pids[idx])
        return found

    return mine(hop1_queries, "hard_negative_query"), mine(hop2_queries, "hard_negative_conditioned")


def _candidate_ids(batch: list[dict[str, Any]], positive_key: str, neg_key: str, all_gold: bool) -> list[str]:
    values: list[str] = []
    for row in batch:
        positives = row["_gold"] if all_gold else [row[positive_key]]
        values.extend(positives)
        values.append(row[neg_key])
    return list(dict.fromkeys(values))


def contrastive_loss(
    torch: Any,
    query_vecs: Any,
    passage_vecs: Any,
    candidate_pids: list[str],
    gold_sets: list[set[str]],
    targets: list[set[str]],
) -> Any:
    scores = query_vecs @ passage_vecs.T / TEMPERATURE
    positive = torch.zeros_like(scores, dtype=torch.bool)
    valid = torch.ones_like(scores, dtype=torch.bool)
    for i, (golds, wanted) in enumerate(zip(gold_sets, targets)):
        for j, pid in enumerate(candidate_pids):
            if pid in golds:
                if pid in wanted:
                    positive[i, j] = True
                else:
                    valid[i, j] = False
    if not positive.any(dim=1).all():
        raise ValueError("every training query must have at least one in-batch positive")
    numerator = torch.logsumexp(scores.masked_fill(~positive, -1e9), dim=1)
    denominator = torch.logsumexp(scores.masked_fill(~valid, -1e9), dim=1)
    return -(numerator - denominator).mean()


def _train_one(
    method: str,
    model: Any,
    base_encoder: MiniEncoder,
    rows: list[dict[str, Any]],
    neg1: list[str],
    neg2: list[str],
    corpus: Corpus,
) -> tuple[Any, dict[str, Any]]:
    torch = base_encoder.torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    encoder = MiniEncoder(base_encoder.tokenizer, model, torch)
    encoder.model.train()
    optimizer = torch.optim.AdamW(
        encoder.model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )
    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    losses: list[float] = []
    batches = math.ceil(len(rows) / BATCH_SIZE)
    for batch_no, offset in enumerate(range(0, len(order), BATCH_SIZE), 1):
        check_deadline()
        indices = order[offset : offset + BATCH_SIZE]
        batch = [rows[i] for i in indices]
        q_texts = [row["question"] for row in batch]
        gold_sets = [set(row["_gold"]) for row in batch]
        optimizer.zero_grad(set_to_none=True)
        if method == "trained_single_shot":
            candidates = _candidate_ids(batch, "", "_neg1", all_gold=True)
            qvecs = encoder.gradient_encode(q_texts, "train_query_single")
            dvecs = encoder.gradient_encode(
                [passage_text(corpus.rows[corpus.pid_to_index[pid]]) for pid in candidates],
                "train_passage_single",
            )
            targets = [set(row["_gold"]) for row in batch]
            loss = contrastive_loss(torch, qvecs, dvecs, candidates, gold_sets, targets)
        elif method == "trained_mdr":
            rows1 = [{**row, "_neg1": neg1[i]} for i, row in zip(indices, batch)]
            rows2 = [{**row, "_neg2": neg2[i]} for i, row in zip(indices, batch)]
            # The hop targets are the first two gold PIDs in the frozen order.
            for row in rows1:
                row["_gold0"] = row["_gold"][0]
            candidates1 = _candidate_ids(rows1, "_gold0", "_neg1", all_gold=False)
            q1 = encoder.gradient_encode(q_texts, "train_query_hop1")
            d1 = encoder.gradient_encode(
                [passage_text(corpus.rows[corpus.pid_to_index[pid]]) for pid in candidates1],
                "train_passage_hop1",
            )
            loss1 = contrastive_loss(
                torch, q1, d1, candidates1, gold_sets,
                [{row["_gold"][0]} for row in batch],
            )
            q2_texts = [
                f"{row['question']} [SEP] {passage_text(corpus.rows[corpus.pid_to_index[row['_gold'][0]]])}"
                for row in batch
            ]
            for row in rows2:
                row["_gold1"] = row["_gold"][1]
            candidates2 = _candidate_ids(rows2, "_gold1", "_neg2", all_gold=False)
            q2 = encoder.gradient_encode(q2_texts, "train_query_hop2")
            d2 = encoder.gradient_encode(
                [passage_text(corpus.rows[corpus.pid_to_index[pid]]) for pid in candidates2],
                "train_passage_hop2",
            )
            loss2 = contrastive_loss(
                torch, q2, d2, candidates2, gold_sets,
                [{row["_gold"][1]} for row in batch],
            )
            loss = (loss1 + loss2) / 2
        else:
            raise ValueError(f"unknown training method: {method}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if batch_no == 1 or batch_no % 10 == 0 or batch_no == batches:
            print(
                f"train method={method} batch={batch_no}/{batches} loss={losses[-1]:.5f}",
                flush=True,
            )
    encoder.model.eval()
    return encoder.model, {
        "questions": len(rows),
        "updates": batches,
        "mean_training_loss": float(np.mean(losses)),
        "truncations": dict(encoder.truncations),
    }


def _save_checkpoint(path: Path, model: Any, metadata: dict[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
    }
    with path.open("xb") as handle:
        torch.save(payload, handle)


def train_models(ready_hash: str) -> dict[str, Any]:
    model_dir = ROOT / "models"
    paths = {name: model_dir / f"{name}.pt" for name in ("trained_mdr", "trained_single_shot")}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("training checkpoint path already exists; refusing to overwrite")
    corpus = load_corpus("hotpotqa")
    rows = build_training_rows(corpus)
    base = load_encoder()
    # Mine from the frozen initializer for both methods before any weight updates.
    neg1, neg2 = mine_hard_negatives(rows, corpus, base)
    neg_single = neg1
    for method in ("trained_mdr", "trained_single_shot"):
        check_deadline()
        model = copy.deepcopy(base.model)
        if method == "trained_single_shot":
            for row, neg in zip(rows, neg_single):
                row["_neg1"] = neg
        trained, train_meta = _train_one(method, model, base, rows, neg1, neg2, corpus)
        metadata = {
            "method": method,
            "seed": SEED,
            "questions": len(rows),
            "training_qids_sha256": hashlib.sha256(
                "\n".join(str(row["qid"]) for row in rows).encode()
            ).hexdigest(),
            "ready_sha256": ready_hash,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "train_meta": train_meta,
            "ordered_gold_policy": "first two distinct valid gold_pids in stored order",
            "single_shot_positive_policy": "all distinct valid gold_pids",
            "hard_negative_policy": "top frozen MiniLM corpus score excluding every gold pid",
        }
        _save_checkpoint(paths[method], trained, metadata, base.torch)
        del trained, model
        print(f"saved {paths[method]}", flush=True)
    return {name: {"sha256": sha256_file(path), "path": str(path)} for name, path in paths.items()}


def load_trained_encoder(path: Path, base: MiniEncoder) -> MiniEncoder:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = base.torch.load(path, map_location="cpu", weights_only=True)
    model = copy.deepcopy(base.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return MiniEncoder(base.tokenizer, model, base.torch)


def embed_corpus(encoder: MiniEncoder, corpus: Corpus, method: str, checkpoint_hash: str) -> np.ndarray:
    cache = ROOT / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    filename = f"{corpus.dataset}.{method}.{checkpoint_hash[:12]}.{corpus.corpus_sha256[:12]}.f16.npy"
    path = cache / filename
    if path.is_file():
        matrix = np.load(path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != (len(corpus.rows), 384):
            raise ValueError(f"cached embedding shape mismatch: {path}")
        return matrix
    if path.exists():
        raise FileExistsError(f"refusing to overwrite non-file cache path: {path}")
    vectors = encoder.encode_texts(corpus.encoder_texts, f"corpus_index_{method}", batch_size=128)[0]
    if vectors.shape != (len(corpus.rows), 384):
        raise ValueError(f"encoded embedding shape mismatch for {corpus.dataset}/{method}")
    with path.open("xb") as handle:
        np.save(handle, vectors.astype(np.float16), allow_pickle=False)
    return np.load(path, mmap_mode="r", allow_pickle=False)


class BM25Index:
    def __init__(self, texts: list[str]):
        self.n = len(texts)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.casefold())
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((idx, tf))
        self.average_length = sum(self.lengths) / max(1, self.n)

    def rank(self, question: str, k: int) -> list[int]:
        scores = np.zeros(self.n, dtype=np.float64)
        for term in re.findall(r"[a-z0-9]+", question.casefold()):
            postings = self.postings.get(term, ())
            df = len(postings)
            if not df:
                continue
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for idx, tf in postings:
                denom = tf + 1.2 * (1.0 - 0.75 + 0.75 * self.lengths[idx] / max(self.average_length, 1e-12))
                scores[idx] += idf * tf * 2.2 / denom
        return stable_topk(scores, k)


def mmr_order(query: np.ndarray, document_vectors: np.ndarray, k: int) -> list[int]:
    relevance = document_vectors @ query
    redundancy = np.zeros(len(document_vectors), dtype=np.float32)
    selected: list[int] = []
    for _ in range(min(k, len(document_vectors))):
        score = 0.5 * relevance - 0.5 * redundancy if selected else relevance.copy()
        if selected:
            score[selected] = -np.inf
        idx = int(np.argmax(score))
        selected.append(idx)
        sims = document_vectors @ document_vectors[idx]
        redundancy = sims if len(selected) == 1 else np.maximum(redundancy, sims)
    return selected


def facility_order(
    query_tokens: np.ndarray,
    candidate_token_vectors: list[np.ndarray],
    token_costs: list[int],
    k: int,
    cost_normalized: bool,
) -> list[int]:
    coverages = [
        np.max(query_tokens @ tokens.T, axis=1) if len(tokens) else np.zeros(len(query_tokens), dtype=np.float32)
        for tokens in candidate_token_vectors
    ]
    selected: list[int] = []
    current = np.zeros(len(query_tokens), dtype=np.float32)
    for _ in range(min(k, len(coverages))):
        best_i, best_score, best_current = -1, -np.inf, None
        for i, coverage in enumerate(coverages):
            if i in selected:
                continue
            new = np.maximum(current, coverage)
            gain = float(np.sum(new - current))
            score = gain / max(1, token_costs[i]) if cost_normalized else gain
            if score > best_score:
                best_i, best_score, best_current = i, score, new
        selected.append(best_i)
        current = np.asarray(best_current, dtype=np.float32)
    return selected


def conditioned_beam(
    question: str,
    corpus: Corpus,
    document_vectors: np.ndarray,
    encoder: MiniEncoder,
    beam_width: int = BEAM_WIDTH,
    hops: int = 2,
) -> tuple[list[int], float]:
    started = time.perf_counter()
    beam: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    for _ in range(hops):
        check_deadline()
        contexts = [
            question if not path else question + " [SEP] " + " [SEP] ".join(
                passage_text(corpus.rows[i]) for i in path
            )
            for path, _ in beam
        ]
        query_vectors = encoder.encode_texts(contexts, "retrieval_conditioned_query", batch_size=beam_width)[0]
        expanded: list[tuple[tuple[int, ...], float]] = []
        for (path, path_score), query_vec in zip(beam, query_vectors):
            scores = document_vectors @ query_vec
            if path:
                scores[list(path)] = -np.inf
            for idx in stable_topk(scores, beam_width):
                expanded.append((path + (idx,), path_score + float(scores[idx])))
        expanded.sort(key=lambda item: (-item[1], item[0]))
        beam = expanded[:beam_width]
    ranked: list[int] = []
    for path, _ in sorted(beam, key=lambda item: (-item[1], item[0])):
        for idx in path:
            if idx not in ranked:
                ranked.append(idx)
            if len(ranked) >= MAX_K:
                return ranked, (time.perf_counter() - started) * 1000.0
    return ranked, (time.perf_counter() - started) * 1000.0


def greedy_frozen_conditioning(
    question: str, corpus: Corpus, document_vectors: np.ndarray, encoder: MiniEncoder, k: int
) -> tuple[list[int], float]:
    started = time.perf_counter()
    selected: list[int] = []
    for _ in range(k):
        context = question if not selected else question + " [SEP] " + " [SEP] ".join(
            passage_text(corpus.rows[i]) for i in selected
        )
        query = encoder.encode_texts([context], "retrieval_frozen_conditioned_query")[0][0]
        scores = document_vectors @ query
        if selected:
            scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))
    return selected, (time.perf_counter() - started) * 1000.0


def retrieve_one(
    question: dict[str, Any],
    corpus: Corpus,
    base: MiniEncoder,
    trained_mdr: MiniEncoder,
    trained_single: MiniEncoder,
    mdr_vectors: np.ndarray,
    single_vectors: np.ndarray,
    bm25: BM25Index,
) -> dict[str, tuple[list[int], float]]:
    text = question["question"]
    results: dict[str, tuple[list[int], float]] = {}

    started = time.perf_counter()
    results["bm25"] = (bm25.rank(text, MAX_K), (time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    qvec = base.encode_texts([text], "retrieval_frozen_dense_query")[0][0]
    results["frozen_dense"] = (stable_topk(corpus.base_embeddings @ qvec, MAX_K), (time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    qvec = trained_single.encode_texts([text], "retrieval_trained_single_query")[0][0]
    results["trained_single_shot"] = (stable_topk(single_vectors @ qvec, MAX_K), (time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    qvec = base.encode_texts([text], "retrieval_mmr_query")[0][0]
    results["frozen_mmr"] = (mmr_order(qvec, corpus.base_embeddings, MAX_K), (time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    qvecs, qtokens = base.encode_texts([text], "retrieval_disco_query", need_tokens=True)
    short = stable_topk(corpus.base_embeddings @ qvecs[0], min(DISCO_SHORTLIST, len(corpus.rows)))
    _, token_rows = base.encode_texts(
        [corpus.encoder_texts[i] for i in short], "retrieval_disco_candidates", need_tokens=True, batch_size=64
    )
    costs = [len(row) for row in token_rows]
    common_ms = (time.perf_counter() - started) * 1000.0
    for name, normalize in (("disco_cost_norm_approx", True), ("disco_no_cost_norm_approx", False)):
        method_started = time.perf_counter()
        local = facility_order(qtokens[0], token_rows, costs, MAX_K, normalize)
        results[name] = ([short[i] for i in local], common_ms + (time.perf_counter() - method_started) * 1000.0)

    results["frozen_iterative_conditioning"] = greedy_frozen_conditioning(
        text, corpus, corpus.base_embeddings, base, MAX_K
    )
    results["trained_mdr"] = conditioned_beam(text, corpus, mdr_vectors, trained_mdr)
    return results


def evaluate(
    split: str,
    datasets: dict[str, Corpus],
    base: MiniEncoder,
    trained_mdr: MiniEncoder,
    trained_single: MiniEncoder,
    model_vectors: dict[tuple[str, str], np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    truncation_before = {
        "base": dict(base.truncations),
        "mdr": dict(trained_mdr.truncations),
        "single": dict(trained_single.truncations),
    }
    for ds in DATASETS:
        corpus = datasets[ds]
        questions = load_questions(ds, split)
        if not questions:
            raise ValueError(f"no {split} questions for {ds}")
        bm25 = BM25Index(corpus.encoder_texts)
        for number, q in enumerate(questions, 1):
            check_deadline()
            gold_pids = unique_valid_gold(q, corpus)
            if not gold_pids or len(gold_pids) != len(set(map(str, q["gold_pids"]))):
                raise ValueError(f"gold PIDs missing from {ds} corpus: qid={q['qid']}")
            gold = {corpus.pid_to_index[pid] for pid in gold_pids}
            retrieval = retrieve_one(
                q, corpus, base, trained_mdr, trained_single,
                model_vectors[(ds, "trained_mdr")],
                model_vectors[(ds, "trained_single_shot")], bm25,
            )
            by_method = {}
            for method, (selected, latency) in retrieval.items():
                values = {
                    str(k): metric_values(selected, gold, corpus.texts, k)
                    for k in range(1, MAX_K + 1)
                }
                by_method[method] = {
                    "metrics": values,
                    "latency_ms": latency,
                    "retrieved_pids_at_4": [corpus.pids[i] for i in selected[:MAX_K]],
                }
            raw.append({
                "dataset": ds,
                "qid": str(q["qid"]),
                "type": str(q.get("type", "unknown")),
                "methods": by_method,
            })
            if number == 1 or number % 100 == 0 or number == len(questions):
                print(f"eval split={split} dataset={ds} q={number}/{len(questions)}", flush=True)
    truncation_after = {
        "base": dict(base.truncations),
        "mdr": dict(trained_mdr.truncations),
        "single": dict(trained_single.truncations),
    }
    truncations = {
        name: {role: value - truncation_before[name].get(role, 0) for role, value in counts.items()}
        for name, counts in truncation_after.items()
    }
    return raw, truncations


def summarize(raw: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for ds in DATASETS:
        dataset_rows = [row for row in raw if row["dataset"] == ds]
        if dataset_rows:
            groups.append((ds, "all", dataset_rows))
            for typ in sorted({row["type"] for row in dataset_rows}):
                groups.append((ds, typ, [row for row in dataset_rows if row["type"] == typ]))
    metric_names = ("full_support", "sp_recall", "evidence_tokens")
    for ds, typ, rows in groups:
        n = len(rows)
        for method in METHODS:
            for k in range(1, MAX_K + 1):
                metrics = [row["methods"][method]["metrics"][str(k)] for row in rows]
                entry = {
                    "dataset": ds,
                    "question_type": typ,
                    "method": method,
                    "k": k,
                    "n": n,
                    "full_support": float(np.mean([m["full_support"] for m in metrics])),
                    "sp_recall": float(np.mean([m["sp_recall"] for m in metrics])),
                    "evidence_tokens": float(np.mean([m["evidence_tokens"] for m in metrics])),
                }
                if k == MAX_K:
                    entry["latency_ms_per_query"] = float(np.mean([
                        row["methods"][method]["latency_ms"] for row in rows
                    ]))
                metric_rows.append(entry)
        for method in COMPARATORS:
            if method == "trained_mdr":
                continue
            for metric in metric_names + ("latency_ms",):
                if metric == "latency_ms":
                    left = [row["methods"]["trained_mdr"]["latency_ms"] for row in rows]
                    right = [row["methods"][method]["latency_ms"] for row in rows]
                    key = "latency_ms_per_query"
                else:
                    left = [row["methods"]["trained_mdr"]["metrics"][str(MAX_K)][metric] for row in rows]
                    right = [row["methods"][method]["metrics"][str(MAX_K)][metric] for row in rows]
                    key = metric
                comparisons.append({
                    "dataset": ds,
                    "question_type": typ,
                    "k": MAX_K,
                    "method_a": "trained_mdr",
                    "method_b": method,
                    "metric": key,
                    **paired_bootstrap(left, right),
                })
    gate_row = next(
        row for row in comparisons
        if row["dataset"] == "hotpotqa" and row["question_type"] == "all"
        and row["method_b"] == "trained_single_shot" and row["metric"] == "full_support"
    )
    gate = {
        **gate_row,
        "threshold": 0.05,
        "threshold_met": gate_row["difference"] >= 0.05,
        "ci_excludes_zero": gate_row["ci95_low"] > 0 or gate_row["ci95_high"] < 0,
    }
    gate["verdict"] = "PASS" if gate["threshold_met"] and gate["ci_excludes_zero"] else "FAIL"
    return metric_rows, comparisons, gate


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _freeze_config(ready_hash: str) -> dict[str, Any]:
    model_paths = {
        "trained_mdr": ROOT / "models" / "trained_mdr.pt",
        "trained_single_shot": ROOT / "models" / "trained_single_shot.pt",
    }
    for path in model_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing trained model: {path}")
    return {
        "task_id": "ire-p5-h2cpu-trainedmdr-luna-011",
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "code_sha256": sha256_file(ROOT / "run_h2.py"),
        "ready_sha256": ready_hash,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "training": {
            "seed": SEED,
            "max_questions": MAX_TRAIN_QUESTIONS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "epochs": 1,
            "threads": THREADS,
        },
        "beam_width": BEAM_WIDTH,
        "disco_shortlist": DISCO_SHORTLIST,
        "checkpoints": {name: sha256_file(path) for name, path in model_paths.items()},
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _load_frozen_config(ready_hash: str) -> dict[str, Any]:
    path = ROOT / "freeze_config.json"
    config = json.loads(path.read_text())
    expected = _freeze_config(ready_hash)
    for key in (
        "protocol_sha256", "code_sha256", "ready_sha256", "checkpoints", "training",
        "beam_width", "disco_shortlist", "model_id", "model_revision",
    ):
        if config.get(key) != expected.get(key):
            raise ValueError(f"frozen holdout config mismatch for {key}; holdout not started")
    return config


def _result_markdown(results: dict[str, Any]) -> str:
    gate = results["gate_g2_double_prime"]
    lines = [
        "# H2 holdout results",
        "",
        f"**G2'' verdict: {gate['verdict']}** — trained MDR minus trained single-shot FULL-SUPPORT@4 "
        f"on HotpotQA holdout = {gate['difference']:+.3f} "
        f"(paired 95% CI [{gate['ci95_low']:+.3f}, {gate['ci95_high']:+.3f}], n={gate['n']}); "
        f"threshold +0.050: {'met' if gate['threshold_met'] else 'not met'}; "
        f"CI excludes zero: {gate['ci_excludes_zero']}.",
        "",
        "## Overall holdout at k=4",
        "",
        "| Dataset | Method | FULL-SUPPORT | SP recall | Evidence tokens | ms/query | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for method in METHODS:
            row = next(r for r in results["metrics"] if r["dataset"] == ds and r["question_type"] == "all" and r["method"] == method and r["k"] == 4)
            lines.append(
                f"| {ds} | {method} | {row['full_support']:.3f} | {row['sp_recall']:.3f} | "
                f"{row['evidence_tokens']:.1f} | {row['latency_ms_per_query']:.1f} | {row['n']} |"
            )
    lines += [
        "",
        "Question-type breakdowns and paired CIs for every k=4 method difference are in `results.json`.",
        "Coverage and beam-search approximations are labeled in `protocol.md`.",
        "",
    ]
    return "\n".join(lines)


def run_train(ready_hash: str) -> None:
    outcomes = train_models(ready_hash)
    print(json.dumps({"status": "trained", "checkpoints": outcomes}, indent=2), flush=True)


def run_tune(ready_hash: str) -> None:
    base = load_encoder()
    mdr_path = ROOT / "models" / "trained_mdr.pt"
    single_path = ROOT / "models" / "trained_single_shot.pt"
    mdr = load_trained_encoder(mdr_path, base)
    single = load_trained_encoder(single_path, base)
    datasets = {ds: load_corpus(ds) for ds in DATASETS}
    vectors: dict[tuple[str, str], np.ndarray] = {}
    for ds, corpus in datasets.items():
        vectors[(ds, "trained_mdr")] = embed_corpus(mdr, corpus, "trained_mdr", sha256_file(mdr_path))
        vectors[(ds, "trained_single_shot")] = embed_corpus(single, corpus, "trained_single_shot", sha256_file(single_path))
    raw, truncations = evaluate("tune", datasets, base, mdr, single, vectors)
    metrics, _, gate = summarize(raw)
    print(json.dumps({"stage": "tune", "n": len(raw), "gate_diagnostic_only": gate, "truncations": truncations}, indent=2), flush=True)
    print("Tune diagnostics complete; inspect them before running --stage freeze.", flush=True)


def run_freeze(ready_hash: str) -> None:
    path = ROOT / "freeze_config.json"
    config = _freeze_config(ready_hash)
    _write_json_new(path, config)
    print(f"froze holdout config sha256={sha256_file(path)}", flush=True)


def run_holdout(ready_hash: str) -> None:
    run_log = ROOT / "holdout_runs.log"
    if run_log.exists():
        raise FileExistsError("holdout_runs.log exists; a holdout attempt has already started")
    config = _load_frozen_config(ready_hash)
    # Validate all holdout questions and gold IDs before consuming the single run.
    datasets = {ds: load_corpus(ds) for ds in DATASETS}
    for ds, corpus in datasets.items():
        questions = load_questions(ds, "holdout")
        if not questions:
            raise ValueError(f"no holdout questions for {ds}")
        for row in questions:
            gold = unique_valid_gold(row, corpus)
            if not gold or len(gold) != len(set(map(str, row["gold_pids"]))):
                raise ValueError(f"invalid/missing holdout gold PID in {ds}, qid={row['qid']}")
    with run_log.open("x") as handle:
        handle.write(json.dumps({
            "status": "STARTED",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "freeze_config_sha256": sha256_file(ROOT / "freeze_config.json"),
            "ready_sha256": ready_hash,
        }) + "\n")
    try:
        base = load_encoder()
        mdr_path = ROOT / "models" / "trained_mdr.pt"
        single_path = ROOT / "models" / "trained_single_shot.pt"
        mdr = load_trained_encoder(mdr_path, base)
        single = load_trained_encoder(single_path, base)
        vectors: dict[tuple[str, str], np.ndarray] = {}
        for ds, corpus in datasets.items():
            vectors[(ds, "trained_mdr")] = embed_corpus(mdr, corpus, "trained_mdr", sha256_file(mdr_path))
            vectors[(ds, "trained_single_shot")] = embed_corpus(single, corpus, "trained_single_shot", sha256_file(single_path))
        raw, truncations = evaluate("holdout", datasets, base, mdr, single, vectors)
        metrics, comparisons, gate = summarize(raw)
        results = {
            "status": "complete",
            "task_id": "ire-p5-h2cpu-trainedmdr-luna-011",
            "ready_sha256": ready_hash,
            "freeze_config_sha256": sha256_file(ROOT / "freeze_config.json"),
            "protocol_sha256": config["protocol_sha256"],
            "gate_g2_double_prime": gate,
            "metrics": metrics,
            "paired_differences_at_k4": comparisons,
            "truncations": truncations,
            "holdout_question_count": len(raw),
            "methods": list(METHODS),
            "question_level": [
                {
                    "dataset": row["dataset"],
                    "qid": row["qid"],
                    "type": row["type"],
                    "methods": {
                        method: {
                            "retrieved_pids_at_4": row["methods"][method]["retrieved_pids_at_4"],
                            "full_support_at_4": row["methods"][method]["metrics"][str(MAX_K)]["full_support"],
                            "sp_recall_at_4": row["methods"][method]["metrics"][str(MAX_K)]["sp_recall"],
                            "evidence_tokens_at_4": row["methods"][method]["metrics"][str(MAX_K)]["evidence_tokens"],
                            "latency_ms": row["methods"][method]["latency_ms"],
                        }
                        for method in METHODS
                    },
                }
                for row in raw
            ],
            "approximation_labels": ["DISCo-style coverage shortlist", "two-hop MDR beam search"],
        }
        _write_json_new(ROOT / "results.json", results)
        with (ROOT / "RESULT.md").open("x") as handle:
            handle.write(_result_markdown(results))
        with run_log.open("a") as handle:
            handle.write(json.dumps({
                "status": "DONE",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "results_sha256": sha256_file(ROOT / "results.json"),
                "gate_verdict": gate["verdict"],
            }) + "\n")
    except Exception as exc:
        with run_log.open("a") as handle:
            handle.write(json.dumps({
                "status": "FAILED",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }) + "\n")
        raise
    print(json.dumps({"gate_g2_double_prime": gate, "results": str(ROOT / "results.json")}, indent=2), flush=True)


def write_manifest() -> None:
    excluded = {"manifest.sha256"}
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name in excluded:
            continue
        rel = path.relative_to(ROOT)
        if rel.parts[0] in {".venv", ".uv-cache", ".hf-cache", ".cache", "__pycache__"}:
            continue
        if rel.suffix in {".pyc", ".lock"} and rel.parts[0] == "__pycache__":
            continue
        files.append(rel)
    lines = [f"{sha256_file(ROOT / path)}  {path.as_posix()}" for path in sorted(files)]
    with (ROOT / "manifest.sha256").open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate H2 retrieval models on CPU.")
    parser.add_argument("--stage", required=True, choices=("check", "train", "tune", "freeze", "holdout", "manifest"))
    args = parser.parse_args()
    if args.stage == "manifest":
        write_manifest()
        return
    hashes, ready_hash = verify_ready()
    print(f"READY verified: {len(hashes)} files; sha256={ready_hash}", flush=True)
    if args.stage == "check":
        for ds in DATASETS:
            corpus = load_corpus(ds)
            print(f"schema OK: {ds} corpus_rows={len(corpus.rows)}", flush=True)
        return
    if args.stage == "train":
        run_train(ready_hash)
    elif args.stage == "tune":
        run_tune(ready_hash)
    elif args.stage == "freeze":
        run_freeze(ready_hash)
    elif args.stage == "holdout":
        run_holdout(ready_hash)


if __name__ == "__main__":
    main()
