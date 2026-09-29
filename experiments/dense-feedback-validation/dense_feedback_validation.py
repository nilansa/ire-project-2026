#!/usr/bin/env python3
"""Dense/cross-encoder validation of budgeted feedback selection for retrieval post-training.

This is the decisive follow-up to the 100-direction lightweight screen.  It uses:
  * a real transformer bi-encoder (query side post-trained; document embeddings frozen),
  * a real cross-encoder reward model,
  * exact, HNSW, IVF and lexical+dense hybrid candidate generation,
  * three feedback budgets and three train/validation/evaluation splits,
  * five promoted policies plus top-candidate and random controls,
  * three BEIR datasets.

The broad sweep measures selected data quality.  Full query-encoder post-training is run for
all seven policies at the central operating point and for top-candidate vs the preliminary
winner across every budget/backend combination.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None
try:
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover
    hnswlib = None

EPS = 1e-8
DATASET_URL = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
}
POLICIES = {
    "top_base": ("Uniform budget + top dense candidates", "uniform", "top_base"),
    "random": ("Uniform budget + random exploration", "uniform", "random"),
    "difficulty_leverage": (
        "Difficulty-adaptive budget + feature-space leverage",
        "difficulty",
        "leverage",
    ),
    "uniform_mmr_uncertainty": (
        "Uniform budget + uncertainty-diversity MMR",
        "uniform",
        "mmr_uncertainty",
    ),
    "disagreement_hard_negative": (
        "Disagreement-adaptive budget + hard-negative proxy",
        "disagreement",
        "hard_negative",
    ),
    "disagreement_mmr_relevance": (
        "Disagreement-adaptive budget + relevance-diversity MMR",
        "disagreement",
        "mmr_relevance",
    ),
    "disagreement_uniform_rank": (
        "Disagreement-adaptive budget + uniform rank coverage",
        "disagreement",
        "uniform_rank",
    ),
}
BACKENDS = ("exact", "hnsw", "ivf", "hybrid")
BUDGETS = (4, 12, 32)
SEEDS = (0, 1, 2)


@dataclass
class Dataset:
    name: str
    doc_ids: list[str]
    titles: list[str]
    texts: list[str]
    query_ids: list[str]
    queries: list[str]
    qrels: dict[int, set[int]]


@dataclass
class Features:
    dense: np.ndarray
    bm25: np.ndarray
    char: np.ndarray
    coverage: np.ndarray
    normalized: np.ndarray
    doc_embeddings: np.ndarray
    query_embeddings: np.ndarray
    candidates: dict[str, list[np.ndarray]]
    backend_dense_scores: dict[str, list[np.ndarray]]
    candidate_exact_recall: dict[str, float]
    candidate_qrel_recall: dict[str, float]


def stable_int(*parts: object) -> int:
    raw = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % (2**32 - 1)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def download(url: str, path: Path, retries: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ire-dense-validation/1.0"})
            with urllib.request.urlopen(req, timeout=180) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            return path
        except Exception as exc:  # pragma: no cover
            last = exc
            path.unlink(missing_ok=True)
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last}")


def load_beir(name: str, cache: Path, max_queries: int = 300) -> Dataset:
    if name not in DATASET_URL:
        raise ValueError(f"Unknown dataset: {name}")
    zpath = download(DATASET_URL[name], cache / f"{name}.zip")
    root = cache / f"{name}_unzipped"
    marker = root / ".complete"
    if not marker.exists():
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(root)
        marker.write_text("ok\n")
    data_root = next(
        (p for p in (root / name, root) if (p / "corpus.jsonl").exists()), None
    )
    if data_root is None:
        raise FileNotFoundError(f"BEIR archive layout not recognized for {name}")

    doc_ids: list[str] = []
    titles: list[str] = []
    texts: list[str] = []
    with (data_root / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            did = str(row["_id"])
            title = str(row.get("title", "")).strip()
            body = str(row.get("text", "")).strip()
            doc_ids.append(did)
            titles.append(title)
            texts.append((title + " " + body).strip())

    all_queries: dict[str, str] = {}
    with (data_root / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            all_queries[str(row["_id"])] = str(row["text"])

    did_to_i = {did: i for i, did in enumerate(doc_ids)}
    qrels_by_id: dict[str, set[int]] = defaultdict(set)
    qrel_path = data_root / "qrels" / "test.tsv"
    with qrel_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row.get("query-id") or row.get("query_id") or row.get("qid"))
            did = str(row.get("corpus-id") or row.get("corpus_id") or row.get("docid"))
            score = float(row.get("score", 0))
            if score > 0 and did in did_to_i and qid in all_queries:
                qrels_by_id[qid].add(did_to_i[did])

    eligible = [qid for qid in all_queries if qrels_by_id.get(qid)]
    eligible.sort(key=lambda qid: hashlib.sha256(f"{name}|{qid}".encode()).hexdigest())
    chosen = eligible[: min(max_queries, len(eligible))]
    return Dataset(
        name=name,
        doc_ids=doc_ids,
        titles=titles,
        texts=texts,
        query_ids=chosen,
        queries=[all_queries[qid] for qid in chosen],
        qrels={i: qrels_by_id[qid] for i, qid in enumerate(chosen)},
    )


def mean_pool(last_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp_min(EPS)


@torch.inference_mode()
def encode_texts(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    batch_size: int = 64,
    max_length: int = 192,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        tok = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        out = model(**tok)
        emb = F.normalize(mean_pool(out.last_hidden_state, tok["attention_mask"]), p=2, dim=1)
        rows.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def bm25_and_coverage(docs: list[str], queries: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vec = CountVectorizer(stop_words="english", min_df=1, max_features=30000)
    dmat = vec.fit_transform(docs).tocsr().astype(np.float32)
    qmat = vec.transform(queries).tocsr().astype(np.float32)
    n_docs = dmat.shape[0]
    dl = np.asarray(dmat.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(dl.mean() + EPS)
    df = np.diff(dmat.tocsc().indptr).astype(np.float32)
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    k1, b = 1.2, 0.75
    weighted = dmat.copy()
    rows = np.repeat(np.arange(n_docs), np.diff(weighted.indptr))
    tf = weighted.data
    denom = tf + k1 * (1 - b + b * dl[rows] / avgdl)
    weighted.data = (tf * (k1 + 1) / (denom + EPS)) * idf[weighted.indices]
    qbin = qmat.copy()
    qbin.data[:] = 1.0
    scores = (qbin @ weighted.T).toarray().astype(np.float32)
    dbin = dmat.copy()
    dbin.data[:] = 1.0
    overlap = (qbin @ dbin.T).toarray().astype(np.float32)
    qlen = np.asarray(qbin.sum(axis=1)).ravel().astype(np.float32)
    coverage = overlap / (qlen[:, None] + EPS)
    return scores, coverage.astype(np.float32)


def row_minmax(x: np.ndarray) -> np.ndarray:
    lo = x.min(1, keepdims=True)
    hi = x.max(1, keepdims=True)
    return ((x - lo) / (hi - lo + EPS)).astype(np.float32)


def topk(row: np.ndarray, k: int) -> np.ndarray:
    k = min(k, row.size)
    if k == row.size:
        return np.argsort(-row, kind="stable")
    idx = np.argpartition(-row, k - 1)[:k]
    return idx[np.argsort(-row[idx], kind="stable")]


def build_features(
    ds: Dataset,
    student_model_name: str,
    cache: Path,
    candidate_k: int,
) -> tuple[Features, AutoTokenizer, AutoModel]:
    print(f"[{ds.name}] loading student {student_model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(student_model_name, cache_dir=cache / "hf")
    model = AutoModel.from_pretrained(student_model_name, cache_dir=cache / "hf")
    model.to("cpu")
    doc_embeddings = encode_texts(model, tokenizer, ds.texts, batch_size=64)
    query_embeddings = encode_texts(model, tokenizer, ds.queries, batch_size=64)
    dense = (query_embeddings @ doc_embeddings.T).astype(np.float32)
    bm25, coverage = bm25_and_coverage(ds.texts, ds.queries)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    dchar = char_vec.fit_transform(ds.texts)
    qchar = char_vec.transform(ds.queries)
    char = (qchar @ dchar.T).toarray().astype(np.float32)

    ndense, nbm25, nchar, ncov = map(row_minmax, (dense, bm25, char, coverage))
    disagreement = np.stack([ndense, nbm25, nchar], axis=-1).std(-1).astype(np.float32)
    normalized = np.stack([ndense, nbm25, nchar, ncov, disagreement], axis=-1)

    exact = [topk(dense[i], candidate_k) for i in range(len(ds.queries))]
    exact_scores = [dense[i, ids] for i, ids in enumerate(exact)]

    if hnswlib is None:
        raise RuntimeError("hnswlib is required")
    hindex = hnswlib.Index(space="cosine", dim=doc_embeddings.shape[1])
    hindex.init_index(max_elements=len(ds.doc_ids), ef_construction=100, M=16, random_seed=17)
    hindex.add_items(doc_embeddings, np.arange(len(ds.doc_ids)))
    hindex.set_ef(max(candidate_k, 100))
    h_ids, h_dist = hindex.knn_query(query_embeddings, k=min(candidate_k, len(ds.doc_ids)))
    hnsw = [row.astype(np.int64) for row in h_ids]
    h_scores = [(1.0 - row).astype(np.float32) for row in h_dist]

    if faiss is None:
        raise RuntimeError("faiss-cpu is required")
    dim = doc_embeddings.shape[1]
    nlist = int(max(8, min(64, round(math.sqrt(len(ds.doc_ids))))))
    quantizer = faiss.IndexFlatIP(dim)
    iindex = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    iindex.train(np.ascontiguousarray(doc_embeddings))
    iindex.add(np.ascontiguousarray(doc_embeddings))
    iindex.nprobe = max(1, nlist // 8)
    i_scores_arr, i_ids_arr = iindex.search(
        np.ascontiguousarray(query_embeddings), min(candidate_k, len(ds.doc_ids))
    )
    ivf = [row[row >= 0].astype(np.int64) for row in i_ids_arr]
    i_scores = [row[: len(ivf[j])].astype(np.float32) for j, row in enumerate(i_scores_arr)]

    hybrid: list[np.ndarray] = []
    hybrid_scores: list[np.ndarray] = []
    for qi in range(len(ds.queries)):
        kd = min(candidate_k, len(ds.doc_ids))
        dtop = topk(dense[qi], kd)
        btop = topk(bm25[qi], kd)
        fused: dict[int, float] = defaultdict(float)
        for rank, did in enumerate(dtop):
            fused[int(did)] += 1.0 / (60.0 + rank + 1)
        for rank, did in enumerate(btop):
            fused[int(did)] += 1.0 / (60.0 + rank + 1)
        ordered = sorted(fused, key=lambda did: (-fused[did], did))[:candidate_k]
        arr = np.asarray(ordered, dtype=np.int64)
        hybrid.append(arr)
        hybrid_scores.append(np.asarray([fused[int(d)] for d in arr], dtype=np.float32))

    candidates = {"exact": exact, "hnsw": hnsw, "ivf": ivf, "hybrid": hybrid}
    backend_scores = {
        "exact": exact_scores,
        "hnsw": h_scores,
        "ivf": i_scores,
        "hybrid": hybrid_scores,
    }
    exact_sets = [set(map(int, ids)) for ids in exact]
    exact_recall: dict[str, float] = {}
    qrel_recall: dict[str, float] = {}
    for backend, rows in candidates.items():
        exact_recall[backend] = float(
            np.mean([len(set(map(int, r)) & exact_sets[i]) / max(1, len(exact_sets[i])) for i, r in enumerate(rows)])
        )
        qrel_recall[backend] = float(
            np.mean([
                len(set(map(int, r)) & ds.qrels[i]) / max(1, len(ds.qrels[i]))
                for i, r in enumerate(rows)
            ])
        )
    return (
        Features(
            dense=dense,
            bm25=bm25,
            char=char,
            coverage=coverage,
            normalized=normalized,
            doc_embeddings=doc_embeddings,
            query_embeddings=query_embeddings,
            candidates=candidates,
            backend_dense_scores=backend_scores,
            candidate_exact_recall=exact_recall,
            candidate_qrel_recall=qrel_recall,
        ),
        tokenizer,
        model,
    )


def split_queries(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_int("split", seed, n))
    order = rng.permutation(n)
    n_train = max(10, int(round(0.60 * n)))
    n_val = max(5, int(round(0.20 * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def pair_features(feat: Features, qi: int, doc_ids: np.ndarray) -> np.ndarray:
    base = feat.normalized[qi, doc_ids]
    dense = feat.dense[qi, doc_ids]
    order = np.argsort(-dense, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    rank_score = 1.0 - rank / max(1, len(rank) - 1)
    return np.concatenate([base, rank_score[:, None].astype(np.float32)], axis=1)


def choose_anchor(ds: Dataset, feat: Features, qi: int) -> int:
    rels = sorted(ds.qrels[qi])
    return max(rels, key=lambda d: (float(feat.dense[qi, d]), -d))


def fit_cheap_prior(
    ds: Dataset,
    feat: Features,
    train_idx: np.ndarray,
    seed: int,
) -> LogisticRegression:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    rng = np.random.default_rng(stable_int("prior", ds.name, seed))
    for qi0 in train_idx:
        qi = int(qi0)
        anchor = choose_anchor(ds, feat, qi)
        xs.append(pair_features(feat, qi, np.asarray([anchor]))[0])
        ys.append(1)
        cand = feat.candidates["exact"][qi]
        tail = cand[max(1, len(cand) * 2 // 3) :]
        if len(tail) == 0:
            tail = cand
        take = rng.choice(tail, size=min(4, len(tail)), replace=False)
        for did in take:
            xs.append(pair_features(feat, qi, np.asarray([int(did)]))[0])
            ys.append(0)
    model = LogisticRegression(
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
        solver="liblinear",
    )
    model.fit(np.asarray(xs, dtype=np.float32), np.asarray(ys))
    return model


def prior_probs(model: LogisticRegression, feat: Features, qi: int, ids: np.ndarray) -> np.ndarray:
    return model.predict_proba(pair_features(feat, qi, ids))[:, 1].astype(np.float32)


def allocation_scores(
    allocation: str,
    feat: Features,
    prior: LogisticRegression,
    train_idx: np.ndarray,
) -> np.ndarray:
    if allocation == "uniform":
        return np.ones(len(train_idx), dtype=np.float64)
    vals = []
    for qi0 in train_idx:
        qi = int(qi0)
        ids = feat.candidates["exact"][qi]
        p = prior_probs(prior, feat, qi, ids)
        entropy = float(np.mean(-(p * np.log(p + EPS) + (1 - p) * np.log(1 - p + EPS))))
        stack = feat.normalized[qi, ids][:, [0, 1, 2]]
        disagree = float(stack.std(1).mean())
        vals.append(entropy if allocation == "difficulty" else disagree)
    arr = np.asarray(vals, dtype=np.float64)
    return arr - arr.min() + 0.05


def integer_allocation(weights: np.ndarray, budget: int) -> np.ndarray:
    n = len(weights)
    total = budget * n
    low = max(1, budget // 2)
    high = max(low, 2 * budget)
    raw = weights / max(weights.sum(), EPS) * total
    out = np.clip(np.floor(raw).astype(int), low, high)
    # Adjust to the exact global budget, preferring largest fractional residuals.
    residual = raw - np.floor(raw)
    while out.sum() < total:
        eligible = np.where(out < high)[0]
        if len(eligible) == 0:
            break
        j = int(eligible[np.argmax(residual[eligible])])
        out[j] += 1
        residual[j] = -1
    while out.sum() > total:
        eligible = np.where(out > low)[0]
        if len(eligible) == 0:
            break
        j = int(eligible[np.argmin(residual[eligible])])
        out[j] -= 1
        residual[j] = 2
    return out


def mmr_select(ids: np.ndarray, utility: np.ndarray, doc_emb: np.ndarray, k: int, lam: float = 0.62) -> np.ndarray:
    if k >= len(ids):
        return ids.copy()
    util = (utility - utility.min()) / (np.ptp(utility) + EPS)
    selected_local: list[int] = [int(np.argmax(util))]
    remaining = set(range(len(ids))) - set(selected_local)
    while len(selected_local) < k and remaining:
        sel_emb = doc_emb[ids[selected_local]]
        best, best_score = None, -1e30
        for j in remaining:
            redundancy = float(np.max(sel_emb @ doc_emb[int(ids[j])]))
            score = lam * float(util[j]) - (1 - lam) * redundancy
            if score > best_score:
                best, best_score = j, score
        assert best is not None
        selected_local.append(int(best))
        remaining.remove(int(best))
    return ids[np.asarray(selected_local, dtype=int)]


def select_docs(
    selector: str,
    ids: np.ndarray,
    feat: Features,
    prior: LogisticRegression,
    qi: int,
    k: int,
    seed: int,
) -> np.ndarray:
    k = min(k, len(ids))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if selector == "top_base":
        return ids[:k].copy()
    if selector == "random":
        rng = np.random.default_rng(stable_int("select", seed, qi, tuple(map(int, ids))))
        return rng.choice(ids, size=k, replace=False).astype(np.int64)
    if selector == "uniform_rank":
        pos = np.unique(np.rint(np.linspace(0, len(ids) - 1, k)).astype(int))
        if len(pos) < k:
            extras = [i for i in range(len(ids)) if i not in set(pos)]
            pos = np.concatenate([pos, np.asarray(extras[: k - len(pos)])])
        return ids[pos[:k]].copy()

    p = prior_probs(prior, feat, qi, ids)
    x = pair_features(feat, qi, ids).astype(np.float64)
    if selector == "hard_negative":
        dense = feat.normalized[qi, ids, 0]
        disagreement = feat.normalized[qi, ids, 4]
        score = dense * (1 - p) + 0.25 * disagreement
        return ids[np.argsort(-score, kind="stable")[:k]].copy()
    if selector == "leverage":
        x = StandardScaler().fit_transform(x)
        gram_inv = np.linalg.pinv(x.T @ x + 0.1 * np.eye(x.shape[1]))
        lev = np.einsum("ij,jk,ik->i", x, gram_inv, x)
        return ids[np.argsort(-lev, kind="stable")[:k]].copy()
    if selector == "mmr_uncertainty":
        uncertainty = 1.0 - np.abs(2.0 * p - 1.0)
        return mmr_select(ids, uncertainty, feat.doc_embeddings, k)
    if selector == "mmr_relevance":
        return mmr_select(ids, p, feat.doc_embeddings, k)
    raise ValueError(selector)


def selection_key(seed: int, policy: str, budget: int, backend: str) -> str:
    return f"s{seed}__{policy}__b{budget}__{backend}"


def build_all_selections(
    ds: Dataset,
    feat: Features,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[int, LogisticRegression]]:
    all_sel: dict[str, dict[int, np.ndarray]] = {}
    splits: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    priors: dict[int, LogisticRegression] = {}
    for seed in SEEDS:
        train, val, test = split_queries(len(ds.queries), seed)
        splits[seed] = (train, val, test)
        prior = fit_cheap_prior(ds, feat, train, seed)
        priors[seed] = prior
        for policy, (_, allocation, selector) in POLICIES.items():
            weights = allocation_scores(allocation, feat, prior, train)
            for budget in BUDGETS:
                quotas = integer_allocation(weights, budget)
                for backend in BACKENDS:
                    key = selection_key(seed, policy, budget, backend)
                    mapping: dict[int, np.ndarray] = {}
                    for pos, qi0 in enumerate(train):
                        qi = int(qi0)
                        anchor = choose_anchor(ds, feat, qi)
                        ids = np.asarray(
                            [int(d) for d in feat.candidates[backend][qi] if int(d) != anchor],
                            dtype=np.int64,
                        )
                        mapping[qi] = select_docs(
                            selector, ids, feat, prior, qi, int(quotas[pos]), seed
                        )
                    all_sel[key] = mapping
    return all_sel, splits, priors


def teacher_union(
    ds: Dataset,
    feat: Features,
    selections: dict[str, dict[int, np.ndarray]],
    splits: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for seed, (train, _, _) in splits.items():
        for qi0 in train:
            qi = int(qi0)
            pairs.add((qi, choose_anchor(ds, feat, qi)))
    for mapping in selections.values():
        for qi, docs in mapping.items():
            pairs.update((int(qi), int(d)) for d in docs)
    return sorted(pairs)


@torch.inference_mode()
def score_teacher(
    ds: Dataset,
    pairs: list[tuple[int, int]],
    teacher_name: str,
    cache: Path,
    batch_size: int = 64,
) -> dict[tuple[int, int], float]:
    print(f"[{ds.name}] scoring {len(pairs):,} unique pairs with {teacher_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, cache_dir=cache / "hf")
    model = AutoModelForSequenceClassification.from_pretrained(teacher_name, cache_dir=cache / "hf")
    model.to("cpu").eval()
    out: dict[tuple[int, int], float] = {}
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        qs = [ds.queries[q] for q, _ in chunk]
        docs = [ds.texts[d] for _, d in chunk]
        tok = tokenizer(
            qs,
            docs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        logits = model(**tok).logits.squeeze(-1).detach().cpu().numpy().reshape(-1)
        for pair, score in zip(chunk, logits):
            out[pair] = float(score)
        if (start // batch_size) % 50 == 0:
            print(f"[{ds.name}] teacher {min(start + batch_size, len(pairs)):,}/{len(pairs):,}", flush=True)
    del model
    return out


def ndcg_at_k(order: np.ndarray, relevant: set[int], k: int = 10) -> float:
    hits = np.asarray([1.0 if int(d) in relevant else 0.0 for d in order[:k]], dtype=np.float64)
    denom = np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits / denom).sum())
    ideal_n = min(k, len(relevant))
    idcg = float((np.ones(ideal_n) / np.log2(np.arange(2, ideal_n + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(order: np.ndarray, relevant: set[int], k: int = 10) -> float:
    for rank, did in enumerate(order[:k], start=1):
        if int(did) in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(order: np.ndarray, relevant: set[int], k: int = 100) -> float:
    return len(set(map(int, order[:k])) & relevant) / max(1, len(relevant))


def metrics_for_scores(ds: Dataset, query_indices: Sequence[int], scores: np.ndarray) -> dict[str, float]:
    ndcg, mrr, recall = [], [], []
    for row, qi0 in enumerate(query_indices):
        qi = int(qi0)
        order = np.argsort(-scores[row], kind="stable")
        ndcg.append(ndcg_at_k(order, ds.qrels[qi], 10))
        mrr.append(mrr_at_k(order, ds.qrels[qi], 10))
        recall.append(recall_at_k(order, ds.qrels[qi], 100))
    return {"ndcg10": float(np.mean(ndcg)), "mrr10": float(np.mean(mrr)), "recall100": float(np.mean(recall))}


def trainable_top_layers(model: AutoModel, n_layers: int = 2) -> list[str]:
    for p in model.parameters():
        p.requires_grad = False
    layer_container = None
    for candidate in (
        getattr(getattr(model, "encoder", None), "layer", None),
        getattr(getattr(getattr(model, "transformer", None), "layer", None), "__iter__", None),
    ):
        if candidate is not None:
            layer_container = candidate
            break
    if layer_container is None and hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        layer_container = model.encoder.layer
    names: list[str] = []
    if layer_container is not None and not callable(layer_container):
        layers = list(layer_container)
        for layer in layers[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        names.append(f"last_{min(n_layers, len(layers))}_encoder_layers")
    else:
        # Generic fallback: unfreeze the final quarter of parameter tensors.
        params = list(model.named_parameters())
        for name, p in params[max(0, len(params) * 3 // 4) :]:
            p.requires_grad = True
        names.append("final_parameter_quarter")
    return names


def reset_model(model: AutoModel, base_state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(base_state, strict=True)
    model.to("cpu")


def encode_queries_grad(model: AutoModel, tokenizer: AutoTokenizer, texts: list[str]) -> torch.Tensor:
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    )
    out = model(**tok)
    return F.normalize(mean_pool(out.last_hidden_state, tok["attention_mask"]), p=2, dim=1)


def train_branch(
    ds: Dataset,
    feat: Features,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    base_state: dict[str, torch.Tensor],
    teacher: dict[tuple[int, int], float],
    selection: dict[int, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    steps: int,
    lr: float,
) -> dict[str, float]:
    reset_model(model, base_state)
    # Paired policy comparisons must share the same optimizer/dropout/query-batch RNG.
    # The caller therefore provides a seed that excludes policy identity.
    set_deterministic(seed)
    trained_parts = trainable_top_layers(model, 2)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(stable_int("train", ds.name, seed, len(selection)))
    model.train()
    final_loss = math.nan
    batch_size = min(8, len(train_idx))
    for step in range(steps):
        qbatch = rng.choice(train_idx, size=batch_size, replace=len(train_idx) < batch_size)
        lists: list[list[int]] = []
        targets: list[list[float]] = []
        for qi0 in qbatch:
            qi = int(qi0)
            anchor = choose_anchor(ds, feat, qi)
            docs = [anchor] + [int(d) for d in selection[qi] if int(d) != anchor]
            lists.append(docs)
            targets.append([teacher[(qi, d)] for d in docs])
        max_len = max(map(len, lists))
        doc_tensor = torch.zeros((batch_size, max_len, feat.doc_embeddings.shape[1]), dtype=torch.float32)
        teacher_tensor = torch.full((batch_size, max_len), -1e9, dtype=torch.float32)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for i, (docs, target) in enumerate(zip(lists, targets)):
            n = len(docs)
            doc_tensor[i, :n] = torch.from_numpy(feat.doc_embeddings[np.asarray(docs)])
            teacher_tensor[i, :n] = torch.tensor(target, dtype=torch.float32)
            mask[i, :n] = True
        qemb = encode_queries_grad(model, tokenizer, [ds.queries[int(q)] for q in qbatch])
        slogits = torch.einsum("bd,bld->bl", qemb, doc_tensor) / 0.05
        slogits = slogits.masked_fill(~mask, -1e9)
        target_prob = F.softmax(teacher_tensor / 2.0, dim=1)
        loss = -(target_prob * F.log_softmax(slogits, dim=1)).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        final_loss = float(loss.detach())

    model.eval()
    trained_val = encode_texts(model, tokenizer, [ds.queries[int(i)] for i in val_idx], batch_size=64)
    trained_test = encode_texts(model, tokenizer, [ds.queries[int(i)] for i in test_idx], batch_size=64)
    base_val = feat.query_embeddings[val_idx]
    base_test = feat.query_embeddings[test_idx]
    best_alpha, best_val = 0.0, -1.0
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = (1 - alpha) * base_val + alpha * trained_val
        q /= np.linalg.norm(q, axis=1, keepdims=True) + EPS
        m = metrics_for_scores(ds, val_idx, q @ feat.doc_embeddings.T)
        if m["ndcg10"] > best_val + 1e-12:
            best_val, best_alpha = m["ndcg10"], alpha
    qtest = (1 - best_alpha) * base_test + best_alpha * trained_test
    qtest /= np.linalg.norm(qtest, axis=1, keepdims=True) + EPS
    test_metrics = metrics_for_scores(ds, test_idx, qtest @ feat.doc_embeddings.T)
    return {
        **test_metrics,
        "val_ndcg10": best_val,
        "alpha": best_alpha,
        "final_loss": final_loss,
        "trainable": "+".join(trained_parts),
    }


def selection_diagnostics(
    ds: Dataset,
    feat: Features,
    selections: dict[str, dict[int, np.ndarray]],
    teacher: dict[tuple[int, int], float],
    splits: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, mapping in selections.items():
        m = re.fullmatch(r"s(\d+)__(.+)__b(\d+)__(exact|hnsw|ivf|hybrid)", key)
        assert m
        seed, policy, budget, backend = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
        train = splits[seed][0]
        positives = 0
        total = 0
        query_hits = 0
        diversity: list[float] = []
        teacher_mean: list[float] = []
        teacher_max: list[float] = []
        unique_docs: set[int] = set()
        for qi0 in train:
            qi = int(qi0)
            ids = mapping[qi]
            total += len(ids)
            hit = len(set(map(int, ids)) & ds.qrels[qi])
            positives += hit
            query_hits += int(hit > 0)
            unique_docs.update(map(int, ids))
            if len(ids) > 1:
                sim = feat.doc_embeddings[ids] @ feat.doc_embeddings[ids].T
                tri = sim[np.triu_indices(len(ids), 1)]
                diversity.append(float(1.0 - tri.mean()))
            ts = [teacher[(qi, int(d))] for d in ids]
            if ts:
                teacher_mean.append(float(np.mean(ts)))
                teacher_max.append(float(np.max(ts)))
        rows.append(
            {
                "dataset": ds.name,
                "seed": seed,
                "policy": policy,
                "direction": POLICIES[policy][0],
                "budget": budget,
                "backend": backend,
                "selected_pairs": total,
                "positive_yield": positives / max(1, total),
                "query_positive_coverage": query_hits / max(1, len(train)),
                "semantic_diversity": float(np.mean(diversity)) if diversity else 0.0,
                "unique_doc_fraction": len(unique_docs) / max(1, total),
                "mean_teacher_score": float(np.mean(teacher_mean)) if teacher_mean else math.nan,
                "mean_query_max_teacher": float(np.mean(teacher_max)) if teacher_max else math.nan,
                "candidate_exact_recall": feat.candidate_exact_recall[backend],
                "candidate_qrel_recall": feat.candidate_qrel_recall[backend],
            }
        )
    return pd.DataFrame(rows)


def branches_to_train() -> list[tuple[int, str, int, str]]:
    out: set[tuple[int, str, int, str]] = set()
    for seed in SEEDS:
        for policy in POLICIES:
            out.add((seed, policy, 12, "exact"))
        for policy in ("top_base", "difficulty_leverage"):
            for budget in BUDGETS:
                for backend in BACKENDS:
                    out.add((seed, policy, budget, backend))
    return sorted(out)


def run_dataset(args: argparse.Namespace) -> None:
    start = time.time()
    outdir = Path(args.output) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    set_deterministic(0)
    ds = load_beir(args.dataset, cache, max_queries=args.max_queries)
    feat, tokenizer, model = build_features(
        ds, args.student_model, cache, candidate_k=args.candidate_k
    )
    selections, splits, priors = build_all_selections(ds, feat)
    pairs = teacher_union(ds, feat, selections, splits)
    teacher = score_teacher(ds, pairs, args.teacher_model, cache, batch_size=args.teacher_batch)

    diagnostics = selection_diagnostics(ds, feat, selections, teacher, splits)
    diagnostics.to_csv(outdir / "selection_diagnostics.csv", index=False)

    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    baselines: list[dict[str, object]] = []
    for seed in SEEDS:
        train, val, test = splits[seed]
        scores = feat.query_embeddings[test] @ feat.doc_embeddings.T
        baselines.append({"dataset": ds.name, "seed": seed, **metrics_for_scores(ds, test, scores)})
    pd.DataFrame(baselines).to_csv(outdir / "baselines.csv", index=False)

    rows: list[dict[str, object]] = []
    branches = branches_to_train()
    print(f"[{ds.name}] training {len(branches)} query-side post-training branches", flush=True)
    for branch_i, (seed, policy, budget, backend) in enumerate(branches, start=1):
        train, val, test = splits[seed]
        key = selection_key(seed, policy, budget, backend)
        result = train_branch(
            ds,
            feat,
            tokenizer,
            model,
            base_state,
            teacher,
            selections[key],
            train,
            val,
            test,
            seed=stable_int("paired-train", ds.name, seed, budget, backend),
            steps=args.train_steps,
            lr=args.lr,
        )
        row: dict[str, object] = {
            "dataset": ds.name,
            "seed": seed,
            "policy": policy,
            "direction": POLICIES[policy][0],
            "budget": budget,
            "backend": backend,
            "teacher_calls": int(sum(len(x) for x in selections[key].values()) + len(train)),
            **result,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(outdir / "training_runs.partial.csv", index=False)
        print(
            f"[{ds.name}] {branch_i}/{len(branches)} {policy} b={budget} {backend}: "
            f"nDCG@10={result['ndcg10']:.4f} alpha={result['alpha']}",
            flush=True,
        )
    runs = pd.DataFrame(rows)
    runs.to_csv(outdir / "training_runs.csv", index=False)
    (outdir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": ds.name,
                "documents": len(ds.doc_ids),
                "queries": len(ds.query_ids),
                "student_model": args.student_model,
                "teacher_model": args.teacher_model,
                "candidate_k": args.candidate_k,
                "budgets": list(BUDGETS),
                "backends": list(BACKENDS),
                "policies": POLICIES,
                "seeds": list(SEEDS),
                "train_steps": args.train_steps,
                "lr": args.lr,
                "unique_teacher_pairs": len(pairs),
                "candidate_exact_recall": feat.candidate_exact_recall,
                "candidate_qrel_recall": feat.candidate_qrel_recall,
                "runtime_seconds": time.time() - start,
                "python": sys.version,
                "torch": torch.__version__,
            },
            indent=2,
        )
        + "\n"
    )


def paired_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean()) if len(values) else math.nan
    if len(values) < 2:
        return mean, math.nan, math.nan
    se = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def aggregate(args: argparse.Namespace) -> None:
    root = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_frames, diag_frames, base_frames = [], [], []
    metadata = []
    for dataset in DATASET_URL:
        droot_candidates = list(root.glob(f"**/{dataset}"))
        droot = next((p for p in droot_candidates if (p / "training_runs.csv").exists()), None)
        if droot is None:
            raise FileNotFoundError(f"Could not find completed artifact for {dataset} under {root}")
        run_frames.append(pd.read_csv(droot / "training_runs.csv"))
        diag_frames.append(pd.read_csv(droot / "selection_diagnostics.csv"))
        base_frames.append(pd.read_csv(droot / "baselines.csv"))
        metadata.append(json.loads((droot / "metadata.json").read_text()))
    runs = pd.concat(run_frames, ignore_index=True)
    diagnostics = pd.concat(diag_frames, ignore_index=True)
    baselines = pd.concat(base_frames, ignore_index=True)
    runs.to_csv(out / "ALL_TRAINING_RUNS.csv", index=False)
    diagnostics.to_csv(out / "ALL_SELECTION_DIAGNOSTICS.csv", index=False)
    baselines.to_csv(out / "BASELINES.csv", index=False)

    main = runs[(runs.budget == 12) & (runs.backend == "exact")].copy()
    control = main[main.policy == "top_base"][["dataset", "seed", "ndcg10", "mrr10", "recall100"]].rename(
        columns={"ndcg10": "control_ndcg10", "mrr10": "control_mrr10", "recall100": "control_recall100"}
    )
    main = main.merge(control, on=["dataset", "seed"], how="left")
    main["delta_ndcg10"] = main.ndcg10 - main.control_ndcg10
    main["delta_mrr10"] = main.mrr10 - main.control_mrr10
    main["delta_recall100"] = main.recall100 - main.control_recall100
    summary_rows = []
    for policy, grp in main.groupby("policy"):
        mean, low, high = paired_interval(grp.delta_ndcg10.to_numpy())
        per_dataset = grp.groupby("dataset").delta_ndcg10.mean().to_dict()
        summary_rows.append(
            {
                "policy": policy,
                "direction": POLICIES[policy][0],
                "runs": len(grp),
                "mean_ndcg10": grp.ndcg10.mean(),
                "mean_delta_control_ndcg10": mean,
                "ci95_low": low,
                "ci95_high": high,
                "win_rate": float((grp.delta_ndcg10 > 0).mean()),
                "worst_dataset_delta": min(per_dataset.values()),
                "mean_delta_mrr10": grp.delta_mrr10.mean(),
                "mean_delta_recall100": grp.delta_recall100.mean(),
                **{f"delta_{k}": v for k, v in per_dataset.items()},
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_delta_control_ndcg10", "worst_dataset_delta"], ascending=False
    )
    summary.to_csv(out / "POLICY_SUMMARY.csv", index=False)

    promoted = summary[~summary.policy.isin(["top_base", "random"])].copy()
    if promoted.empty:
        raise RuntimeError("No promoted non-control policy was evaluated")
    winner = str(promoted.iloc[0].policy)
    robustness_policy = "difficulty_leverage"
    robust = runs[runs.policy.isin(["top_base", robustness_policy])].copy()
    ctrl = robust[robust.policy == "top_base"][["dataset", "seed", "budget", "backend", "ndcg10"]].rename(
        columns={"ndcg10": "control_ndcg10"}
    )
    robust = robust.merge(ctrl, on=["dataset", "seed", "budget", "backend"], how="left")
    robust["delta_ndcg10"] = robust.ndcg10 - robust.control_ndcg10
    robust_summary = (
        robust[robust.policy == robustness_policy]
        .groupby(["budget", "backend"], as_index=False)
        .agg(
            mean_ndcg10=("ndcg10", "mean"),
            mean_delta_control_ndcg10=("delta_ndcg10", "mean"),
            std_delta=("delta_ndcg10", "std"),
            win_rate=("delta_ndcg10", lambda x: float((x > 0).mean())),
            runs=("delta_ndcg10", "size"),
        )
    )
    robust_summary.to_csv(out / "WINNER_ROBUSTNESS.csv", index=False)

    diag_summary = (
        diagnostics.groupby(["policy", "budget", "backend"], as_index=False)
        .agg(
            positive_yield=("positive_yield", "mean"),
            query_positive_coverage=("query_positive_coverage", "mean"),
            semantic_diversity=("semantic_diversity", "mean"),
            unique_doc_fraction=("unique_doc_fraction", "mean"),
            mean_teacher_score=("mean_teacher_score", "mean"),
            mean_query_max_teacher=("mean_query_max_teacher", "mean"),
            candidate_exact_recall=("candidate_exact_recall", "mean"),
            candidate_qrel_recall=("candidate_qrel_recall", "mean"),
        )
    )
    diag_summary.to_csv(out / "SELECTION_DIAGNOSTIC_SUMMARY.csv", index=False)

    top5 = promoted.head(5).copy()
    top5.to_csv(out / "FINAL_TOP5.csv", index=False)

    base_mean = baselines.groupby("dataset", as_index=False).mean(numeric_only=True)
    report = []
    report.append("# Final empirical evaluation: budget-aware feedback selection for retrieval post-training\n")
    report.append("## Executive conclusion\n")
    winner_row = promoted.iloc[0]
    report.append(
        f"The original ANN-backend-only question should remain an ablation. The completed dense validation selects **{POLICIES[winner][0]}** as the leading policy under the fixed central setting, with mean paired ΔNDCG@10 **{winner_row.mean_delta_control_ndcg10:+.4f}** against standard top-candidate feedback across 3 datasets × 3 splits. Its descriptive 95% interval is [{winner_row.ci95_low:+.4f}, {winner_row.ci95_high:+.4f}], so this is a promoted direction rather than a universal superiority claim.\n"
    )
    report.append("The practical project thesis is therefore: **how should a fixed budget of expensive relevance/reward judgments be allocated across queries and candidate documents to maximize retrieval post-training?**\n")
    report.append("## Completed experiment\n")
    report.append("- Datasets: SciFact, NFCorpus and ArguAna (up to 300 judged queries each).")
    report.append("- Student: transformer bi-encoder with query-side post-training and frozen document embeddings.")
    report.append("- Teacher: MS MARCO MiniLM cross-encoder; labels are observed only after selection.")
    report.append("- Candidate generators: exact dense, HNSW, IVF-Flat and lexical+dense hybrid.")
    report.append("- Feedback budgets: 4, 12 and 32 selected documents per query on average.")
    report.append("- Policies: five promoted mechanisms plus top-candidate and random controls.")
    report.append("- Repeated splits: seeds 0, 1 and 2; full-corpus exact evaluation on held-out queries.\n")
    report.append("## Central policy comparison (exact candidates, budget 12)\n")
    report.append("| Rank | Direction | Mean NDCG@10 | Δ vs top-candidate | 95% interval | Worst-dataset Δ | Win rate |")
    report.append("|---:|---|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(summary.itertuples(index=False), start=1):
        report.append(
            f"| {rank} | {row.direction} | {row.mean_ndcg10:.4f} | {row.mean_delta_control_ndcg10:+.4f} | [{row.ci95_low:+.4f}, {row.ci95_high:+.4f}] | {row.worst_dataset_delta:+.4f} | {row.win_rate:.0%} |"
        )
    report.append("\n## Untouched dense-retriever controls\n")
    report.append("| Dataset | NDCG@10 | MRR@10 | Recall@100 |")
    report.append("|---|---:|---:|---:|")
    for row in base_mean.itertuples(index=False):
        report.append(f"| {row.dataset} | {row.ndcg10:.4f} | {row.mrr10:.4f} | {row.recall100:.4f} |")
    report.append("\n## Pre-registered candidate robustness across budget and candidate generator\n")
    report.append(f"Pre-registered robustness candidate from the 100-direction screen: **{POLICIES[robustness_policy][0]}**.\n")
    report.append("| Budget | Backend | ΔNDCG@10 vs matched top-candidate | Win rate |")
    report.append("|---:|---|---:|---:|")
    for row in robust_summary.sort_values(["budget", "backend"]).itertuples(index=False):
        report.append(
            f"| {int(row.budget)} | {row.backend} | {row.mean_delta_control_ndcg10:+.4f} | {row.win_rate:.0%} |"
        )
    report.append("\n## Final five directions\n")
    for i, row in enumerate(top5.itertuples(index=False), start=1):
        report.append(
            f"{i}. **{row.direction}** — paired ΔNDCG@10 {row.mean_delta_control_ndcg10:+.4f}; worst-dataset Δ {row.worst_dataset_delta:+.4f}."
        )
    report.append("\n## Claim boundary\n")
    report.append("- This is transformer query-side post-training, not a claim about fully updating both query and document encoders.")
    report.append("- Three small-to-medium BEIR domains and three splits support direction selection, not universal dominance.")
    report.append("- The ANN family is evaluated as a candidate-source factor; the scientific object is the feedback allocation policy.")
    report.append("- A positive mean with an interval spanning zero is treated as a hypothesis for the full MS MARCO-scale experiment, not as statistical proof.\n")
    report.append("## Final project decision\n")
    report.append("Proceed with **Budget-Aware Feedback Selection for Retrieval Post-Training**. Keep ANN backend, teacher choice, feedback budget and loss formulation as controlled axes. Do not use ‘which ANN backend is best for post-training?’ as the standalone thesis.\n")
    (out / "FINAL_REPORT.md").write_text("\n".join(report) + "\n")

    proposal = f"""# IRE Final Project Proposal

## Brief of the idea

This project studies **budget-aware feedback selection for retrieval post-training**. A retriever produces candidate documents, but only a limited number of query-document pairs can be scored by an expensive cross-encoder, LLM reward model, or human assessor. The central question is:

> Under a fixed reward-scoring budget, which queries and candidate documents should receive feedback so that the post-trained retriever improves most reliably?

Candidate generation—including exact search, ANN methods and lexical-dense hybrids—is treated as one controlled component. The main research object is the policy that allocates feedback across queries and documents.

## 1. What is the idea?

We will compare standard top-ranked feedback against selection policies based on uncertainty, retriever disagreement, semantic diversity, hard-negative likelihood and feature-space information. The same retriever, teacher, budget, loss and evaluation protocol will be used for fair comparisons.

### a. Why is it important?

High-quality relevance feedback is expensive and retrieval datasets contain labels for only a small fraction of possible query-document pairs. A document that is not selected cannot be scored, and feedback that is redundant or uninformative wastes teacher or human-assessment calls. A successful policy would improve search and RAG systems for the same feedback cost, or reach a target quality with fewer reward-model calls.

### b. Relevant literature and gap

ANCE establishes that retrieved negatives affect dense-retriever training. RocketQA uses a strong cross-encoder to filter retrieved candidates. Active-learning, hard-negative mining, diversity-aware retrieval and experimental-design methods each provide possible selection principles. Existing work usually proposes one mining rule or optimizes inference-time retrieval; it does not provide a controlled comparison of **where a fixed post-training feedback budget should be spent across both queries and documents**, while also varying candidate generation.

Our preliminary ANN pilot found that HNSW and IVF exposed different passage identities, but their cross-encoder supervision was almost identical and both trained branches degraded relative to the untouched ANCE checkpoint. The completed 100-direction proxy screen and dense follow-up therefore motivate a broader feedback-allocation thesis rather than an ANN-family-only question.

## 2. What is the plan?

1. **Controlled baselines:** untouched retriever, top-ranked feedback and random feedback.
2. **Selection policies:** evaluate the five promoted mechanisms from the empirical screen, headed by **{POLICIES[winner][0]}**.
3. **Budget ablation:** compare 4, 12 and 32 reward calls per query on average.
4. **Candidate-source ablation:** exact dense, HNSW, IVF-Flat and lexical-dense hybrid candidates.
5. **Reward feedback:** score selected pairs using a cross-encoder; record score distributions, positive yield, diversity and redundancy.
6. **Post-training:** optimize a dense retriever under the same loss and training schedule for all policies.
7. **Evaluation:** NDCG@10, MRR@10, Recall@100, teacher calls, runtime, candidate recall, policy overlap and repeated-seed uncertainty.
8. **Scale-up:** validate the final policy on the existing 50k MS MARCO setup and at least two BEIR domains.

## Empirical gate already completed

The dense follow-up compared seven policies, three budgets, four candidate generators, three datasets and three repeated splits. At the central exact-search budget-12 setting, the leading policy achieved mean paired ΔNDCG@10 **{winner_row.mean_delta_control_ndcg10:+.4f}** against standard top-candidate feedback; its interval [{winner_row.ci95_low:+.4f}, {winner_row.ci95_high:+.4f}] requires conservative interpretation. These results are used to select the final direction rather than to claim universal superiority.

## Team size and resources

The work separates naturally into literature and experiment design, retrieval/teacher infrastructure, and statistical evaluation. It is feasible with the available GPU cluster for the final scale-up; the completed screening and dense validation establish the pipeline and reduce expensive runs to a small, justified shortlist.
"""
    (out / "FINAL_PROPOSAL.md").write_text(proposal)
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "winner": winner,
                "winner_direction": POLICIES[winner][0],
                "robustness_policy": robustness_policy,
                "robustness_direction": POLICIES[robustness_policy][0],
                "datasets": list(DATASET_URL),
                "policies": POLICIES,
                "budgets": list(BUDGETS),
                "backends": list(BACKENDS),
                "seeds": list(SEEDS),
                "dataset_metadata": metadata,
            },
            indent=2,
        )
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run-dataset")
    r.add_argument("--dataset", choices=sorted(DATASET_URL), required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--student-model", default="sentence-transformers/all-MiniLM-L6-v2")
    r.add_argument("--teacher-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    r.add_argument("--max-queries", type=int, default=300)
    r.add_argument("--candidate-k", type=int, default=100)
    r.add_argument("--teacher-batch", type=int, default=64)
    r.add_argument("--train-steps", type=int, default=40)
    r.add_argument("--lr", type=float, default=2e-5)
    a = sub.add_parser("aggregate")
    a.add_argument("--input", required=True)
    a.add_argument("--output", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "run-dataset":
        run_dataset(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
