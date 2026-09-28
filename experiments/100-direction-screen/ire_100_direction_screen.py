#!/usr/bin/env python3
"""Empirical screen of 100 broad feedback-selection directions for retrieval post-training.

The screen is intentionally a *sanity experiment*, not a paper-level claim. It fixes a
lightweight retrieval/ranking model and compares 25 selection primitives under four
query-budget allocation policies (100 configurations total) on two public IR datasets.

No expensive teacher is called before selection. After a pair is selected, its Cranfield/
SciFact relevance judgment is revealed and used as the feedback label. Every query also
gets one fixed seed positive outside the feedback budget, matching common dense-retriever
training setups where at least one positive is already known.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

EPS = 1e-8


@dataclass
class IRDataset:
    name: str
    doc_ids: list[str]
    doc_titles: list[str]
    doc_texts: list[str]
    query_ids: list[str]
    query_texts: list[str]
    qrels: dict[int, set[int]]


@dataclass
class FeatureBundle:
    names: list[str]
    features: np.ndarray  # [Q, D, F], float32
    base: np.ndarray  # [Q, D], float32
    raw_scores: dict[str, np.ndarray]
    doc_repr: np.ndarray  # [D, K], unit-normalized
    query_repr: np.ndarray  # [Q, K], unit-normalized
    candidates: list[np.ndarray]


@dataclass(frozen=True)
class SelectorSpec:
    key: str
    title: str
    hypothesis: str


@dataclass(frozen=True)
class AllocationSpec:
    key: str
    title: str
    hypothesis: str


def stable_int(*parts: object) -> int:
    text = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16) % (2**32 - 1)


def download(url: str, path: Path, retries: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ire-screen/1.0"})
            with urllib.request.urlopen(req, timeout=90) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            return path
        except Exception as exc:  # pragma: no cover - network branch
            last = exc
            if path.exists():
                path.unlink()
            time.sleep(2**attempt)
    raise RuntimeError(f"Could not download {url}: {last}")


def _tag(block: str, name: str) -> str:
    m = re.search(fr"<{name}>\s*(.*?)\s*</{name}>", block, flags=re.I | re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def load_cranfield(cache: Path) -> IRDataset:
    base = "https://raw.githubusercontent.com/oussbenk/cranfield-trec-dataset/main"
    docs_p = download(f"{base}/cran.all.1400.xml", cache / "cranfield" / "docs.xml")
    q_p = download(f"{base}/cran.qry.xml", cache / "cranfield" / "queries.xml")
    r_p = download(f"{base}/cranqrel.trec.txt", cache / "cranfield" / "qrels.txt")

    docs_raw = docs_p.read_text(encoding="utf-8", errors="replace")
    doc_blocks = re.findall(r"<doc>(.*?)</doc>", docs_raw, flags=re.I | re.S)
    doc_ids, titles, texts = [], [], []
    for b in doc_blocks:
        did = _tag(b, "docno")
        title = _tag(b, "title")
        body = _tag(b, "text")
        if did:
            doc_ids.append(did)
            titles.append(title)
            texts.append((title + " " + body).strip())

    queries_raw = q_p.read_text(encoding="utf-8", errors="replace")
    q_blocks = re.findall(r"<top>(.*?)</top>", queries_raw, flags=re.I | re.S)
    query_ids, query_texts = [], []
    for b in q_blocks:
        qid = _tag(b, "num")
        text = _tag(b, "title")
        if qid:
            query_ids.append(qid)
            query_texts.append(text)

    did_to_i = {x: i for i, x in enumerate(doc_ids)}
    qid_to_i = {x: i for i, x in enumerate(query_ids)}
    qrels: dict[int, set[int]] = defaultdict(set)
    for line in r_p.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 4 and p[0] in qid_to_i and p[2] in did_to_i:
            try:
                rel = float(p[3])
            except ValueError:
                continue
            if rel > 0:
                qrels[qid_to_i[p[0]]].add(did_to_i[p[2]])
    keep = [i for i in range(len(query_ids)) if qrels.get(i)]
    remap = {old: new for new, old in enumerate(keep)}
    return IRDataset(
        name="cranfield",
        doc_ids=doc_ids,
        doc_titles=titles,
        doc_texts=texts,
        query_ids=[query_ids[i] for i in keep],
        query_texts=[query_texts[i] for i in keep],
        qrels={remap[i]: qrels[i] for i in keep},
    )


def load_scifact(cache: Path) -> IRDataset:
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    zpath = download(url, cache / "scifact.zip")
    root = cache / "scifact_unzipped"
    marker = root / ".done"
    if not marker.exists():
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(root)
        marker.write_text("ok")
    candidates = [root / "scifact", root]
    data_root = next((p for p in candidates if (p / "corpus.jsonl").exists()), None)
    if data_root is None:
        raise FileNotFoundError("SciFact archive layout not recognized")

    doc_ids, titles, texts = [], [], []
    with (data_root / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            doc_ids.append(str(x["_id"]))
            title = str(x.get("title", ""))
            body = str(x.get("text", ""))
            titles.append(title)
            texts.append((title + " " + body).strip())
    all_queries: dict[str, str] = {}
    with (data_root / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            all_queries[str(x["_id"])] = str(x["text"])
    did_to_i = {x: i for i, x in enumerate(doc_ids)}
    qrel_file = data_root / "qrels" / "test.tsv"
    qrels_by_id: dict[str, set[int]] = defaultdict(set)
    with qrel_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for x in reader:
            qid = str(x.get("query-id") or x.get("query_id") or x.get("qid"))
            did = str(x.get("corpus-id") or x.get("corpus_id") or x.get("docid"))
            score = float(x.get("score", 0))
            if score > 0 and did in did_to_i:
                qrels_by_id[qid].add(did_to_i[did])
    query_ids = [qid for qid in all_queries if qrels_by_id.get(qid)]
    query_texts = [all_queries[qid] for qid in query_ids]
    qrels = {i: qrels_by_id[qid] for i, qid in enumerate(query_ids)}
    return IRDataset("scifact", doc_ids, titles, texts, query_ids, query_texts, qrels)


def load_synthetic(seed: int = 0) -> IRDataset:
    """Tiny deterministic corpus for local smoke tests only."""
    rng = np.random.default_rng(seed)
    topics = ["aero heat wing", "graph neural retrieval", "medical trial evidence", "climate ocean model"]
    docs, titles, doc_ids = [], [], []
    labels = []
    for t, words in enumerate(topics):
        for j in range(35):
            noise = " ".join(f"n{rng.integers(30)}" for _ in range(8))
            title = f"{words} study {j}"
            titles.append(title)
            docs.append(f"{title} {words} {words} {noise}")
            doc_ids.append(str(len(doc_ids)))
            labels.append(t)
    query_ids, query_texts, qrels = [], [], {}
    for i in range(40):
        t = i % len(topics)
        query_ids.append(str(i))
        query_texts.append(topics[t] + " evidence")
        rel = {j for j, x in enumerate(labels) if x == t}
        qrels[i] = rel
    return IRDataset("synthetic", doc_ids, titles, docs, query_ids, query_texts, qrels)


def row_minmax(x: np.ndarray) -> np.ndarray:
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return ((x - lo) / (hi - lo + EPS)).astype(np.float32)


def row_rank_score(x: np.ndarray) -> np.ndarray:
    q, d = x.shape
    order = np.argsort(-x, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(q)[:, None]
    ranks[rows, order] = np.arange(d)[None, :]
    return (1.0 - ranks / max(d - 1, 1)).astype(np.float32)


def topk_idx(row: np.ndarray, k: int) -> np.ndarray:
    k = min(k, row.size)
    if k == row.size:
        return np.argsort(-row, kind="stable")
    part = np.argpartition(-row, k - 1)[:k]
    return part[np.argsort(-row[part], kind="stable")]


def bm25_scores(doc_texts: list[str], query_texts: list[str], max_features: int = 25000) -> tuple[np.ndarray, np.ndarray]:
    vec = CountVectorizer(stop_words="english", min_df=2, max_features=max_features)
    dmat = vec.fit_transform(doc_texts).tocsr().astype(np.float32)
    qmat = vec.transform(query_texts).tocsr().astype(np.float32)
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


def build_features(ds: IRDataset, seed: int, candidate_k: int = 240) -> FeatureBundle:
    print(f"[{ds.name}] building features for {len(ds.query_ids)} queries x {len(ds.doc_ids)} docs", flush=True)
    docs, queries = ds.doc_texts, ds.query_texts
    max_word = 30000 if len(docs) > 2000 else 20000
    word_vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_features=max_word,
        sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    d_word = word_vec.fit_transform(docs)
    q_word = word_vec.transform(queries)
    word = (q_word @ d_word.T).toarray().astype(np.float32)

    title_vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=2, max_features=15000,
        sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    try:
        d_title = title_vec.fit_transform(ds.doc_titles)
        q_title = title_vec.transform(queries)
        title = (q_title @ d_title.T).toarray().astype(np.float32)
    except ValueError:
        title = word.copy()

    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=22000,
        sublinear_tf=True, norm="l2", dtype=np.float32,
    )
    d_char = char_vec.fit_transform(docs)
    q_char = char_vec.transform(queries)
    char = (q_char @ d_char.T).toarray().astype(np.float32)

    bm25, coverage = bm25_scores(docs, queries, max_features=25000)

    n_comp = int(max(8, min(64, d_word.shape[0] - 1, d_word.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    d_lsa = svd.fit_transform(d_word).astype(np.float32)
    q_lsa = svd.transform(q_word).astype(np.float32)
    d_lsa /= np.linalg.norm(d_lsa, axis=1, keepdims=True) + EPS
    q_lsa /= np.linalg.norm(q_lsa, axis=1, keepdims=True) + EPS
    lsa = (q_lsa @ d_lsa.T).astype(np.float32)

    raw = {"word": word, "bm25": bm25, "char": char, "lsa": lsa, "title": title}
    norm = {k: row_minmax(v) for k, v in raw.items()}
    rank = {k: row_rank_score(v) for k, v in raw.items()}
    stack = np.stack([norm[k] for k in ["word", "bm25", "char", "lsa", "title"]], axis=-1)
    disagreement = stack.std(axis=-1).astype(np.float32)
    mean_score = stack.mean(axis=-1).astype(np.float32)
    max_score = stack.max(axis=-1).astype(np.float32)
    min_score = stack.min(axis=-1).astype(np.float32)
    sparse_dense_gap = np.abs(norm["bm25"] - norm["lsa"]).astype(np.float32)
    word_char_gap = np.abs(norm["word"] - norm["char"]).astype(np.float32)
    rrf = np.zeros_like(word, dtype=np.float32)
    for k in raw:
        # rank score is top=1, convert back to rank-like denominator.
        rr = 1.0 + (1.0 - rank[k]) * max(word.shape[1] - 1, 1)
        rrf += 1.0 / (60.0 + rr)
    rrf = row_minmax(rrf)

    base = (
        0.28 * norm["word"] + 0.28 * norm["bm25"] + 0.16 * norm["char"]
        + 0.18 * norm["lsa"] + 0.10 * norm["title"]
    ).astype(np.float32)
    base_rank = row_rank_score(base)
    top_margin = (base.max(axis=1, keepdims=True) - base).astype(np.float32)
    top_margin = 1.0 - row_minmax(top_margin)

    # Document length features, normalized to [0, 1].
    doc_len = np.asarray([len(re.findall(r"\w+", x)) for x in docs], dtype=np.float32)
    doc_len = (doc_len - doc_len.min()) / (doc_len.max() - doc_len.min() + EPS)
    q_len = np.asarray([len(re.findall(r"\w+", x)) for x in queries], dtype=np.float32)
    median_dl = float(np.median([max(1, len(re.findall(r"\w+", x))) for x in docs]))
    len_match = np.exp(-np.abs(np.log((doc_len[None, :] * median_dl + 1.0) / (q_len[:, None] + 1.0))))
    len_match = len_match.astype(np.float32)

    names = [
        "word", "bm25", "char", "lsa", "title", "rrf", "retriever_disagreement",
        "sparse_dense_gap", "word_char_gap", "query_coverage", "base_rank",
        "word_rank", "bm25_rank", "char_rank", "lsa_rank", "title_rank",
        "doc_length", "length_match", "top_margin", "mean_signal", "max_signal", "min_signal",
    ]
    arrays = [
        norm["word"], norm["bm25"], norm["char"], norm["lsa"], norm["title"], rrf,
        disagreement, sparse_dense_gap, word_char_gap, coverage, base_rank,
        rank["word"], rank["bm25"], rank["char"], rank["lsa"], rank["title"],
        np.broadcast_to(doc_len[None, :], base.shape), len_match, top_margin,
        mean_score, max_score, min_score,
    ]
    features = np.stack(arrays, axis=-1).astype(np.float32)

    candidates: list[np.ndarray] = []
    per_source = max(50, min(90, candidate_k // 3))
    for qi in range(base.shape[0]):
        pool: set[int] = set()
        for score in [base, norm["word"], norm["bm25"], norm["char"], norm["lsa"], rrf]:
            pool.update(topk_idx(score[qi], per_source).tolist())
        arr = np.fromiter(pool, dtype=np.int32)
        # Keep the strongest candidate_k by RRF/base blend.
        blend = 0.65 * rrf[qi, arr] + 0.35 * base[qi, arr]
        arr = arr[np.argsort(-blend, kind="stable")[:candidate_k]]
        candidates.append(arr.astype(np.int32))

    return FeatureBundle(names, features, base, norm, d_lsa, q_lsa, candidates)


def metric_one(scores: np.ndarray, rel: set[int], k_ndcg: int = 10, k_recall: int = 100) -> dict[str, float]:
    order = topk_idx(scores, max(k_ndcg, k_recall))
    top10 = order[:k_ndcg]
    hits = np.asarray([1.0 if int(d) in rel else 0.0 for d in top10])
    rr = 0.0
    hit_pos = np.flatnonzero(hits > 0)
    if hit_pos.size:
        rr = 1.0 / float(hit_pos[0] + 1)
    discount = 1.0 / np.log2(np.arange(2, k_ndcg + 2))
    dcg = float((hits * discount).sum())
    ideal_n = min(len(rel), k_ndcg)
    idcg = float(discount[:ideal_n].sum()) if ideal_n else 1.0
    recall = sum(int(d) in rel for d in order[:k_recall]) / max(len(rel), 1)
    return {"ndcg10": dcg / idcg, "mrr10": rr, "recall100": float(recall)}


def evaluate(score_matrix: np.ndarray, qids: Sequence[int], qrels: dict[int, set[int]]) -> dict[str, float]:
    vals = [metric_one(score_matrix[q], qrels[q]) for q in qids]
    return {k: float(np.mean([x[k] for x in vals])) for k in vals[0]}


def split_queries(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_teacher = max(10, int(round(0.20 * n)))
    n_student = max(15, int(round(0.45 * n)))
    n_val = max(8, int(round(0.15 * n)))
    # Keep at least 10 for test.
    if n_teacher + n_student + n_val > n - 10:
        n_student = max(10, n - 10 - n_teacher - n_val)
    a = n_teacher
    b = a + n_student
    c = b + n_val
    return idx[:a], idx[a:b], idx[b:c], idx[c:]


def training_rows(
    ds: IRDataset, fb: FeatureBundle, qids: Sequence[int], selected: dict[int, Sequence[int]] | None,
    include_all_candidates: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for q in qids:
        rel = ds.qrels[q]
        # The fixed seed positive is the highest-baseline known relevant document.
        seed_pos = max(rel, key=lambda d: float(fb.base[q, d]))
        docs: list[int] = [seed_pos]
        if include_all_candidates:
            docs.extend(int(x) for x in fb.candidates[q] if int(x) != seed_pos)
            # Ensure all judged positives can train the prior model.
            docs.extend(int(x) for x in rel if int(x) != seed_pos)
        elif selected is not None:
            docs.extend(int(x) for x in selected[q] if int(x) != seed_pos)
        docs = list(dict.fromkeys(docs))
        per_q = max(len(docs), 1)
        for d in docs:
            xs.append(fb.features[q, d])
            ys.append(1 if d in rel else 0)
            groups.append(1.0 / per_q)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int8), np.asarray(groups, dtype=np.float32)


def fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int) -> object | None:
    if len(np.unique(y)) < 2:
        return None
    # Class balance while retaining equal total weight per query.
    counts = np.bincount(y, minlength=2).astype(float)
    class_mult = np.where(counts > 0, len(y) / (2 * counts + EPS), 1.0)
    sw = weights * class_mult[y]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.7, max_iter=600, solver="liblinear", random_state=seed),
    )
    model.fit(x, y, logisticregression__sample_weight=sw)
    return model


def predict_model(model: object | None, feat: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(feat.shape[:-1], dtype=np.float32)
    shape = feat.shape[:-1]
    flat = feat.reshape(-1, feat.shape[-1])
    out = model.predict_proba(flat)[:, 1].reshape(shape)
    return out.astype(np.float32)


def fit_prior_committee(ds: IRDataset, fb: FeatureBundle, teacher_q: Sequence[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    x, y, w = training_rows(ds, fb, teacher_q, selected=None, include_all_candidates=True)
    base_model = fit_logistic(x, y, w, seed)
    prior = predict_model(base_model, fb.features)
    preds = []
    rng = np.random.default_rng(seed)
    for b in range(3):
        ids = rng.integers(0, len(y), len(y))
        model = fit_logistic(x[ids], y[ids], w[ids], seed + 101 + b)
        preds.append(predict_model(model, fb.features))
    committee_std = np.std(np.stack(preds, axis=0), axis=0).astype(np.float32)
    return prior, committee_std


SELECTORS = [
    SelectorSpec("random", "Random exploration", "A random but reproducible sample is a strong unbiased control."),
    SelectorSpec("top_base", "Top current-retriever candidates", "Exploit the current retriever's highest-scoring documents."),
    SelectorSpec("uniform_rank", "Uniform rank coverage", "Spread labels across the whole candidate ranking instead of only its head."),
    SelectorSpec("head_tail", "Head–tail mixture", "Label both likely positives/hard negatives and clear negatives."),
    SelectorSpec("top_bm25", "Lexical-first selection", "Spend feedback on BM25's strongest candidates."),
    SelectorSpec("top_lsa", "Semantic-first selection", "Spend feedback on latent-semantic candidates."),
    SelectorSpec("top_char", "Character-robust selection", "Favor candidates supported by character-level similarity."),
    SelectorSpec("top_title", "Title-focused selection", "Favor concise title evidence before full-document evidence."),
    SelectorSpec("rrf_consensus", "Multi-retriever consensus", "Reward candidates jointly supported by several retrievers."),
    SelectorSpec("retriever_disagreement", "Retriever disagreement", "Label pairs on which lexical, character and semantic retrievers disagree."),
    SelectorSpec("uncertainty", "Current-model uncertainty", "Label candidates nearest the current model's decision boundary."),
    SelectorSpec("committee", "Bootstrap committee disagreement", "Use epistemic disagreement across cheap student replicas."),
    SelectorSpec("false_positive", "Likely false-positive hunting", "Inspect high-ranked candidates the current model distrusts."),
    SelectorSpec("false_negative", "Likely false-negative hunting", "Inspect low-ranked candidates the current model considers promising."),
    SelectorSpec("coverage", "Query-term coverage", "Prefer documents covering more of the query's content."),
    SelectorSpec("rank_strata", "Rank-stratified sampling", "Guarantee supervision from each rank band."),
    SelectorSpec("mmr_relevance", "Relevance–diversity MMR", "Balance likely relevance with non-redundant documents."),
    SelectorSpec("mmr_uncertainty", "Uncertainty–diversity MMR", "Cover multiple uncertain semantic regions."),
    SelectorSpec("farthest_first", "Semantic k-center coverage", "Cover the candidate manifold with farthest-first traversal."),
    SelectorSpec("cluster_representatives", "Candidate-cluster representatives", "Take representative documents from multiple semantic clusters."),
    SelectorSpec("center_outlier_mix", "Typical–outlier mixture", "Label both dense-region representatives and unusual candidates."),
    SelectorSpec("hard_negative", "Hard-negative proxy", "Prioritize high retrieval score but low relevance probability."),
    SelectorSpec("leverage", "Feature-space leverage", "Select statistically influential examples in the student feature space."),
    SelectorSpec("d_optimal", "D-optimal experimental design", "Choose a set that maximizes feature-space information volume."),
    SelectorSpec("portfolio", "Diversified policy portfolio", "Hedge across exploitation, uncertainty, disagreement, coverage and diversity."),
]

ALLOCATIONS = [
    AllocationSpec("uniform", "Uniform per-query budget", "Give every query the same number of feedback calls."),
    AllocationSpec("difficulty", "Difficulty-adaptive budget", "Allocate more labels to low-confidence/high-entropy queries."),
    AllocationSpec("disagreement", "Disagreement-adaptive budget", "Allocate more labels where retrievers and model replicas disagree."),
    AllocationSpec("topic_balanced", "Topic-balanced budget", "Equalize total feedback across query clusters, not raw query counts."),
]


def allocate_integer(total: int, weights: np.ndarray, min_each: int, max_each: int) -> np.ndarray:
    n = len(weights)
    if n == 0:
        return np.array([], dtype=int)
    min_each = min(min_each, total // n)
    alloc = np.full(n, min_each, dtype=int)
    remaining = total - int(alloc.sum())
    weights = np.maximum(weights.astype(float), EPS)
    while remaining > 0:
        eligible = alloc < max_each
        if not eligible.any():
            break
        w = weights * eligible
        raw = remaining * w / (w.sum() + EPS)
        add = np.floor(raw).astype(int)
        add = np.minimum(add, max_each - alloc)
        if add.sum() == 0:
            frac = raw - np.floor(raw)
            idx = int(np.argmax(np.where(eligible, frac, -1)))
            alloc[idx] += 1
            remaining -= 1
        else:
            alloc += add
            remaining -= int(add.sum())
    return alloc


def query_budgets(
    allocation: str, student_q: Sequence[int], fb: FeatureBundle, prior: np.ndarray,
    committee: np.ndarray, per_query: int, max_each: int, seed: int,
) -> dict[int, int]:
    qs = np.asarray(student_q, dtype=int)
    total = len(qs) * per_query
    if allocation == "uniform":
        weights = np.ones(len(qs))
    elif allocation == "difficulty":
        vals = []
        for q in qs:
            c = fb.candidates[q]
            p = np.clip(prior[q, c], EPS, 1 - EPS)
            ent = -p * np.log(p) - (1 - p) * np.log(1 - p)
            confidence_gap = float(np.max(p) - np.partition(p, -2)[-2]) if len(p) > 1 else 0.0
            vals.append(float(ent.mean() + 0.5 * (1 - confidence_gap)))
        weights = np.asarray(vals)
    elif allocation == "disagreement":
        vals = []
        for q in qs:
            c = fb.candidates[q]
            vals.append(float(committee[q, c].mean() + fb.features[q, c, 6].mean()))
        weights = np.asarray(vals)
    elif allocation == "topic_balanced":
        k = min(8, max(2, len(qs) // 8))
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=256)
        labels = km.fit_predict(fb.query_repr[qs])
        counts = np.bincount(labels, minlength=k)
        weights = 1.0 / np.maximum(counts[labels], 1)
    else:
        raise KeyError(allocation)
    weights = (weights - weights.min()) / (weights.max() - weights.min() + EPS) + 0.2
    alloc = allocate_integer(total, weights, min_each=max(4, per_query // 2), max_each=max_each)
    return {int(q): int(n) for q, n in zip(qs, alloc)}


def _unique_order(items: Iterable[int], allowed: set[int]) -> list[int]:
    out, seen = [], set()
    for x in items:
        x = int(x)
        if x in allowed and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def greedy_mmr(c: np.ndarray, signal: np.ndarray, z: np.ndarray, n: int, lam: float = 0.55) -> list[int]:
    if len(c) <= n:
        return c.tolist()
    sig = signal[c]
    sig = (sig - sig.min()) / (sig.max() - sig.min() + EPS)
    selected_local: list[int] = []
    available = np.ones(len(c), dtype=bool)
    for _ in range(n):
        if not selected_local:
            j = int(np.argmax(sig))
        else:
            sims = z[c] @ z[c[selected_local]].T
            penalty = sims.max(axis=1)
            score = lam * sig - (1 - lam) * penalty
            score[~available] = -np.inf
            j = int(np.argmax(score))
        selected_local.append(j)
        available[j] = False
    return [int(c[j]) for j in selected_local]


def farthest_order(c: np.ndarray, z: np.ndarray, start_scores: np.ndarray, n: int) -> list[int]:
    if len(c) <= n:
        return c.tolist()
    first = int(np.argmax(start_scores[c]))
    sel = [first]
    available = np.ones(len(c), bool)
    available[first] = False
    min_dist = 1.0 - (z[c] @ z[c[first]])
    for _ in range(1, n):
        x = min_dist.copy()
        x[~available] = -np.inf
        j = int(np.argmax(x))
        sel.append(j)
        available[j] = False
        min_dist = np.minimum(min_dist, 1.0 - (z[c] @ z[c[j]]))
    return [int(c[j]) for j in sel]


def leverage_order(x: np.ndarray, n: int) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True) + EPS
    xc = xc / scale
    try:
        u, _, _ = np.linalg.svd(xc, full_matrices=False)
        rank = min(10, u.shape[1])
        lev = np.sum(u[:, :rank] ** 2, axis=1)
    except np.linalg.LinAlgError:
        lev = np.sum(xc**2, axis=1)
    return np.argsort(-lev, kind="stable")[:n]


def d_optimal_order(x: np.ndarray, n: int, ridge: float = 1.0) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    xc = xc / (x.std(axis=0, keepdims=True) + EPS)
    d = xc.shape[1]
    inv = np.eye(d, dtype=np.float64) / ridge
    chosen: list[int] = []
    available = np.ones(len(xc), bool)
    for _ in range(min(n, len(xc))):
        # Matrix determinant lemma: gain = log(1 + x^T A^-1 x).
        gains = np.einsum("ij,jk,ik->i", xc, inv, xc, optimize=True)
        gains[~available] = -np.inf
        j = int(np.argmax(gains))
        chosen.append(j)
        available[j] = False
        v = inv @ xc[j]
        inv = inv - np.outer(v, v) / (1.0 + float(xc[j] @ v))
    return np.asarray(chosen, dtype=int)


def selector_order(
    key: str, q: int, fb: FeatureBundle, prior: np.ndarray, committee: np.ndarray,
    max_n: int, seed: int,
) -> list[int]:
    c = fb.candidates[q]
    allowed = set(map(int, c))
    base = fb.base[q]
    feat = fb.features[q]
    rng = np.random.default_rng(stable_int(seed, q, key))
    entropy = -(np.clip(prior[q], EPS, 1 - EPS) * np.log(np.clip(prior[q], EPS, 1 - EPS)) +
                (1 - np.clip(prior[q], EPS, 1 - EPS)) * np.log(np.clip(1 - prior[q], EPS, 1 - EPS)))

    if key == "random":
        order = rng.permutation(c).tolist()
    elif key == "top_base":
        order = c[np.argsort(-base[c], kind="stable")].tolist()
    elif key == "uniform_rank":
        ranked = c[np.argsort(-base[c], kind="stable")]
        pos = np.unique(np.round(np.linspace(0, len(ranked) - 1, max_n)).astype(int))
        order = ranked[pos].tolist() + ranked.tolist()
    elif key == "head_tail":
        ranked = c[np.argsort(-base[c], kind="stable")]
        order = []
        for a, b in zip(ranked, ranked[::-1]):
            order.extend([int(a), int(b)])
    elif key == "top_bm25":
        order = c[np.argsort(-fb.raw_scores["bm25"][q, c], kind="stable")].tolist()
    elif key == "top_lsa":
        order = c[np.argsort(-fb.raw_scores["lsa"][q, c], kind="stable")].tolist()
    elif key == "top_char":
        order = c[np.argsort(-fb.raw_scores["char"][q, c], kind="stable")].tolist()
    elif key == "top_title":
        order = c[np.argsort(-fb.raw_scores["title"][q, c], kind="stable")].tolist()
    elif key == "rrf_consensus":
        order = c[np.argsort(-feat[c, 5], kind="stable")].tolist()
    elif key == "retriever_disagreement":
        order = c[np.argsort(-feat[c, 6], kind="stable")].tolist()
    elif key == "uncertainty":
        order = c[np.argsort(-entropy[c], kind="stable")].tolist()
    elif key == "committee":
        order = c[np.argsort(-committee[q, c], kind="stable")].tolist()
    elif key == "false_positive":
        score = base - prior[q]
        order = c[np.argsort(-score[c], kind="stable")].tolist()
    elif key == "false_negative":
        score = prior[q] - base
        order = c[np.argsort(-score[c], kind="stable")].tolist()
    elif key == "coverage":
        order = c[np.argsort(-feat[c, 9], kind="stable")].tolist()
    elif key == "rank_strata":
        ranked = c[np.argsort(-base[c], kind="stable")]
        bins = np.array_split(ranked, max_n)
        order = []
        for b in bins:
            if len(b):
                order.append(int(b[np.argmax(entropy[b])]))
        order += ranked.tolist()
    elif key == "mmr_relevance":
        order = greedy_mmr(c, base, fb.doc_repr, max_n, lam=0.55)
        order += c[np.argsort(-base[c], kind="stable")].tolist()
    elif key == "mmr_uncertainty":
        order = greedy_mmr(c, entropy, fb.doc_repr, max_n, lam=0.55)
        order += c[np.argsort(-entropy[c], kind="stable")].tolist()
    elif key == "farthest_first":
        order = farthest_order(c, fb.doc_repr, base, max_n)
        order += c.tolist()
    elif key == "cluster_representatives":
        k = min(max_n, max(2, min(12, len(c))))
        km = MiniBatchKMeans(n_clusters=k, random_state=stable_int(seed, q, key), n_init=3, batch_size=128)
        labels = km.fit_predict(fb.doc_repr[c])
        order = []
        for cluster in range(k):
            ids = np.flatnonzero(labels == cluster)
            if ids.size:
                center = km.cluster_centers_[cluster]
                j = ids[np.argmin(np.sum((fb.doc_repr[c[ids]] - center) ** 2, axis=1))]
                order.append(int(c[j]))
        order += c[np.argsort(-base[c], kind="stable")].tolist()
    elif key == "center_outlier_mix":
        center = fb.doc_repr[c].mean(axis=0)
        dist = np.linalg.norm(fb.doc_repr[c] - center, axis=1)
        central = c[np.argsort(dist, kind="stable")]
        outlier = c[np.argsort(-dist, kind="stable")]
        order = []
        for a, b in zip(central, outlier):
            order.extend([int(a), int(b)])
    elif key == "hard_negative":
        score = base * (1.0 - prior[q])
        order = c[np.argsort(-score[c], kind="stable")].tolist()
    elif key == "leverage":
        ids = leverage_order(feat[c], max_n)
        order = c[ids].tolist() + c.tolist()
    elif key == "d_optimal":
        ids = d_optimal_order(feat[c], max_n)
        order = c[ids].tolist() + c.tolist()
    elif key == "portfolio":
        source_orders = [
            c[np.argsort(-base[c], kind="stable")],
            c[np.argsort(-entropy[c], kind="stable")],
            c[np.argsort(-feat[c, 6], kind="stable")],
            np.asarray(farthest_order(c, fb.doc_repr, base, max_n), dtype=int),
            c[np.argsort(-feat[c, 9], kind="stable")],
        ]
        order = []
        for i in range(max(len(x) for x in source_orders)):
            for x in source_orders:
                if i < len(x):
                    order.append(int(x[i]))
    else:
        raise KeyError(key)
    out = _unique_order(order, allowed)
    if len(out) < len(c):
        out.extend(_unique_order(c.tolist(), allowed - set(out)))
    return out


def tune_mix(model_scores: np.ndarray, base: np.ndarray, val_q: Sequence[int], qrels: dict[int, set[int]]) -> tuple[float, dict[str, float]]:
    best_a, best_m = 0.0, evaluate(base, val_q, qrels)
    for a in [0.20, 0.40, 0.60, 0.80, 1.00]:
        score = (1 - a) * base + a * model_scores
        m = evaluate(score, val_q, qrels)
        if m["ndcg10"] > best_m["ndcg10"] + 1e-12:
            best_a, best_m = a, m
    return best_a, best_m


def selection_diagnostics(ds: IRDataset, fb: FeatureBundle, selected: dict[int, Sequence[int]], qids: Sequence[int]) -> dict[str, float]:
    total = pos = q_with_pos = 0
    ranks, diversities = [], []
    distinct = set()
    for q in qids:
        docs = list(map(int, selected[q]))
        total += len(docs)
        hits = sum(d in ds.qrels[q] for d in docs)
        pos += hits
        q_with_pos += int(hits > 0)
        distinct.update(docs)
        ranks.extend((1.0 - fb.features[q, docs, 10]).tolist())
        if len(docs) > 1:
            z = fb.doc_repr[docs]
            sim = z @ z.T
            tri = sim[np.triu_indices(len(docs), 1)]
            diversities.append(float(np.mean(1.0 - tri)))
    return {
        "feedback_positive_rate": pos / max(total, 1),
        "queries_with_positive_rate": q_with_pos / max(len(qids), 1),
        "mean_rank_percentile": float(np.mean(ranks)) if ranks else 0.0,
        "semantic_diversity": float(np.mean(diversities)) if diversities else 0.0,
        "unique_doc_fraction": len(distinct) / max(total, 1),
    }


def run_dataset(ds: IRDataset, seeds: Sequence[int], per_query_budget: int, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    # Feature extraction does not use labels; hold it fixed across split seeds.
    fb = build_features(ds, seed=0)
    selector_by_key = {x.key: x for x in SELECTORS}
    allocation_by_key = {x.key: x for x in ALLOCATIONS}

    for seed in seeds:
        print(f"[{ds.name}] seed {seed}", flush=True)
        teacher_q, student_q, val_q, test_q = split_queries(len(ds.query_ids), seed)
        prior, committee = fit_prior_committee(ds, fb, teacher_q, seed)
        base_val = evaluate(fb.base, val_q, ds.qrels)
        base_test = evaluate(fb.base, test_q, ds.qrels)
        prior_test = evaluate(prior, test_q, ds.qrels)
        candidate_recall = []
        for q in student_q:
            candidate_recall.append(len(set(map(int, fb.candidates[q])) & ds.qrels[q]) / max(len(ds.qrels[q]), 1))
        baseline_rows.append({
            "dataset": ds.name, "seed": seed, "n_queries": len(ds.query_ids), "n_docs": len(ds.doc_ids),
            **{f"base_{k}": v for k, v in base_test.items()},
            **{f"prior_{k}": v for k, v in prior_test.items()},
            "student_candidate_recall": float(np.mean(candidate_recall)),
            "teacher_queries": len(teacher_q), "student_queries": len(student_q),
            "val_queries": len(val_q), "test_queries": len(test_q),
        })

        max_each = per_query_budget * 2
        # Compute each primitive order once; allocation variants only change prefix length.
        orders: dict[str, dict[int, list[int]]] = {s.key: {} for s in SELECTORS}
        for s in SELECTORS:
            for q in student_q:
                orders[s.key][int(q)] = selector_order(s.key, int(q), fb, prior, committee, max_each, seed)

        for alloc in ALLOCATIONS:
            budgets = query_budgets(alloc.key, student_q, fb, prior, committee, per_query_budget, max_each, seed)
            for sel in SELECTORS:
                config_id = f"{alloc.key}__{sel.key}"
                selected: dict[int, Sequence[int]] = {}
                for q in student_q:
                    rel = ds.qrels[int(q)]
                    seed_pos = max(rel, key=lambda d: float(fb.base[int(q), d]))
                    order = [d for d in orders[sel.key][int(q)] if d != seed_pos]
                    selected[int(q)] = order[: budgets[int(q)]]

                x, y, w = training_rows(ds, fb, student_q, selected, include_all_candidates=False)
                model = fit_logistic(x, y, w, stable_int(seed, config_id))
                model_scores = predict_model(model, fb.features)
                alpha, val_metrics = tune_mix(model_scores, fb.base, val_q, ds.qrels)
                final_scores = (1 - alpha) * fb.base + alpha * model_scores
                test_metrics = evaluate(final_scores, test_q, ds.qrels)
                diag = selection_diagnostics(ds, fb, selected, student_q)
                all_rows.append({
                    "dataset": ds.name,
                    "seed": seed,
                    "config_id": config_id,
                    "direction": f"{alloc.title} + {sel.title}",
                    "selector": sel.key,
                    "selector_title": sel.title,
                    "allocation": alloc.key,
                    "allocation_title": alloc.title,
                    "selector_hypothesis": sel.hypothesis,
                    "allocation_hypothesis": alloc.hypothesis,
                    "budget_total": int(sum(budgets.values())),
                    "budget_mean": float(np.mean(list(budgets.values()))),
                    "label_positive_rate_train": float(y.mean()),
                    "alpha": alpha,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **{f"delta_base_{k}": test_metrics[k] - base_test[k] for k in test_metrics},
                    **diag,
                })
                print(
                    f"  {config_id:42s} ndcg={test_metrics['ndcg10']:.4f} "
                    f"Δbase={test_metrics['ndcg10']-base_test['ndcg10']:+.4f} a={alpha:.1f}",
                    flush=True,
                )

    runs = pd.DataFrame(all_rows)
    baselines = pd.DataFrame(baseline_rows)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / f"{ds.name}_runs.csv", index=False)
    baselines.to_csv(output / f"{ds.name}_baselines.csv", index=False)
    return runs, baselines


def aggregate_results(runs: pd.DataFrame) -> pd.DataFrame:
    control = runs[runs.config_id == "uniform__top_base"][
        ["dataset", "seed", "test_ndcg10", "test_mrr10", "test_recall100"]
    ].rename(columns={
        "test_ndcg10": "control_ndcg10", "test_mrr10": "control_mrr10", "test_recall100": "control_recall100"
    })
    x = runs.merge(control, on=["dataset", "seed"], how="left")
    x["delta_control_ndcg10"] = x.test_ndcg10 - x.control_ndcg10
    x["delta_control_mrr10"] = x.test_mrr10 - x.control_mrr10
    x["delta_control_recall100"] = x.test_recall100 - x.control_recall100
    x["win_control"] = (x.delta_control_ndcg10 > 1e-12).astype(float)
    x["noninferior_control"] = (x.delta_control_ndcg10 >= -0.005).astype(float)
    agg = x.groupby(
        ["config_id", "direction", "selector", "selector_title", "allocation", "allocation_title",
         "selector_hypothesis", "allocation_hypothesis"], as_index=False
    ).agg(
        runs=("test_ndcg10", "size"),
        mean_ndcg10=("test_ndcg10", "mean"),
        std_ndcg10=("test_ndcg10", "std"),
        mean_mrr10=("test_mrr10", "mean"),
        mean_recall100=("test_recall100", "mean"),
        mean_delta_base_ndcg10=("delta_base_ndcg10", "mean"),
        mean_delta_control_ndcg10=("delta_control_ndcg10", "mean"),
        win_rate_vs_control=("win_control", "mean"),
        noninferior_rate_vs_control=("noninferior_control", "mean"),
        mean_positive_rate=("feedback_positive_rate", "mean"),
        mean_query_positive_coverage=("queries_with_positive_rate", "mean"),
        mean_semantic_diversity=("semantic_diversity", "mean"),
        mean_unique_doc_fraction=("unique_doc_fraction", "mean"),
        mean_alpha=("alpha", "mean"),
    )
    # Conservative score: improvement plus robustness; avoid promoting a one-dataset spike.
    dataset_means = x.groupby(["config_id", "dataset"], as_index=False).agg(
        dataset_delta=("delta_control_ndcg10", "mean"), dataset_ndcg=("test_ndcg10", "mean")
    )
    min_delta = dataset_means.groupby("config_id").dataset_delta.min().rename("worst_dataset_delta")
    agg = agg.merge(min_delta, on="config_id", how="left")
    agg["screen_score"] = (
        agg.mean_delta_control_ndcg10
        + 0.35 * agg.worst_dataset_delta
        + 0.01 * (agg.win_rate_vs_control - 0.5)
    )
    agg = agg.sort_values(["screen_score", "mean_ndcg10"], ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", np.arange(1, len(agg) + 1))
    return agg


def fmt(x: float, signed: bool = False) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.4f}" if signed else f"{x:.4f}"


def write_report(
    out: Path, runs: pd.DataFrame, baselines: pd.DataFrame, agg: pd.DataFrame,
    seeds: Sequence[int], per_query_budget: int, elapsed: float,
) -> None:
    agg.to_csv(out / "SUMMARY_100_DIRECTIONS.csv", index=False)
    runs.to_csv(out / "ALL_RUNS.csv", index=False)
    baselines.to_csv(out / "BASELINES.csv", index=False)
    top5 = agg.head(5)
    top5.to_csv(out / "TOP5.csv", index=False)

    lines: list[str] = []
    lines += [
        "# 100-direction empirical screen for retrieval post-training",
        "",
        "## Bottom line",
        "",
        "This is a broad **sanity screen**, not a publication claim. It compares 100 practical feedback-selection configurations under a fixed label budget. The 100 configurations are 25 substantially different pair-selection primitives crossed with four broadly useful query-budget allocation policies.",
        "",
        f"- Datasets: {', '.join(sorted(runs.dataset.unique()))}",
        f"- Split seeds: {', '.join(map(str, seeds))}",
        f"- Configurations: {runs.config_id.nunique()}",
        f"- Dataset-seed-config runs: {len(runs)}",
        f"- Feedback budget: {per_query_budget} pairs/query on average; one known positive/query is fixed outside the feedback budget.",
        "- Student: one global logistic retrieval scorer over lexical, semantic, character, title, coverage, rank and disagreement signals; interpolation with the untouched retriever is tuned on validation queries.",
        "- Feedback oracle: relevance judgments are revealed **only after** a pair is selected. No selector sees the hidden label.",
        f"- Wall-clock runtime: {elapsed/60:.1f} minutes.",
        "",
        "## Evaluation of the submitted plan",
        "",
        "The submitted ANN-backend question is valid as a diagnostic, but too narrow as the final project thesis. Its own pilot already showed the key limitation: candidate identities changed sharply, yet coarse difficulty and teacher distributions were almost unchanged, and both post-trained branches degraded. A stronger, general direction is therefore **budgeted feedback allocation for retrieval post-training**: candidate generation is one source of variation, but the research object is the policy that decides which query–document pairs receive expensive labels.",
        "",
        "The screen below treats ANN choice as one component inside a larger data-selection problem and directly compares exploitation, uncertainty, disagreement, semantic coverage, experimental design, query allocation and policy portfolios.",
        "",
        "## Best 5 directions",
        "",
        "| Rank | Direction | Mean NDCG@10 | Δ vs standard top-candidate labeling | Worst-dataset Δ | Win rate | Positive yield | Semantic diversity |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in top5.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['direction']} | {fmt(r['mean_ndcg10'])} | "
            f"{fmt(r['mean_delta_control_ndcg10'], True)} | {fmt(r['worst_dataset_delta'], True)} | "
            f"{r['win_rate_vs_control']:.0%} | {r['mean_positive_rate']:.1%} | {r['mean_semantic_diversity']:.3f} |"
        )
    lines += [
        "",
        "### How to interpret the top five",
        "",
        "Promote a direction only when it is not merely high on the mean: it should be non-negative on the worse dataset, beat the standard top-candidate policy repeatedly, and retain a plausible mechanism such as finding hidden positives, covering distinct candidate regions, or allocating more budget to difficult queries.",
        "",
        "## Baselines and controls",
        "",
        "| Dataset | Untouched NDCG@10 | Untouched MRR@10 | Untouched R@100 | Cheap prior NDCG@10 | Candidate relevance recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, g in baselines.groupby("dataset"):
        lines.append(
            f"| {dataset} | {g.base_ndcg10.mean():.4f} | {g.base_mrr10.mean():.4f} | "
            f"{g.base_recall100.mean():.4f} | {g.prior_ndcg10.mean():.4f} | "
            f"{g.student_candidate_recall.mean():.4f} |"
        )
    lines += [
        "",
        "## All 100 directions",
        "",
        "| Rank | ID | Direction | Mean NDCG@10 | Δ top-candidate control | Worst-dataset Δ | Win rate | Positive yield | Diversity |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {int(r['rank'])} | `{r['config_id']}` | {r['direction']} | {fmt(r['mean_ndcg10'])} | "
            f"{fmt(r['mean_delta_control_ndcg10'], True)} | {fmt(r['worst_dataset_delta'], True)} | "
            f"{r['win_rate_vs_control']:.0%} | {r['mean_positive_rate']:.1%} | {r['mean_semantic_diversity']:.3f} |"
        )
    lines += [
        "",
        "## What has already been done versus what this adds",
        "",
        "Previously completed:",
        "- A 50k-passage MS MARCO pilot comparing exact, HNSW and IVF candidate pools.",
        "- A teacher-scored HNSW-versus-IVF micro-pilot.",
        "- Evidence that similar ANN recall can hide large pair-identity differences.",
        "- No evidence yet that ANN family choice improves post-training; all reported trained branches were below the untouched ANCE control.",
        "",
        "Added by this screen:",
        "- A much broader 100-configuration comparison centered on the general decision problem: where should a fixed feedback budget be spent?",
        "- Two datasets and repeated query splits rather than a single seed.",
        "- Untouched, standard top-candidate, random, uncertainty, diversity, disagreement and experimental-design controls under one pipeline.",
        "- A concrete shortlist whose hypotheses can be moved next to the full MS MARCO + cross-encoder setup.",
        "",
        "## Claim boundary",
        "",
        "- This is a lightweight reranking/post-training proxy, not dense-encoder fine-tuning.",
        "- Cranfield and SciFact are small; the results identify promising mechanisms, not final effectiveness claims.",
        "- Relevance judgments act as the feedback oracle. A real LLM/cross-encoder can supply graded, noisy feedback and may change the ranking.",
        "- The next decisive experiment is to take the best five selectors, keep the same fixed budget, and run them with ANCE (or another dense retriever), a real cross-encoder, full-corpus exact evaluation, and at least three training seeds.",
        "",
        "## Files",
        "",
        "- `SUMMARY_100_DIRECTIONS.csv`: one aggregate row per direction.",
        "- `ALL_RUNS.csv`: every dataset × seed × direction run.",
        "- `BASELINES.csv`: untouched and prior controls.",
        "- `TOP5.csv`: promoted shortlist.",
        "- `metadata.json`: exact run configuration.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "datasets": sorted(runs.dataset.unique().tolist()),
        "seeds": list(map(int, seeds)),
        "n_directions": int(runs.config_id.nunique()),
        "n_runs": int(len(runs)),
        "per_query_budget": int(per_query_budget),
        "selectors": [s.__dict__ for s in SELECTORS],
        "allocations": [a.__dict__ for a in ALLOCATIONS],
        "runtime_seconds": elapsed,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/100-direction-screen")
    p.add_argument("--cache", default=".cache/ire-screen")
    p.add_argument("--datasets", nargs="+", default=["cranfield", "scifact"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--budget", type=int, default=12)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    t0 = time.time()
    cache = Path(args.cache)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    datasets: list[IRDataset] = []
    if args.synthetic:
        datasets = [load_synthetic()]
    else:
        for name in args.datasets:
            if name == "cranfield":
                datasets.append(load_cranfield(cache))
            elif name == "scifact":
                datasets.append(load_scifact(cache))
            else:
                raise ValueError(f"Unknown dataset: {name}")

    run_parts, base_parts = [], []
    for ds in datasets:
        r, b = run_dataset(ds, args.seeds, args.budget, out)
        run_parts.append(r)
        base_parts.append(b)
    runs = pd.concat(run_parts, ignore_index=True)
    baselines = pd.concat(base_parts, ignore_index=True)
    agg = aggregate_results(runs)
    elapsed = time.time() - t0
    write_report(out, runs, baselines, agg, args.seeds, args.budget, elapsed)
    print(f"Wrote {out / 'REPORT.md'}; top five:")
    print(agg.head(5)[["rank", "config_id", "mean_ndcg10", "mean_delta_control_ndcg10", "win_rate_vs_control"]].to_string(index=False))


if __name__ == "__main__":
    main()
