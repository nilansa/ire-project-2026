from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from common import COLBERT_PATH, DATA, MODEL_NAME, MODEL_REVISION, ROOT, read_jsonl, sha256_file, write_jsonl_new

SYSTEM_PROMPT = (
    "You decompose information-retrieval questions. Return at most three concise, "
    "independently searchable sub-questions that together preserve the original "
    "entities and constraints. Do not answer or add facts. If decomposition adds "
    "no value, use an empty sub_queries array. Output exactly one JSON object "
    "of the form {\"sub_queries\":[\"...\"]}; when empty use "
    "{\"sub_queries\":[]}. Do not output a bare array or markdown."
)
STOP = set("a an the and or but if then than of in on at to for from by with without is are was were be been being do does did how what who whom when where why which that this these those it its as into about over under after before between during can could should would may might will shall not no yes we they he she them their our your my i you me us has have had having".split())
OLLAMA_MODEL = "qwen3.5:4b-mlx"
OLLAMA_DIGEST = "61aa3858e9d3022e8fca725550089addd9289c4446f33bd09dfd12f95f2a6792"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.docs = docs
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, docs)) / max(1, len(docs))
        df = Counter()
        for doc in docs:
            df.update(set(doc))
        n = len(docs)
        idf = {term: np.log(n - count + 0.5) - np.log(count + 0.5) for term, count in df.items()}
        avg_idf = sum(idf.values()) / max(1, len(idf))
        self.idf = {term: (epsilon * avg_idf if value < 0 else value) for term, value in idf.items()}
        self.lengths = np.asarray([len(x) for x in docs], dtype=np.float32)
        self.counts = [Counter(x) for x in docs]

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        for term in tokenize(query):
            weight = self.idf.get(term)
            if weight is None:
                continue
            tf = np.fromiter((d.get(term, 0) for d in self.counts), dtype=np.float32, count=len(self.docs))
            denom = tf + self.k1 * (1 - self.b + self.b * self.lengths / max(self.avgdl, 1e-12))
            out += weight * tf * (self.k1 + 1) / np.maximum(denom, 1e-12)
        return out


def top_ids(scores: np.ndarray, ids: list[str], n: int) -> list[str]:
    order = np.argsort(-scores, kind="stable")[:n]
    return [ids[int(i)] for i in order]


def rrf(lists: list[list[str]], k: int = 60) -> list[str]:
    sums: dict[str, float] = {}
    first: dict[str, int] = {}
    serial = 0
    for ranking in lists:
        for rank, docid in enumerate(ranking, 1):
            sums[docid] = sums.get(docid, 0.0) + 1.0 / (k + rank)
            if docid not in first:
                first[docid] = serial
                serial += 1
    return sorted(sums, key=lambda d: (-sums[d], first[d]))


def stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def load_dense_model():
    os.environ["HF_HOME"] = str(ROOT / "hf_cache")
    from sentence_transformers import SentenceTransformer
    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    return SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device=device), device


def load_colbert():
    import torch
    import colbert.parameters as parameters
    import colbert.modeling.base_colbert as base
    import colbert.modeling.tokenization.query_tokenization as qt
    import colbert.modeling.tokenization.doc_tokenization as dt
    from colbert.infra import ColBERTConfig
    from colbert.modeling.colbert import ColBERT
    from colbert.modeling.checkpoint import Checkpoint

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    parameters.DEVICE = device
    base.DEVICE = device
    qt.DEVICE = device
    dt.DEVICE = device
    ColBERT.loaded_extensions = True
    config = ColBERTConfig(
        checkpoint=str(COLBERT_PATH), query_maxlen=32, doc_maxlen=180,
        gpus=0, mask_punctuation=True, attend_to_mask_tokens=False,
    )
    model = Checkpoint(str(COLBERT_PATH), colbert_config=config, verbose=0)
    model.amp_manager.activated = False
    return model, device


def corpus_rows(dataset: str):
    rows = list(read_jsonl(DATA / dataset / "corpus.jsonl"))
    ids = [str(x["doc_id"]) for x in rows]
    texts = [(str(x.get("title", "")) + " " + str(x.get("text", ""))).strip() for x in rows]
    return ids, texts


def load_dense_corpus(model, dataset: str, texts: list[str]) -> np.ndarray:
    cache = ROOT / "cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"{dataset}_tasb.npy"
    meta = cache / f"{dataset}_tasb.json"
    if path.exists() or meta.exists():
        if not path.exists() or not meta.exists():
            raise RuntimeError(f"incomplete dense cache for {dataset}")
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info.get("revision") != MODEL_REVISION or info.get("corpus_sha256") != sha256_file(DATA / dataset / "corpus.jsonl"):
            raise RuntimeError(f"dense cache identity mismatch for {dataset}")
        return np.load(path, mmap_mode="r")
    vectors = model.encode(texts, batch_size=48, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=False)
    with path.open("xb") as f:
        np.save(f, np.asarray(vectors, dtype=np.float32))
    with meta.open("x", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "revision": MODEL_REVISION, "corpus_sha256": sha256_file(DATA / dataset / "corpus.jsonl"), "shape": list(vectors.shape)}, f, indent=2)
        f.write("\n")
    return np.load(path, mmap_mode="r")


class Ollama:
    def __init__(self):
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self.opener.open("http://127.0.0.1:11434/api/tags", timeout=10) as response:
            tags = json.loads(response.read())
        match = [m for m in tags.get("models", []) if m.get("name") == OLLAMA_MODEL]
        if len(match) != 1 or match[0].get("digest") != OLLAMA_DIGEST:
            raise RuntimeError("the required local Ollama model tag or digest is unavailable")
        with self.opener.open("http://127.0.0.1:11434/api/version", timeout=10) as response:
            version = json.loads(response.read()).get("version")
        self.version = version
        self.digest = OLLAMA_DIGEST
        self.calls = 0
        self.tokens = 0
        self.history = []

    def decompose(self, query: str) -> dict:
        body = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
            "stream": False, "think": False,
            "options": {"temperature": 0, "seed": 0},
            "format": {"type": "object", "properties": {"sub_queries": {"type": "array", "items": {"type": "string"}}}, "required": ["sub_queries"], "additionalProperties": False},
        }).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=body, headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=900) as response:
            result = json.loads(response.read())
        raw = result.get("message", {}).get("content", "")
        count = result.get("eval_count")
        self.calls += 1
        self.tokens += int(count or 0)
        self.history.append({"raw": raw, "tokens": count, "total_duration_ns": result.get("total_duration")})
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"decomposition must be a JSON object: {raw[:300]}")
        subs = parsed.get("sub_queries")
        if not isinstance(subs, list) or len(subs) > 3 or any(not isinstance(x, str) for x in subs):
            raise ValueError(f"decomposition shape invalid: {raw[:300]}")
        return {"raw": raw, "sub_queries": subs, "tokens": count, "total_duration_ns": result.get("total_duration")}


def encode_doc_vectors(model, docids: list[str], texts_by_id: dict[str, str]) -> dict[str, object]:
    import torch
    if not docids:
        return {}
    raw, lengths = model.docFromText(
        [texts_by_id[d] for d in docids], bsize=16, keep_dims="flatten",
        to_cpu=False, showprogress=False,
    )
    out = {}
    start = 0
    for docid, length in zip(docids, lengths):
        out[docid] = raw[start:start + int(length)].to(torch.float16)
        start += int(length)
    if start != raw.shape[0]:
        raise AssertionError("ColBERT token lengths do not match flattened vectors")
    return out


def padded_vectors(docids: list[str], vectors: dict[str, object], device):
    import torch
    if not docids:
        return torch.empty((0, 1, 128), dtype=torch.float32, device=device), torch.empty((0, 1), dtype=torch.bool, device=device)
    maxlen = max(int(vectors[d].shape[0]) for d in docids)
    docs = torch.zeros((len(docids), maxlen, 128), dtype=torch.float32)
    mask = torch.zeros((len(docids), maxlen), dtype=torch.bool)
    for i, docid in enumerate(docids):
        vec = vectors[docid].float()
        docs[i, :len(vec)] = vec
        mask[i, :len(vec)] = True
    return docs.to(device), mask.to(device)


def similarities(qvec, docids: list[str], vectors: dict[str, object], device):
    import torch
    docs, mask = padded_vectors(docids, vectors, device)
    if not docids:
        return torch.empty((0, qvec.shape[1]), device=device), torch.empty((0, qvec.shape[1], 1), device=device)
    sim = torch.einsum("qd,nld->nql", qvec.squeeze(0).float(), docs)
    sim = sim.masked_fill(~mask[:, None, :], -1e4)
    return sim.max(dim=-1).values, sim


def greedy(docids: list[str], doc_best, limit: int = 20, seed_ids: list[str] | None = None) -> list[str]:
    import torch
    if not docids:
        return []
    positions = {d: i for i, d in enumerate(docids)}
    chosen = list(seed_ids or [])
    current = torch.zeros(doc_best.shape[1], dtype=doc_best.dtype, device=doc_best.device)
    for d in chosen:
        current = torch.maximum(current, doc_best[positions[d]])
    used = set(chosen)
    while len(chosen) < min(limit, len(docids)):
        gain = torch.clamp(doc_best - current.unsqueeze(0), min=0).sum(dim=1)
        for d in used:
            gain[positions[d]] = -1e9
        idx = int(torch.argmax(gain).item())
        d = docids[idx]
        chosen.append(d)
        used.add(d)
        current = torch.maximum(current, doc_best[idx])
    return chosen


def residual_text(query: str, qvec, sim, colbert, selected: list[str], docids: list[str], device) -> str:
    import torch
    if not selected:
        return ""
    qtokens = colbert.raw_tokenizer(query, add_special_tokens=False, return_offsets_mapping=True)
    offsets = qtokens["offset_mapping"][:29]
    selected_rows = [docids.index(d) for d in selected]
    coverage = sim[selected_rows].max(dim=0).values.max(dim=-1).values
    token_cov = coverage[2:2 + len(offsets)]
    valid = [(i, a, b) for i, (a, b) in enumerate(offsets) if b > a and query[a:b].strip()]
    if not valid:
        return ""
    median = torch.median(torch.stack([token_cov[i] for i, _, _ in valid])).item()
    words = []
    for match in re.finditer(r"\w+", query, flags=re.UNICODE):
        word = match.group(0)
        if word.lower() in STOP:
            continue
        idxs = [i for i, a, b in valid if a < match.end() and b > match.start()]
        if idxs and float(torch.stack([token_cov[i] for i in idxs]).mean().item()) < median:
            words.append(word)
    return " ".join(dict.fromkeys(words))


def random_control(qid: str, qvec, docids: list[str], vectors: dict[str, object], device) -> tuple[list[str], list[str]]:
    import torch
    digest = hashlib.sha256(("p5-random:" + qid).encode()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rq = torch.randn(qvec.shape, generator=generator)
    rq = torch.nn.functional.normalize(rq, dim=-1).to(device)
    random_vecs = {}
    for d in docids:
        n = int(vectors[d].shape[0])
        x = torch.randn((n, 128), generator=generator)
        random_vecs[d] = torch.nn.functional.normalize(x, dim=-1)
    best, _ = similarities(rq, docids, random_vecs, device)
    random_greedy = greedy(docids, best, 20)
    permuted = list(docids)
    rng.shuffle(permuted)
    return random_greedy, permuted[:20]


def fuse_for_query(bm25: BM25, dense: np.ndarray, ids: list[str], model, query: str, topn: int) -> tuple[list[str], list[str], int]:
    br = top_ids(bm25.scores(query), ids, topn)
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)[0]
    dr = top_ids(dense @ q, ids, topn)
    return br, dr, 2


def run_one(dataset: str, query_rows: list[dict], ids: list[str], texts: list[str], dense: np.ndarray, bm25: BM25, dense_model, colbert, cdevice, llm: Ollama, output_decomps: list[dict], progress_prefix: str) -> list[dict]:
    texts_by_id = dict(zip(ids, texts))
    out = []
    for qi, row in enumerate(query_rows, 1):
        start_q = time.perf_counter()
        qid, query = str(row["qid"]), str(row["text"])
        t = time.perf_counter(); b1 = top_ids(bm25.scores(query), ids, 200); lat_a1 = time.perf_counter() - t
        t = time.perf_counter(); qdense = dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=False)[0]; d1 = top_ids(dense @ qdense, ids, 200); lat_a2 = time.perf_counter() - t
        t = time.perf_counter(); base = rrf([b1, d1])[:200]; lat_a3 = time.perf_counter() - t

        t = time.perf_counter()
        qvec = colbert.queryFromText([query])
        pool_vecs = encode_doc_vectors(colbert, base, texts_by_id)
        best, sim = similarities(qvec, base, pool_vecs, cdevice)
        a4 = [base[i] for i in np.argsort(-best.sum(dim=1).detach().cpu().numpy(), kind="stable")]
        lat_a4 = time.perf_counter() - t

        t = time.perf_counter(); a5 = greedy(base, best, 20); lat_a5 = time.perf_counter() - t

        t = time.perf_counter()
        seed3 = a5[:3]
        residual = residual_text(query, qvec, sim, colbert, seed3, base, cdevice)
        a6pool = list(base)
        extra6 = []
        if residual:
            rb, rd, _ = fuse_for_query(bm25, dense, ids, dense_model, residual, 100)
            for d in rrf([rb, rd])[:100]:
                if d not in set(a6pool):
                    a6pool.append(d); extra6.append(d)
        vec6 = encode_doc_vectors(colbert, extra6, texts_by_id)
        all6vec = {**pool_vecs, **vec6}
        best6, _ = similarities(qvec, a6pool, all6vec, cdevice)
        a6 = greedy(a6pool, best6, 20, seed_ids=seed3)
        lat_a6 = time.perf_counter() - t

        t = time.perf_counter()
        dec = llm.decompose(query)
        subs = dec["sub_queries"]
        output_decomps.append({"dataset": dataset, "qid": qid, "phase": progress_prefix, "prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), **dec})
        decomp_lists = []
        for sub in subs:
            sb, sd, _ = fuse_for_query(bm25, dense, ids, dense_model, sub, 200)
            decomp_lists.append(rrf([sb, sd])[:200])
        decomp_work_s = time.perf_counter() - t
        t_fuse = time.perf_counter()
        a7_lists = [rrf([b1, d1])[:200]] + decomp_lists
        a7 = rrf(a7_lists)[:200]
        lat_a7 = decomp_work_s + time.perf_counter() - t_fuse
        t_fuse = time.perf_counter()
        a7b = rrf(decomp_lists)[:200] if decomp_lists else []
        lat_a7b = decomp_work_s + time.perf_counter() - t_fuse

        t = time.perf_counter()
        a8pool = list(dict.fromkeys(d for r in a7_lists for d in r))
        extra8 = [d for d in a8pool if d not in all6vec]
        vec8 = encode_doc_vectors(colbert, extra8, texts_by_id)
        all8vec = {**all6vec, **vec8}
        best8, _ = similarities(qvec, a8pool, all8vec, cdevice)
        a8 = greedy(a8pool, best8, 20)
        lat_a8 = decomp_work_s + time.perf_counter() - t

        random_a5, random_order = random_control(qid, qvec, base, pool_vecs, cdevice)
        timings = {"A1": lat_a1, "A2": lat_a2, "A3": lat_a3, "A4": lat_a4, "A5": lat_a5, "A6": lat_a6, "A7": lat_a7, "A7b": lat_a7b, "A8": lat_a8}
        results = {
            "dataset": dataset, "qid": qid, "query": query,
            "rankings": {"A1": b1, "A2": d1, "A3": base, "A4": a4, "A5": a5, "A6": a6, "A7": a7, "A7b": a7b, "A8": a8},
            "base_pool": base, "a7_pool": a8pool,
            "random_a5": random_a5, "random_order": random_order,
            "latency_s": timings,
            "retrieval_calls": {"A1": 1, "A2": 1, "A3": 0, "A4": 0, "A5": 0, "A6": 2 if residual else 0, "A7": 2 * len(subs), "A7b": 2 * len(subs), "A8": 0},
            "doc_encodings": {"A1": 0, "A2": 0, "A3": 0, "A4": len(base), "A5": 0, "A6": len(extra6), "A7": 0, "A7b": 0, "A8": len(extra8)},
            "llm_calls": {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0, "A6": 0, "A7": 1, "A7b": 1, "A8": 1},
            "llm_tokens": {"A1": 0, "A2": 0, "A3": 0, "A4": 0, "A5": 0, "A6": 0, "A7": dec["tokens"], "A7b": dec["tokens"], "A8": dec["tokens"]},
            "residual_query": residual, "query_wall_s": time.perf_counter() - start_q,
        }
        out.append(results)
        print(json.dumps({"phase": progress_prefix, "dataset": dataset, "qid": qid, "done": qi, "total": len(query_rows), "wall_s": round(results["query_wall_s"], 3), "at": stamp()}), flush=True)
    return out


def execute(smoke: bool) -> None:
    import torch
    t0 = time.perf_counter()
    print(json.dumps({"phase": "retrieve_start", "pid": os.getpid(), "smoke": smoke, "at": stamp()}), flush=True)
    dense_model, dense_device = load_dense_model()
    colbert, cdevice = load_colbert()
    llm = Ollama()
    datasets = ["scifact"] if smoke else ["scifact", "nfcorpus", "musique"]
    output_rows = []
    output_decomps = []
    try:
        for dataset in datasets:
            ids, texts = corpus_rows(dataset)
            dense = load_dense_corpus(dense_model, dataset, texts)
            bm25 = BM25([tokenize(t) for t in texts])
            query_path = DATA / "smoke_scifact_train.jsonl" if smoke else DATA / dataset / "queries.jsonl"
            queries = list(read_jsonl(query_path))
            if smoke:
                queries = queries[:5]
                prefix = "smoke"
            else:
                prefix = "main"
            output_rows.extend(run_one(dataset, queries, ids, texts, dense, bm25, dense_model, colbert, cdevice, llm, output_decomps, prefix))
    except Exception:
        if llm.history:
            path = ROOT / ("smoke_failed_llm_outputs.jsonl" if smoke else "failed_llm_outputs.jsonl")
            write_jsonl_new(path, llm.history)
        raise
    outdir = ROOT / "smoke" if smoke else ROOT / "retrieval"
    outdir.mkdir(exist_ok=False)
    write_jsonl_new(outdir / "retrieval.jsonl", output_rows)
    write_jsonl_new(outdir / "decompositions.jsonl", output_decomps)
    report = {"pid": os.getpid(), "smoke": smoke, "datasets": datasets, "rows": len(output_rows), "llm_calls": llm.calls, "llm_output_tokens": llm.tokens, "ollama_model": OLLAMA_MODEL, "ollama_digest": llm.digest, "ollama_version": llm.version, "dense_device": dense_device, "colbert_device": str(cdevice), "model_revision": MODEL_REVISION, "wall_s": time.perf_counter() - t0, "completed_utc": stamp()}
    with (outdir / "run_summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2); f.write("\n")
    print(json.dumps({"phase": "retrieve_done", **report}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"--smoke", "--main"}:
        raise SystemExit("usage: retrieve.py --smoke|--main")
    execute(sys.argv[1] == "--smoke")

