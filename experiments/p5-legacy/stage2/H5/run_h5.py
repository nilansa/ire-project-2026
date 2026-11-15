"""Local HotpotQA tune preparation and dense MiniLM comparator for H5."""

import json
import hashlib
import os
import re
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "3"
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def read_jsonl(path, predicate=None):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if predicate is not None and not predicate(line):
                continue
            rows.append(json.loads(line))
    return rows


def _ready_hash_entries(value, prefix=""):
    entries = []
    if isinstance(value, list):
        for item in value:
            entries.extend(_ready_hash_entries(item, prefix))
    elif isinstance(value, dict):
        path = value.get("path") or value.get("file") or value.get("name")
        digest = value.get("sha256") or value.get("sha256sum")
        if path and digest:
            entries.append((str(path), str(digest)))
        elif digest and prefix:
            entries.append((prefix, str(digest)))
        for key, item in value.items():
            if key in {"path", "file", "name", "sha256", "sha256sum"}:
                continue
            if isinstance(item, str) and re.fullmatch(r"[0-9a-fA-F]{64}", item):
                entries.append((os.path.join(prefix, key), item.lower()))
            elif isinstance(item, (dict, list)):
                next_prefix = prefix if key in {"files", "hashes", "checksums", "sha256"} else os.path.join(prefix, key)
                entries.extend(_ready_hash_entries(item, next_prefix))
    return entries


def verify_hotpotqa_hashes(ready_path, data_root):
    ready = json.loads(Path(ready_path).read_text(encoding="utf-8"))
    entries = _ready_hash_entries(ready)
    required = ("corpus.jsonl", "questions.jsonl", "emb_minilm.f16.npy")
    root = Path(data_root)
    verified = {}
    for name in required:
        matches = [
            expected for path, expected in entries
            if path.replace("\\", "/").removeprefix("data/").endswith(f"hotpotqa/{name}")
        ]
        if len(matches) != 1:
            raise ValueError(f"READY.json must contain one SHA-256 for hotpotqa/{name}; found {len(matches)}")
        actual = hashlib.sha256((root / "hotpotqa" / name).read_bytes()).hexdigest()
        if actual != matches[0]:
            raise ValueError(f"READY.json SHA-256 mismatch for hotpotqa/{name}")
        verified[name] = actual
    return verified


def load_hotpotqa_tune(data_root, n_questions=200):
    root = Path(data_root)
    ready = root / "READY.json"
    if not ready.is_file():
        raise FileNotFoundError(f"data readiness gate missing: {ready}")
    ready_sha256 = hashlib.sha256(ready.read_bytes()).hexdigest()
    dataset = root / "hotpotqa"
    corpus_path = dataset / "corpus.jsonl"
    questions_path = dataset / "questions.jsonl"
    embeddings_path = dataset / "emb_minilm.f16.npy"
    missing = [str(p) for p in (corpus_path, questions_path, embeddings_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"READY exists but required inputs are missing: {missing}")
    verified_hashes = verify_hotpotqa_hashes(ready, root)

    corpus = read_jsonl(corpus_path)
    pids = [row["pid"] for row in corpus]
    if len(pids) != len(set(pids)):
        raise ValueError("corpus pids must be unique")
    tune_marker = re.compile(br'"split"\s*:\s*"tune"')
    tune = read_jsonl(questions_path, lambda line: bool(tune_marker.search(line.encode())))
    if any(row.get("split") != "tune" for row in tune):
        raise ValueError("question split filter returned a non-tune row")
    tune.sort(key=lambda row: str(row["qid"]))
    questions = tune[:n_questions]
    if len({row["qid"] for row in questions}) != len(questions):
        raise ValueError("tune qids must be unique")
    known = set(pids)
    for row in questions:
        if not row.get("gold_pids") or not set(row["gold_pids"]).issubset(known):
            raise ValueError(f"invalid gold_pids for tune qid {row['qid']}")

    import numpy as np

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] != len(corpus):
        raise ValueError("MiniLM embeddings must be 2-D and row-aligned with corpus.jsonl")
    return {
        "corpus": corpus,
        "questions": questions,
        "embeddings": embeddings,
        "ready_sha256": ready_sha256,
        "verified_hashes": verified_hashes,
    }


def select_corpus(corpus, questions, max_docs=None):
    if max_docs is None or max_docs >= len(corpus):
        return corpus
    gold = {pid for question in questions for pid in question["gold_pids"]}
    if len(gold) > max_docs:
        raise ValueError("max_docs is smaller than the selected queries' gold-passage set")
    keep = set(gold)
    for row in corpus:
        if len(keep) >= max_docs:
            break
        keep.add(row["pid"])
    return [row for row in corpus if row["pid"] in keep]


def write_beir_view(destination, corpus, questions):
    """Create a new BEIR-format view; refuse to overwrite any prior run data."""
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing BEIR data: {target}")
    (target / "qrels").mkdir(parents=True)
    with open(target / "corpus.jsonl", "w", encoding="utf-8") as stream:
        for row in corpus:
            stream.write(json.dumps({"_id": row["pid"], "title": row["title"], "text": row["text"]}) + "\n")
    with open(target / "queries.jsonl", "w", encoding="utf-8") as stream:
        for row in questions:
            stream.write(json.dumps({"_id": row["qid"], "text": row["question"]}) + "\n")
    with open(target / "qrels" / "test.tsv", "w", encoding="utf-8") as stream:
        stream.write("query-id\tcorpus-id\tscore\n")
        for row in questions:
            for pid in row["gold_pids"]:
                stream.write(f"{row['qid']}\t{pid}\t1\n")


def dense_minilm(data, corpus, k=10):
    import faiss
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(3)
    faiss.omp_set_num_threads(3)
    matrix = data["embeddings"]
    pid_to_row = {row["pid"]: i for i, row in enumerate(data["corpus"])}
    rows = np.asarray([pid_to_row[row["pid"]] for row in corpus], dtype=np.int64)
    index = faiss.IndexFlatIP(matrix.shape[1])
    for start in range(0, len(rows), 8192):
        batch = np.asarray(matrix[rows[start : start + 8192]], dtype=np.float32, order="C")
        index.add(batch)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    rankings, latencies = {}, {}
    for question in data["questions"]:
        start = time.perf_counter()
        vector = model.encode(
            question["question"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        _, found = index.search(vector.reshape(1, -1), k)
        latencies[question["qid"]] = time.perf_counter() - start
        rankings[question["qid"]] = [corpus[i]["pid"] for i in found[0] if i >= 0]
    return rankings, latencies, index.ntotal


def main():
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../data")
    try:
        data = load_hotpotqa_tune(data_root)
    except FileNotFoundError as error:
        print(json.dumps({"status": "BLOCKED_DATA_NOT_READY", "reason": str(error)}))
        return 2
    print(json.dumps({
        "status": "READY",
        "dataset": "hotpotqa",
        "split": "tune",
        "n_questions": len(data["questions"]),
        "n_corpus": len(data["corpus"]),
        "ready_sha256": data["ready_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
