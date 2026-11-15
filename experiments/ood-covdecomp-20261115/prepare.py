from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from itertools import chain
from pathlib import Path

from common import DATA, MUSIQUE_CORPUS, MUSIQUE_DEV, RAW, SALT, STAGE1_SAMPLE, read_jsonl, salted, sha256_file, write_json_new, write_jsonl_new

BEIR = {
    "scifact": ("5f7d1de60b170fc8027bb7898e2efca1", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"),
    "nfcorpus": ("a89dba18a62ef92f7d323ec890a0d38d", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"),
}


def member(zf: zipfile.ZipFile, suffix: str) -> str:
    found = [x for x in zf.namelist() if x.endswith(suffix)]
    if len(found) != 1:
        raise RuntimeError(f"expected one {suffix}, found {len(found)}")
    return found[0]


def zip_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, expected_md5: str, url: str) -> Path:
    path = RAW / f"{name}.zip"
    RAW.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "ire-p5-frozen-pilot/1.0"})
        with urllib.request.urlopen(req, timeout=90) as response, path.open("xb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    actual = zip_md5(path)
    if actual != expected_md5:
        raise RuntimeError(f"{name} zip MD5 mismatch: {actual}")
    return path


def read_query_map(zf: zipfile.ZipFile, name: str) -> dict[str, str]:
    path = member(zf, f"{name}/queries.jsonl")
    out = {}
    with zf.open(path) as f:
        for raw in f:
            rec = json.loads(raw)
            out[str(rec["_id"])] = rec["text"]
    return out


def qid_column(zf: zipfile.ZipFile, name: str, split: str) -> list[str]:
    path = member(zf, f"{name}/qrels/{split}.tsv")
    out = []
    with zf.open(path) as f:
        first = next(f, None)
        lines = f if first is not None and first.lower().startswith(b"query-id\t") else chain([first], f) if first is not None else iter(())
        for raw in lines:
            cols = raw.decode("utf-8").rstrip("\n").split("\t")
            if cols:
                out.append(cols[0])
    return sorted(set(out), key=lambda q: (salted(q), q))


def copy_corpus(zf: zipfile.ZipFile, name: str, dest: Path) -> int:
    path = member(zf, f"{name}/corpus.jsonl")
    n = 0
    with zf.open(path) as src, dest.open("x", encoding="utf-8") as out:
        for raw in src:
            rec = json.loads(raw)
            out.write(json.dumps({"doc_id": str(rec["_id"]), "title": rec.get("title", ""), "text": rec.get("text", "")}, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def prep_beir(name: str, expected_md5: str, url: str) -> dict:
    root = DATA / name
    root.mkdir(parents=True, exist_ok=False)
    archive = download(name, expected_md5, url)
    with zipfile.ZipFile(archive) as zf:
        queries = read_query_map(zf, name)
        test_ids = qid_column(zf, name, "test")
        selected = test_ids[:100]
        if len(selected) != 100:
            raise RuntimeError(f"{name}: expected >=100 test ids, got {len(selected)}")
        train_rows = []
        if name == "scifact":
            train_ids = qid_column(zf, name, "train")
            train_rows = [{"qid": q, "text": queries[q]} for q in train_ids[:5] if q in queries]
        n_docs = copy_corpus(zf, name, root / "corpus.jsonl")
    write_jsonl_new(root / "queries.jsonl", ({"qid": q, "text": queries[q]} for q in selected))
    if name == "scifact":
        if len(train_rows) != 5:
            raise RuntimeError(f"SciFact train smoke set must contain 5 queries, got {len(train_rows)}")
        write_jsonl_new(DATA / "smoke_scifact_train.jsonl", train_rows)
    return {
        "zip": str(archive), "zip_md5": expected_md5, "zip_sha256": sha256_file(archive),
        "corpus_path": str(root / "corpus.jsonl"), "corpus_docs": n_docs,
        "corpus_sha256": sha256_file(root / "corpus.jsonl"),
        "queries_path": str(root / "queries.jsonl"), "query_ids": selected,
        "query_sha256": sha256_file(root / "queries.jsonl"),
    }


def sample_musique() -> dict:
    with STAGE1_SAMPLE.open(encoding="utf-8") as f:
        stage1_ids = {str(x["id"]) for x in json.load(f)}
    candidates = []
    with MUSIQUE_DEV.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            qid = str(rec.get("id", ""))
            if qid.startswith("2hop__") and qid not in stage1_ids:
                candidates.append((salted(qid), qid, str(rec["question"])))
    candidates.sort()
    selected = candidates[:100]
    if len(selected) != 100:
        raise RuntimeError(f"MuSiQue: only {len(selected)} candidate ids after Stage-1 exclusion")
    root = DATA / "musique"
    root.mkdir(parents=True, exist_ok=False)
    with MUSIQUE_CORPUS.open(encoding="utf-8") as src, (root / "corpus.jsonl").open("x", encoding="utf-8") as out:
        n_docs = 0
        for line in src:
            rec = json.loads(line)
            out.write(json.dumps({"doc_id": str(rec["doc_id"]), "title": rec.get("title", ""), "text": rec.get("text", "")}, ensure_ascii=False, separators=(",", ":")) + "\n")
            n_docs += 1
    write_jsonl_new(root / "queries.jsonl", ({"qid": q, "text": question} for _, q, question in selected))
    return {
        "corpus_path": str(root / "corpus.jsonl"), "corpus_docs": n_docs,
        "corpus_sha256": sha256_file(root / "corpus.jsonl"),
        "queries_path": str(root / "queries.jsonl"), "query_ids": [q for _, q, _ in selected],
        "query_sha256": sha256_file(root / "queries.jsonl"),
        "raw_dev_path": str(MUSIQUE_DEV),
        "stage1_id_source": str(STAGE1_SAMPLE), "stage1_id_count": len(stage1_ids),
        "candidate_2hop_ids_after_exclusion": len(candidates),
    }


def main() -> None:
    if DATA.exists():
        raise FileExistsError(f"refusing to reuse or overwrite {DATA}")
    DATA.mkdir()
    report = {"started_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "pid": __import__("os").getpid(), "beir": {}, "musique": {}}
    for name, (md5, url) in BEIR.items():
        report["beir"][name] = prep_beir(name, md5, url)
    report["musique"] = sample_musique()
    report["smoke_queries_sha256"] = sha256_file(DATA / "smoke_scifact_train.jsonl")
    report["completed_utc"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    write_json_new(DATA / "prepare_manifest.json", report)
    print(json.dumps({"phase": "prepare", "pid": report["pid"], "data": str(DATA), "completed_utc": report["completed_utc"]}))


if __name__ == "__main__":
    main()
