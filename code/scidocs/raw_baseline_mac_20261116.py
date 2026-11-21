#!/usr/bin/env python3
"""Source-only raw Contriever-msmarco SCIDOCS R@100 reproduction.

No teacher scores, no qrel use in candidate selection, no weight updates.
This standalone source-faithful baseline writes an immutable manifest and
cache for the same frozen encoder/SCIDOCS corpus used by ReFIT Table 2.
"""
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

EXPECTED_ZIP = "96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def load(archive):
    with zipfile.ZipFile(archive) as z:
        corpus = [json.loads(x) for x in z.read("scidocs/corpus.jsonl").splitlines()]
        queries = [json.loads(x) for x in z.read("scidocs/queries.jsonl").splitlines()]
    assert len(corpus) == 25657 and len(queries) == 1000
    return corpus, queries

def embeddings(texts, model, tok, device, batch=16):
    parts = []
    tic = time.monotonic()
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            t = tok(texts[i:i+batch], padding=True, truncation=True,
                    max_length=512, return_tensors="pt").to(device)
            o = model(**t)
            mask = t["attention_mask"].unsqueeze(-1)
            v = (o.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            parts.append(v.detach().cpu().numpy().astype(np.float32, copy=False))
            if (i // batch) % 80 == 0:
                print("ENCODED", min(i+batch,len(texts)), "/", len(texts),
                      "seconds", round(time.monotonic()-tic,1), flush=True)
    return np.concatenate(parts, axis=0)

def read_qrels(archive):
    with zipfile.ZipFile(archive) as z:
        f = io.StringIO(z.read("scidocs/qrels/test.tsv").decode("utf-8"))
    q = {}
    for row in csv.DictReader(f, delimiter="\t"):
        if int(row["score"]) > 0:
            q.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return q

def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: raw_baseline_mac_20261116.py <scidocs.zip> <retriever-dir> <output-dir> <device>")
    archive, ckpt, outdir, requested = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    assert archive.exists() and ckpt.exists()
    archive_sha = sha(archive)
    assert archive_sha == EXPECTED_ZIP, f"SCIDOCS source changed: {archive_sha}"
    outdir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    corpus, queries = load(archive)
    ids = [x["_id"] for x in corpus]
    qids = [x["_id"] for x in queries]
    assert len(ids) == len(set(ids)) and len(qids) == len(set(qids))
    texts = [(x.get("title", "") + " " + x.get("text", "")).strip() for x in corpus]
    device = torch.device(requested)
    tok = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
    model = AutoModel.from_pretrained(str(ckpt), local_files_only=True).to(device).eval()
    assert all(not p.requires_grad or p.grad is None for p in model.parameters())
    docs = embeddings(texts, model, tok, device)
    qs = embeddings([x["text"] for x in queries], model, tok, device)
    np.savez(outdir / "frozen_vectors.npz", doc_vectors=docs, query_vectors=qs,
             doc_ids=np.array(ids), query_ids=np.array(qids))
    assert docs.shape == (25657, 768) and qs.shape == (1000, 768)
    dt = torch.from_numpy(docs).to(device)
    qt = torch.from_numpy(qs).to(device)
    matches = {v:i for i,v in enumerate(ids)}
    ranks = {}
    with torch.inference_mode():
        for start in range(0,len(qids),32):
            scores = qt[start:start+32] @ dt.T
            for i,qid in enumerate(qids[start:start+32]):
                if qid in matches:
                    scores[i,matches[qid]]=-torch.inf
            top = torch.topk(scores,k=125,dim=1,largest=True,sorted=True)
            for i, qid in enumerate(qids[start:start+32]):
                ranks[qid] = [ids[int(j)] for j in top.indices[i].cpu().tolist()]
            print("RETRIEVED",min(start+32,len(qids)),len(qids),flush=True)
    (outdir / "raw_top125.json").write_text(json.dumps(ranks))
    qrels = read_qrels(archive)
    assert set(ranks) == set(qrels)
    r100 = {qid: len(set(ranks[qid][:100]) & pos) / len(pos)
            for qid, pos in qrels.items()}
    value = sum(r100.values())/len(r100)
    manifest = {
        "status":"ACTUAL_RETRIEVAL_AND_QREL_EVALUATION",
        "scope":"raw Contriever-msmarco SCIDOCS baseline only; no ReFIT or reranker executed",
        "run_timestamp_unix":started,
        "dataset_sha256":archive_sha,
        "model_weights_sha256":sha(ckpt/"pytorch_model.bin"),
        "script_sha256":sha(__file__),
        "model":"facebook/contriever-msmarco",
        "device":str(device),
        "dtype":"float32",
        "pooling":"attention_masked_mean",
        "max_length":512,
        "corpus":len(ids),"queries":len(qids),
        "recall_at_100":value,
        "elapsed_seconds":time.time()-started,
        "per_query_recall":r100,
        "predictions_sha256":sha(outdir/"raw_top125.json"),
        "vectors_sha256":sha(outdir/"frozen_vectors.npz"),
    }
    (outdir/"raw_result.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("RAW_SCIDOCS_RESULT R@100",round(value,8), "time",round(manifest["elapsed_seconds"],2),
          "path",outdir,flush=True)

if __name__=="__main__":
    main()
