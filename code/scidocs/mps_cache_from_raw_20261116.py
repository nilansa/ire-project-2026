#!/usr/bin/env python3
"""Convert already-computed original frozen Contriever vectors to sharded.py cache."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from benchmark import _model_files

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(2*1024*1024),b""):h.update(block)
    return h.hexdigest()

if len(sys.argv)!=4:
    raise SystemExit("usage: mps_cache_from_raw_20261116.py raw_npz retriever_dir cache_out")
npz, model, out=map(Path,sys.argv[1:])
with np.load(npz,allow_pickle=False) as a:
    raw_ids=[str(x) for x in a["doc_ids"].tolist()]
    qids=[str(x) for x in a["query_ids"].tolist()]
    d=np.array(a["doc_vectors"],dtype=np.float32)
    q=np.array(a["query_vectors"],dtype=np.float32)
assert len(raw_ids)==25657 and len(set(raw_ids))==len(raw_ids)
assert len(qids)==1000 and len(set(qids))==len(qids)
assert d.shape==(25657,768) and q.shape==(1000,768)
order=sorted(range(len(raw_ids)),key=lambda i:raw_ids[i])
sorted_ids=[raw_ids[i] for i in order]
cache={"doc_ids":sorted_ids,"query_ids":qids,
       "docs":torch.from_numpy(d[order].copy()),
       "queries":torch.from_numpy(q.copy()),
       "archive_sha256":"96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1",
       "retriever_files":_model_files(model),
       "script_sha256":sha(__file__),
       "original_npz_sha256":sha(npz),
       "encode_seconds":0.0}
assert not out.exists()
torch.save(cache,out)
test=torch.load(out,map_location="cpu",weights_only=True)
assert test["doc_ids"]==sorted_ids and test["query_ids"]==qids
assert test["docs"].shape==(25657,768)
print("CACHE_CONVERT_OK",out,"sha256",sha(out),"original_npz_sha256",sha(npz),flush=True)
