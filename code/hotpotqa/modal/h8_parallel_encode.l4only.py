"""Ceous H8 — isolated rsai-r GPU-parallel corpus encoder.
Task: ceous-disco-ttt-20261116-0820. All input data is public MTEB text, no qrels.
Re-encodes all corpus shards consistently on CUDA (preserves Mac shards as fallback).
All GPU work is L4, <=4 concurrent containers, <=1000 s per shard.
Use only with the explicit rsai-r profile and independent billing preflight.
"""
import os, json, time, hashlib
import modal

app = modal.App("h8-qwen-parallel-rsai-r-20261116")
vol = modal.Volume.from_name("h8-disco-ttt-rsai-r-20261116", create_if_missing=False)
img = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.7.1", "transformers==4.53.2", "huggingface_hub==0.33.4",
    "numpy<2.3", "pyarrow==20.0.0")
MODEL = "Qwen/Qwen3-Embedding-0.6B"
REV = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
CHUNK = 8192
QP = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

@app.function(image=img, cpu=2, memory=4096, timeout=900, volumes={"/vol": vol})
def prefetch():
    from huggingface_hub import snapshot_download
    t = time.time()
    snapshot_download(repo_id=MODEL, revision=REV, token=False,
                      local_dir="/vol/models/embed",
                      allow_patterns=["*.json", "*.safetensors", "*.txt",
                                      "*.jinja", "vocab.*", "merges.txt"])
    import glob
    weights = glob.glob("/vol/models/embed/*.safetensors")
    assert weights and all(os.stat(p).st_size > 0 for p in weights), "missing model weights"
    vol.commit()
    return dict(ready=True, safetensors=[(os.path.basename(p),os.stat(p).st_size) for p in weights],
                elapsed_s=round(time.time()-t,2))

@app.cls(image=img, gpu="L4", cpu=3, memory=8192, timeout=1000,
         max_containers=4, volumes={"/vol": vol})
class Encoder:
    @modal.enter()
    def setup(self):
        import torch, numpy as np, pyarrow.parquet as pq
        from transformers import AutoTokenizer, AutoModel
        os.environ["TOKENIZERS_PARALLELISM"]="false"
        self.torch=torch
        self.np=np
        self.tok=AutoTokenizer.from_pretrained("/vol/models/embed",padding_side="left")
        self.model=AutoModel.from_pretrained(
            "/vol/models/embed",torch_dtype=torch.float16).to("cuda").eval()
        tab=pq.read_table("/vol/data/corpus.parquet",columns=["_id","title","text"]).to_pydict()
        self.ids=[str(x) for x in tab["_id"]]
        self.texts=[(f"{ti} {tx}".strip() if ti else tx)
                    for ti,tx in zip(tab["title"],tab["text"])]
        self.order=np.argsort([len(t) for t in self.texts],kind="stable")
        self.ndocs=len(self.ids)
        assert self.ndocs==225621, f"corpus count changed: {self.ndocs}"
    def _encode(self, texts, bs=48):
        np=self.np
        torch=self.torch
        embeddings=np.zeros((len(texts),self.model.config.hidden_size),dtype=np.float32)
        for s in range(0,len(texts),bs):
            batch=self.tok(texts[s:s+bs],padding=True,truncation=True,
                           max_length=512,return_tensors="pt").to("cuda")
            with torch.inference_mode():
                h=self.model(**batch).last_hidden_state[:,-1].float()
                h=torch.nn.functional.normalize(h,dim=-1)
            embeddings[s:s+bs]=h.cpu().numpy()
        assert np.isfinite(embeddings).all()
        return embeddings
    @modal.method()
    def shard(self,k:int):
        import numpy as np
        t=time.monotonic()
        assert 0<=k<28
        idx=self.order[k*CHUNK:min((k+1)*CHUNK,self.ndocs)]
        E=self._encode([self.texts[int(i)] for i in idx],bs=48)
        d="/vol/out/shards"
        os.makedirs(d,exist_ok=True)
        dest=f"{d}/shard_{k:03d}.npy"
        temp=dest+".temporary.npy"
        np.save(temp,E);os.replace(temp,dest)
        sha=hashlib.sha256(open(dest,"rb").read()).hexdigest()
        vol.commit()
        result=dict(shard=k,rows=len(idx),sha256=sha,
                    elapsed_s=round(time.monotonic()-t,1))
        print("SHARD_DONE",json.dumps(result),flush=True)
        return result
    @modal.method()
    def queries(self):
        import pyarrow.parquet as pq
        import numpy as np
        t=time.monotonic()
        q=pq.read_table("/vol/data/queries.parquet",columns=["_id","text"]).to_pydict()
        assert len(q["_id"])==1000
        Q=self._encode([QP+x for x in q["text"]],bs=32)
        os.makedirs("/vol/out",exist_ok=True)
        np.save("/vol/out/queries_qwen3emb06b_f32.npy",Q)
        json.dump([str(x) for x in q["_id"]],open("/vol/out/query_ids.json","w"))
        vol.commit()
        return dict(query_count=len(Q),elapsed_s=round(time.monotonic()-t,1))

@app.function(image=img,cpu=4,memory=8192,timeout=600,volumes={"/vol":vol})
def assemble():
    import numpy as np,pyarrow.parquet as pq
    t=time.monotonic()
    d=pq.read_table("/vol/data/corpus.parquet",columns=["_id","title","text"]).to_pydict()
    texts=[(f"{a} {b}".strip() if a else b) for a,b in zip(d["title"],d["text"])]
    order=np.argsort([len(t) for t in texts],kind="stable")
    assert len(order)==225621
    D=np.empty((len(order),1024),dtype=np.float32)
    hashes={}
    for k in range(28):
        f=f"/vol/out/shards/shard_{k:03d}.npy"
        assert os.path.isfile(f), f"missing shard {k}"
        e=np.load(f)
        ix=order[k*CHUNK:(k+1)*CHUNK]
        assert e.shape==(len(ix),1024), (k,e.shape,len(ix))
        D[ix]=e
        hashes[str(k)]=hashlib.sha256(open(f,"rb").read()).hexdigest()
    assert np.isfinite(D).all()
    norm=np.linalg.norm(D,axis=1)
    assert norm.min()>0.99 and norm.max()<1.01, (norm.min(),norm.max())
    assert os.path.isfile("/vol/out/queries_qwen3emb06b_f32.npy")
    q=np.load("/vol/out/queries_qwen3emb06b_f32.npy")
    assert q.shape==(1000,1024) and np.isfinite(q).all()
    os.makedirs("/vol/out",exist_ok=True)
    np.save("/vol/out/corpus_qwen3emb06b_f32.npy",D)
    json.dump([str(i) for i in d["_id"]],open("/vol/out/corpus_ids.json","w"))
    files=["corpus_qwen3emb06b_f32.npy","queries_qwen3emb06b_f32.npy",
           "corpus_ids.json","query_ids.json"]
    manifest={"task_id":"ceous-disco-ttt-20261116-0820","backend":"Modal rsai-r CUDA L4",
              "model":MODEL,"revision":REV,"source_docs":len(D),"source_queries":len(q),
              "corpus_shape":list(D.shape),"query_shape":list(q.shape),
              "norm_min":float(norm.min()),"norm_max":float(norm.max()),
              "per_shard_sha256":hashes,
              "sha256":{n:hashlib.sha256(open("/vol/out/"+n,"rb").read()).hexdigest() for n in files},
              "assemble_s":round(time.monotonic()-t,2)}
    json.dump(manifest,open("/vol/out/ENCODE_MANIFEST.json","w"),indent=2)
    vol.commit()
    return {k:manifest[k] for k in ["source_docs","source_queries","corpus_shape","query_shape","assemble_s","norm_min","norm_max"]}

@app.local_entrypoint()
def main(stage:str="prefetch"):
    if stage=="prefetch":
        print("H8_PREFETCH",json.dumps(prefetch.remote()),flush=True)
    elif stage=="encode":
        results=[]
        w=Encoder()
        for r in w.shard.map(range(28),order_outputs=False):
            results.append(r)
            print("H8_PROGRESS",len(results),"/28",json.dumps(r),flush=True)
        assert len(results)==28
        print("H8_QUERIES",json.dumps(w.queries.remote()),flush=True)
        print("H8_ASSEMBLE",json.dumps(assemble.remote()),flush=True)
    elif stage=="assemble":
        print("H8_ASSEMBLE",json.dumps(assemble.remote()),flush=True)
    else:
        raise ValueError("unknown stage")
