"""Ceous H8 full-corpus qrel-blind retrieval on authenticated Modal rsai-r.
Uses pinned CUDA Qwen3-Embedding-0.6B embeddings already encoded in private volume.
Output: dense top-50 pool for ALL 1000 queries, plus matching teacher pair text.
No judgments/qrels are read or uploaded.
"""
import os, json, hashlib, time
import modal
app=modal.App("h8-full-dense-retrieval-rsai-r-20261116")
vol=modal.Volume.from_name("h8-disco-ttt-rsai-r-20261116",create_if_missing=False)
img=modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.7.1", "transformers==4.53.2", "huggingface_hub==0.33.4",
    "numpy<2.3", "pyarrow==20.0.0")

@app.function(image=img,gpu=["L4","A10","A100","T4"],cpu=4,memory=8192,
              volumes={"/vol":vol},timeout=900,max_containers=1)
def retrieve():
    import numpy as np, torch, pyarrow.parquet as pq
    t0=time.monotonic()
    D=np.load("/vol/out/corpus_qwen3emb06b_f32.npy",mmap_mode="r")
    Q=np.load("/vol/out/queries_qwen3emb06b_f32.npy")
    corpus_ids=json.load(open("/vol/out/corpus_ids.json"))
    qids=json.load(open("/vol/out/query_ids.json"))
    assert D.shape==(225621,1024) and Q.shape==(1000,1024)
    assert len(corpus_ids)==len(D) and len(qids)==len(Q)
    mat=torch.from_numpy(np.asarray(D)).cuda()
    qry=torch.from_numpy(Q).cuda()
    torch.backends.cuda.matmul.allow_tf32=False
    scores=qry@mat.T
    topv,topi=torch.topk(scores,k=50,dim=1,largest=True,sorted=True)
    indices=topi.cpu().numpy()
    values=topv.cpu().numpy()
    del mat,qry,scores,topv,topi
    dt=pq.read_table("/vol/data/corpus.parquet",columns=["_id","title","text"]).to_pydict()
    qt=pq.read_table("/vol/data/queries.parquet",columns=["_id","text"]).to_pydict()
    assert [str(i) for i in dt["_id"]]==corpus_ids
    assert [str(i) for i in qt["_id"]]==qids
    texts=[(f"{a} {b}".strip() if a else b) for a,b in zip(dt["title"],dt["text"])]
    qtext={str(i):x for i,x in zip(qt["_id"],qt["text"])}
    pools={}
    pairs=[]
    for qi,qid in enumerate(qids):
        inds=indices[qi].tolist()
        ids=[corpus_ids[z] for z in inds]
        pools[qid]={"dids":ids,"scores":values[qi].astype(float).tolist()}
        for doc_id,row in zip(ids,inds):
            pairs.append({"qid":qid,"did":doc_id,"query":qtext[qid],"doc":texts[row]})
    ordered=sorted(qids,key=lambda q:hashlib.sha256((q+"20261116").encode()).hexdigest())
    os.makedirs("/vol/out",exist_ok=True)
    json.dump(pools,open("/vol/out/pools_all1000_top50.json","w"))
    json.dump(pairs,open("/vol/out/pairs_all1000_top50.json","w"))
    json.dump({"order_first_400":ordered[:400],"smoke":ordered[:8],"dev32":ordered[:32],
               "confirm":ordered[400:]},open("/vol/out/SPLIT.json","w"))
    files=["pools_all1000_top50.json","pairs_all1000_top50.json","SPLIT.json"]
    manifest={"task_id":"ceous-disco-ttt-20261116-0820","status":"qrel_blind",
              "device":torch.cuda.get_device_name(0),"query_count":1000,
              "corpus_count":225621,"pool_size":50,"pair_count":len(pairs),
              "logical_teacher_calls_when_scoring_all":len(pairs),
              "duration_s":round(time.monotonic()-t0,2),
              "sha256":{n:hashlib.sha256(open("/vol/out/"+n,"rb").read()).hexdigest() for n in files}}
    json.dump(manifest,open("/vol/out/RETRIEVAL_MANIFEST.json","w"),indent=2)
    vol.commit()
    return manifest

@app.local_entrypoint()
def main():
    print("H8_FULL_RETRIEVAL",json.dumps(retrieve.remote()),flush=True)
