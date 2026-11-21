"""H8 local MPS corpus/query encoder (same definitions as modal/h8_modal.py). Resumable shards. Frozen weights."""
import os, sys, json, time, hashlib
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
import numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
S=os.path.expanduser("~/.codex/scratch/cross-surface/IRE/ceous-disco-ttt-20261116-0820/shared")
EMB=os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
QP="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
OUT=f"{S}/out"; SH=f"{OUT}/shards"; os.makedirs(SH,exist_ok=True)
c=pq.read_table(f"{S}/data/hotpotqa_hn/corpus/test-00000-of-00001.parquet").to_pydict()
ids=[str(x) for x in c["_id"]]; texts=[(f"{a} {b}".strip() if a else b) for a,b in zip(c["title"],c["text"])]
q=pq.read_table(f"{S}/data/hotpotqa_hn/queries/test-00000-of-00001.parquet").to_pydict(); qids=[str(x) for x in q["_id"]]
tok=AutoTokenizer.from_pretrained(EMB,padding_side="left"); m=AutoModel.from_pretrained(EMB,torch_dtype=torch.float16).to("mps").eval()
def enc(batch,maxlen=512):
    b=tok(batch,padding=True,truncation=True,max_length=maxlen,return_tensors="pt").to("mps")
    with torch.no_grad(): h=m(**b).last_hidden_state[:,-1].float()
    return torch.nn.functional.normalize(h,dim=-1).cpu().numpy()
t0=time.time()
Q=np.concatenate([enc([QP+x for x in q["text"][i:i+64]]) for i in range(0,len(qids),64)])
lens=[len(t) for t in texts]; order=np.argsort(lens,kind="stable"); CH=8192; BS=64
nshard=(len(order)+CH-1)//CH; ntr=0
for k in range(nshard):
    f=f"{SH}/shard_{k:03d}.npy"
    if os.path.exists(f): continue
    idx=order[k*CH:(k+1)*CH]; E=np.zeros((len(idx),1024),np.float32)
    for s in range(0,len(idx),BS):
        bt=[texts[i] for i in idx[s:s+BS]]; E[s:s+BS]=enc(bt)
    np.save(f+".tmp.npy",E); os.replace(f+".tmp.npy",f)
    print(json.dumps(dict(shard=k,of=nshard,elapsed_s=round(time.time()-t0,1))),flush=True)
D=np.zeros((len(ids),1024),np.float32)
for k in range(nshard): D[order[k*CH:(k+1)*CH]]=np.load(f"{SH}/shard_{k:03d}.npy")
ntr=sum(1 for t in texts if len(t)>1500 and len(tok(t)["input_ids"])>512)
np.save(f"{OUT}/corpus_qwen3emb06b_f32.npy",D); np.save(f"{OUT}/queries_qwen3emb06b_f32.npy",Q)
json.dump(ids,open(f"{OUT}/corpus_ids.json","w")); json.dump(qids,open(f"{OUT}/query_ids.json","w"))
nrm=np.linalg.norm(D,axis=1)
man=dict(device="mps",dtype="float16 compute/float32 store",torch=torch.__version__,corpus_shape=list(D.shape),query_shape=list(Q.shape),
  corpus_trunc_gt512=int(ntr),finite=bool(np.isfinite(D).all() and np.isfinite(Q).all()),norm_min=float(nrm.min()),norm_max=float(nrm.max()),
  total_s=round(time.time()-t0,1),sha256={n:hashlib.sha256(open(f"{OUT}/{n}","rb").read()).hexdigest() for n in
  ["corpus_qwen3emb06b_f32.npy","queries_qwen3emb06b_f32.npy","corpus_ids.json","query_ids.json"]})
json.dump(man,open(f"{OUT}/ENCODE_MANIFEST.json","w"),indent=1); print("ENCODE_DONE",json.dumps(man),flush=True)
