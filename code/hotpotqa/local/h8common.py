"""Shared H8 definitions (frozen per H8_D0_OPERATIONAL_ADDENDUM.md v2, SHA 8adcd804...). No qrels here."""
import os, json, hashlib, numpy as np
S=os.path.expanduser("~/.codex/scratch/cross-surface/IRE/ceous-disco-ttt-20261116-0820/shared")
OUT=f"{S}/out_modal"; RES=f"{S}/results"; TAU_T, TAU_S, LAM, M_POOL, B = 0.02, 0.05, 0.3, 50, 10
def split_order(qids): return sorted(qids, key=lambda q: hashlib.sha256((q+"20261116").encode()).hexdigest())
def load_vectors():
    D=np.load(f"{OUT}/corpus_qwen3emb06b_f32.npy",mmap_mode="r"); Q=np.load(f"{OUT}/queries_qwen3emb06b_f32.npy")
    return D, Q, json.load(open(f"{OUT}/corpus_ids.json")), json.load(open(f"{OUT}/query_ids.json"))
def load_texts():
    import pyarrow.parquet as pq
    c=pq.read_table(f"{S}/data/hotpotqa_hn/corpus/test-00000-of-00001.parquet").to_pydict()
    q=pq.read_table(f"{S}/data/hotpotqa_hn/queries/test-00000-of-00001.parquet").to_pydict()
    dt={str(i):(f"{a} {b}".strip() if a else b) for i,a,b in zip(c["_id"],c["title"],c["text"])}
    return dt, {str(i):t for i,t in zip(q["_id"],q["text"])}
def softmax(a): a=a-a.max(); e=np.exp(a); return e/e.sum()
def proj(x,Z): return Z-np.outer(Z@x,x)
def grad0(x,Z,r):
    pT=softmax(r/TAU_T); pS=softmax(Z@x/TAU_S); return proj(x,Z).T@(pS-pT)/TAU_S
def erank(x,Z):
    P=proj(x,Z); ev=np.clip(np.linalg.eigvalsh(P@P.T),0,None); p=ev/ev.sum(); p=p[p>0]; return float(np.exp(-(p*np.log(p)).sum()))
# ---- selectors: inputs pool order (dense rank), Z (M x p), x; return list of pool indices
def sel_dense(n): return list(range(B))
def sel_random(qid,j,n): return sorted(np.random.default_rng(int(hashlib.sha256((qid+"rand"+str(j)).encode()).hexdigest()[:8],16)).choice(n,B,replace=False).tolist())
def sel_mmr(x,Z,lam=0.5):
    rel=Z@x; S=[0]
    while len(S)<B:
        c=[i for i in range(len(Z)) if i not in S]; sc=[lam*rel[i]-(1-lam)*max(Z[i]@Z[s] for s in S) for i in c]; S.append(c[int(np.argmax(sc))])
    return S
def sel_kcenter(x,Z):
    S=[0]
    while len(S)<B:
        c=[i for i in range(len(Z)) if i not in S]; sc=[min(1-Z[i]@Z[s] for s in S) for i in c]; S.append(c[int(np.argmax(sc))])
    return S
def sel_dopt(x,Z,eps=1e-3):
    P=proj(x,Z); S=[0]
    def ld(ix): G=P[ix]@P[ix].T; return np.linalg.slogdet(np.eye(len(ix))+G/eps)[1]
    while len(S)<B:
        c=[i for i in range(len(Z)) if i not in S]; sc=[ld(S+[i]) for i in c]; S.append(c[int(np.argmax(sc))])
    return S
def coverage(qt,toks,S):
    if not S: return 0.0
    best=np.max(np.concatenate([qt@toks[i].T for i in S],axis=1),axis=1); return float(np.maximum(best,0).sum())
def sel_disco(qt,toks):
    n=len(toks); S=[]
    while len(S)<B:
        base=coverage(qt,toks,S); c=[i for i in range(n) if i not in S]
        g=[coverage(qt,toks,S+[i])-base for i in c]; mx=max(g); S.append(min(i for i,gi in zip(c,g) if gi==mx))  # tie -> best dense rank
    return S

def load_pools(): return json.load(open(f"{OUT}/pools_all1000_top50.json"))
def load_split(): return json.load(open(f"{OUT}/SPLIT.json"))
def load_teacher():
    T={}
    for fn in ["teacher_smoke8.jsonl","teacher_rest992.jsonl"]:
        for l in open(f"{OUT}/{fn}"): r=json.loads(l); T[(r["qid"],r["did"])]=r["r"]
    return T
def load_cov(): return json.load(open(f"{OUT}/disco_coverage_all1000.json"))
