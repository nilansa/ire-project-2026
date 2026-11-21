"""EXPLORATORY query-wise TTT-Embed nDCG@10 per selector (paper settings). Reads qrels. Refuses unless D0 decision exists.
Usage: <split> <teacher.jsonl>"""
import sys, os, json, time, numpy as np
sys.path.insert(0, __file__.rsplit("/",1)[0]); from h8common import *
name,tfile=sys.argv[1],sys.argv[2]
assert os.path.exists(f"{OUT}/D0_SUMMARY_{name}.json"), "D0 summary must be written first"
d0=json.load(open(f"{OUT}/D0_SUMMARY_{name}.json")); assert d0["qrels_read"] is False
import pyarrow.parquet as pq
qr=pq.read_table(f"{S}/data/hotpotqa_hn/data/test-00000-of-00001.parquet").to_pydict(); gold={}
for q,d,s in zip(qr["query-id"],qr["corpus-id"],qr["score"]):
    if s>0: gold.setdefault(str(q),set()).add(str(d))
pools=json.load(open(f"{OUT}/pools_{name}.json")); sels=json.load(open(f"{OUT}/selections_{name}.json")); T={}
for l in open(tfile): r=json.loads(l); T[(r["qid"],r["did"])]=r["r"]
D,Q,dids,qids=load_vectors(); D=np.asarray(D); qi={q:i for i,q in enumerate(qids)}; di={d:i for i,d in enumerate(dids)}; dids=np.array(dids)
def unit(a): return a/np.linalg.norm(a)
def lossgrad(v,x,Z,r):
    u=x+v; n=np.linalg.norm(u); e=u/n; pT=softmax(r/TAU_T); pS=softmax(Z@e/TAU_S); g=Z.T@(pS-pT)/TAU_S
    return (g-e*(e@g))/n+2*LAM*v
def ttt(x,Z,r,steps=300,lr=0.01):
    v=np.zeros_like(x); m=np.zeros_like(x); s=np.zeros_like(x)
    for t in range(1,steps+1):
        g=lossgrad(v,x,Z,r); m=.9*m+.1*g; s=.999*s+.001*g*g; v-=lr*(m/(1-.9**t))/(np.sqrt(s/(1-.999**t))+1e-8)
    return v
def top10(qv): s=D@qv; t=np.argpartition(-s,10)[:10]; return list(dids[t[np.argsort(-s[t],kind="stable")]])
def metrics(rank,G):
    dcg=sum(1/np.log2(i+2) for i,d in enumerate(rank[:10]) if d in G); idcg=sum(1/np.log2(i+2) for i in range(min(10,len(G))))
    h=len(set(rank[:10])&G); return dict(ndcg10=dcg/idcg,recall10=h/len(G),both10=float(h==len(G)))
rows=[]; t0=time.time(); alpha=0.5; arms=["dense","mmr","kcenter","dopt","disco"]
for q,p in pools.items():
    x=Q[qi[q]].astype(np.float64); G=gold[q]; base=top10(x.astype(np.float32)); row=dict(qid=q,gold_in_top10=len(set(p["dids"][:10])&G),gold_in_pool50=len(set(p["dids"])&G))
    row["raw"]=metrics(base,G)
    rr=sorted(p["dids"][:10],key=lambda d:-T[(q,d)]); row["rerank_dense10"]=metrics(rr,G)
    for a in arms+["random"]:
        Ss=sels[q][a] if a!="random" else sels[q]["random"]; Ss=[Ss] if a!="random" else Ss; ms=[]; ranks=[]
        for S_ in Ss:
            Z=D[[di[d] for d in S_]].astype(np.float64); r=np.array([T[(q,d)] for d in S_]); v=ttt(x,Z,r)
            rk=top10(unit(x+alpha*v).astype(np.float32)); ms.append(metrics(rk,G)); ranks.append(rk)
        row[f"ttt_{a}"]={k:float(np.mean([m[k] for m in ms])) for k in ms[0]}
        if a!="random":
            row[f"ttt_{a}_top10"]=ranks[0]
            if a=="disco":  # shuffled-teacher negative control on the same pairs
                rs=np.random.default_rng(int(q[:8],16)).permutation(r); v=ttt(x,Z,rs); row["ttt_disco_shuffled"]=metrics(top10(unit(x+alpha*v).astype(np.float32)),G)
                pos=sorted(p["dids"].index(d) for d in S_); lst=list(p["dids"]); od=sorted(S_,key=lambda d:-T[(q,d)])
                for ps,d in zip(pos,od): lst[ps]=d
                row["rerank_slot_disco"]=metrics(lst,G)
    row["raw_top10"]=base; rows.append(row)
A=["raw","rerank_dense10","rerank_slot_disco"]+[f"ttt_{a}" for a in arms+["random"]]+["ttt_disco_shuffled"]
rng=np.random.default_rng(20261116)
def boot(v):
    v=np.asarray(v); b=rng.integers(0,len(v),(10000,len(v))); m=v[b].mean(1); return [float(v.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))]
summ=dict(split=name,n=len(rows),label="EXPLORATORY (n small; not the H8-A test; computed after D0 decision)",d0_decision=d0["decision"],
  means={a:{k:float(np.mean([r[a][k] for r in rows])) for k in ["ndcg10","recall10","both10"]} for a in A},
  paired_ndcg10_vs_ttt_dense={a:boot([r[a]["ndcg10"]-r["ttt_dense"]["ndcg10"] for r in rows]) for a in A if a!="ttt_dense"},
  paired_ndcg10_vs_rerank_dense10={a:boot([r[a]["ndcg10"]-r["rerank_dense10"]["ndcg10"] for r in rows]) for a in A if a!="rerank_dense10"},
  gained_lost_vs_ttt_dense={a:dict(gained=int(sum(r[a]["ndcg10"]>r["ttt_dense"]["ndcg10"]+1e-12 for r in rows)),lost=int(sum(r[a]["ndcg10"]<r["ttt_dense"]["ndcg10"]-1e-12 for r in rows))) for a in A if a!="ttt_dense"},
  gold_in_top10_mean=float(np.mean([r["gold_in_top10"] for r in rows])),gold_in_pool50_mean=float(np.mean([r["gold_in_pool50"] for r in rows])),
  teacher_pairs=dict(deployed_per_query_per_arm=10,diagnostic_pool_per_query=50,logical_total_scored=len(T)),
  alpha=alpha,seconds=round(time.time()-t0,1))
json.dump(rows,open(f"{OUT}/explore_per_query_{name}.json","w"),indent=1); json.dump(summ,open(f"{OUT}/EXPLORE_SUMMARY_{name}.json","w"),indent=1)
print(json.dumps(summ,indent=1))
