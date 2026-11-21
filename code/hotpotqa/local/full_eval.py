"""H8 full evaluation. Usage: dev400 | confirm600. Reads qrels ONLY for scoring. CONFIRM refuses until DEV decision file exists.
Arms: raw dense; teacher rerank of dense top-10 (10 pairs); TTT-Embed query-wise (paper settings, alpha=0.5) on B=10 pairs from
selectors dense(=standard TTT-Embed), disco, mmr, kcenter, dopt, random(20 draws avg); rerank_slot_disco; shuffled-teacher control (disco & dense)."""
import sys, os, json, time, hashlib, numpy as np
sys.path.insert(0, __file__.rsplit("/",1)[0]); from h8common import *
split=sys.argv[1]; assert split in ("dev400","confirm600")
d0=json.load(open(f"{RES}/D0_DECISION_dev32.json")); assert d0["qrels_read"] is False
if split=="confirm600": assert os.path.exists(f"{RES}/H8A_DECISION_dev400.json"), "DEV decision must be frozen first"
sp=load_split(); qs=sp["order_first_400"] if split=="dev400" else sp["confirm"]
import pyarrow.parquet as pq
qr=pq.read_table(f"{S}/data/hotpotqa_hn/data/test-00000-of-00001.parquet").to_pydict(); gold={}
for q,d,s in zip(qr["query-id"],qr["corpus-id"],qr["score"]):
    if s>0: gold.setdefault(str(q),set()).add(str(d))
D,Q,dids,qids=load_vectors(); D=np.ascontiguousarray(D); qi={q:i for i,q in enumerate(qids)}; di={d:i for i,d in enumerate(dids)}; dids=np.array(dids)
pools=load_pools(); T=load_teacher(); C=load_cov()
def unit(a): return a/np.linalg.norm(a)
def ttt(x,Z,r,steps=300,lr=0.01):
    v=np.zeros_like(x); m=np.zeros_like(x); s=np.zeros_like(x)
    for t in range(1,steps+1):
        u=x+v; n=np.linalg.norm(u); e=u/n; pT=softmax(r/TAU_T); pS=softmax(Z@e/TAU_S); g=Z.T@(pS-pT)/TAU_S
        g=(g-e*(e@g))/n+2*LAM*v; m=.9*m+.1*g; s=.999*s+.001*g*g; v-=lr*(m/(1-.9**t))/(np.sqrt(s/(1-.999**t))+1e-8)
    return v
def tops(V):  # V: k x p float32 -> list of top10 id lists
    Sc=V@D.T; out=[]
    for row in Sc: t=np.argpartition(-row,10)[:10]; out.append(list(dids[t[np.argsort(-row[t],kind="stable")]]))
    return out
def met(rank,G):
    dcg=sum(1/np.log2(i+2) for i,d in enumerate(rank[:10]) if d in G); idcg=sum(1/np.log2(i+2) for i in range(min(10,len(G))))
    h=len(set(rank[:10])&G); return dict(ndcg10=dcg/idcg,recall10=h/len(G),both10=float(h==len(G)))
ARMS=["dense","disco","mmr","kcenter","dopt"]; rows=[]; t0=time.time(); alpha=0.5
for n,q in enumerate(qs):
    p=pools[q]; x=Q[qi[q]].astype(np.float64); Zp=np.asarray(D[[di[d] for d in p["dids"]]],dtype=np.float64); rp=np.array([T[(q,d)] for d in p["dids"]]); G=gold[q]
    sel={"dense":sel_dense(50),"disco":C[q]["disco_idx"],"mmr":sel_mmr(x,Zp),"kcenter":sel_kcenter(x,Zp),"dopt":sel_dopt(x,Zp)}
    jobs=[(a,s,rp[s]) for a,s in sel.items()]+[("random",sel_random(q,j,50),None) for j in range(20)]
    rs=np.random.default_rng(int(hashlib.sha256((q+"shuffle").encode()).hexdigest()[:8],16))
    jobs+= [("shuf_"+a,sel[a],rs.permutation(rp[sel[a]])) for a in ("dense","disco")]
    V=[]
    for a,s,r in jobs:
        r=rp[s] if r is None else r; v=ttt(x,Zp[s],r); V.append(unit(x+alpha*v))
    V=np.array([x]+V,dtype=np.float32); R=tops(V)
    row=dict(qid=q,gold_in_dense10=len(set(p["dids"][:10])&G),gold_in_pool50=len(set(p["dids"])&G),raw=met(R[0],G),raw_top10=R[0])
    row["rerank_dense10"]=met(sorted(p["dids"][:10],key=lambda d:-T[(q,d)]),G)
    pos=sorted(sel["disco"]); lst=list(p["dids"]); od=sorted([p["dids"][i] for i in sel["disco"]],key=lambda d:-T[(q,d)])
    for ps,d in zip(pos,od): lst[ps]=d
    row["rerank_slot_disco"]=met(lst,G)
    rnd=[]
    for (a,s,_),rk in zip(jobs,R[1:]):
        m=met(rk,G)
        if a=="random": rnd.append(m)
        else:
            row["ttt_"+a]=m; row["ttt_"+a+"_top10"]=rk
            if a in ARMS: row["sel_"+a]=[p["dids"][i] for i in s]; row["teacher_mean_"+a]=float(rp[s].mean())
    row["ttt_random"]={k:float(np.mean([m[k] for m in rnd])) for k in rnd[0]}
    rows.append(row)
    if n%50==0: print(json.dumps(dict(done=n+1,of=len(qs),s=round(time.time()-t0,1))),flush=True)
A=["raw","rerank_dense10","rerank_slot_disco"]+["ttt_"+a for a in ARMS+["random"]]+["ttt_shuf_dense","ttt_shuf_disco"]
rng=np.random.default_rng(20261116)
def boot(v):
    v=np.asarray(v); b=rng.integers(0,len(v),(10000,len(v))); m=v[b].mean(1); return [float(v.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))]
means={a:{k:float(np.mean([r[a][k] for r in rows])) for k in ["ndcg10","recall10","both10"]} for a in A}
elig=["rerank_dense10","rerank_slot_disco","ttt_dense","ttt_random","ttt_mmr","ttt_kcenter","ttt_dopt"]
best=max(elig,key=lambda a:means[a]["ndcg10"])
diff={a:boot([r["ttt_disco"]["ndcg10"]-r[a]["ndcg10"] for r in rows]) for a in A if a!="ttt_disco"}
gl={a:dict(gained=int(sum(r["ttt_disco"]["ndcg10"]>r[a]["ndcg10"]+1e-12 for r in rows)),lost=int(sum(r["ttt_disco"]["ndcg10"]<r[a]["ndcg10"]-1e-12 for r in rows))) for a in A if a!="ttt_disco"}
summ=dict(split=split,n=len(rows),means=means,strongest_eligible_baseline=best,disco_minus_arm_ndcg10=diff,disco_gained_lost=gl,
  ttt_dense_minus_rerank_dense10=boot([r["ttt_dense"]["ndcg10"]-r["rerank_dense10"]["ndcg10"] for r in rows]),
  ttt_dense_minus_raw=boot([r["ttt_dense"]["ndcg10"]-r["raw"]["ndcg10"] for r in rows]),
  gold_in_dense10_mean=float(np.mean([r["gold_in_dense10"] for r in rows])),gold_in_pool50_mean=float(np.mean([r["gold_in_pool50"] for r in rows])),
  mean_teacher_r_selected={a:float(np.mean([r["teacher_mean_"+a] for r in rows])) for a in ARMS},
  budget=dict(teacher_pairs_deployed_per_query_per_arm=dict(raw=0,rerank_dense10=10,rerank_slot_disco=10,**{"ttt_"+a:10 for a in ARMS+["random"]}),
     teacher_pairs_diagnostic_pool_per_query=50,full_corpus_scans_per_query=dict(raw=1,rerank_dense10=1,rerank_slot_disco=1,ttt_any=2),
     returned_slots_final=10,disco_extra="bert-base-uncased tokens for 50 pool docs + query; greedy coverage (see coverage manifest)"),
  eval_seconds=round(time.time()-t0,1),label=("DEV screen (H8-A decision split)" if split=="dev400" else "CONFIRM (frozen after DEV decision)"),d0_decision=d0["decision"])
if split=="dev400":
    m,lo,hi=diff[best]; mr,lor,_=diff["rerank_slot_disco"] if best!="rerank_slot_disco" else diff[best]
    passA=(m>=0.010) and (lo>0) and (lor>0)
    dec=dict(rule="ttt_disco - strongest eligible >= +0.010 nDCG@10 with paired 95% CI lower > 0, and > rerank_slot_disco",
             strongest=best,delta=diff[best],passed=bool(passA),d0_decision=d0["decision"],
             h8a_status=("H8-A not authorized by D0 (PARK); DEV numbers descriptive" if d0["decision"]=="PARK_H8" else ("PASS" if passA else "FAIL -> PARK H8-A")),
             written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    json.dump(dec,open(f"{RES}/H8A_DECISION_dev400.json","w"),indent=1); summ["decision"]=dec
json.dump(rows,open(f"{RES}/eval_per_query_{split}.json","w")); json.dump(summ,open(f"{RES}/EVAL_SUMMARY_{split}.json","w"),indent=1)
print(json.dumps(summ,indent=1))
