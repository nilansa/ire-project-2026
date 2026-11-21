"""H8-D0 qrel-free analysis + decision (addendum v2). Usage: <split> <teacher.jsonl>. NEVER reads qrels."""
import sys, json, time, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, __file__.rsplit("/",1)[0]); from h8common import *
name, tfile = sys.argv[1], sys.argv[2]
pools=json.load(open(f"{OUT}/pools_{name}.json")); T={}
for l in open(tfile): r=json.loads(l); T[(r["qid"],r["did"])]=r["r"]
D,Q,dids,qids=load_vectors(); qi={q:i for i,q in enumerate(qids)}; di={d:i for i,d in enumerate(dids)}
cos=lambda a,b: float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
per=[]; sels={}
for q,p in pools.items():
    x=Q[qi[q]].astype(np.float64); Z=np.asarray(D[[di[d] for d in p["dids"]]],dtype=np.float64); r=np.array([T[(q,d)] for d in p["dids"]])
    bz=np.load(f"{OUT}/bert_{name}/{q}.npz"); toks=[bz["d"][bz["off"][i]:bz["off"][i+1]] for i in range(len(p["dids"]))]; qt=bz["q"]
    S={"dense":sel_dense(50),"mmr":sel_mmr(x,Z),"kcenter":sel_kcenter(x,Z),"dopt":sel_dopt(x,Z),"disco":sel_disco(qt,toks)}
    RND=[sel_random(q,j,50) for j in range(20)]
    gM=grad0(x,Z,r); pT=softmax(r/TAU_T); pS=softmax(Z@x/TAU_S); surprise=np.abs(pT-pS)
    f0=S["disco"][0]; others=[i for i in range(50) if i!=f0]; c0=coverage(qt,toks,[f0])
    mg=np.array([coverage(qt,toks,[f0,i])-c0 for i in others])
    rho=spearmanr(mg,surprise[others]).correlation if np.ptp(mg)>0 and np.ptp(surprise[others])>0 else float("nan")
    row=dict(qid=q,rho_cov_surprise=None if np.isnan(rho) else float(rho),gM_norm=float(np.linalg.norm(gM)),
             cos_random_mean=float(np.mean([cos(grad0(x,Z[s],r[s]),gM) for s in RND])),
             erank_random_mean=float(np.mean([erank(x,Z[s]) for s in RND])))
    for k,s in S.items():
        row[f"cos_{k}"]=cos(grad0(x,Z[s],r[s]),gM); row[f"erank_{k}"]=erank(x,Z[s]); row[f"teacher_mean_{k}"]=float(r[s].mean())
        row[f"cov_{k}"]=coverage(qt,toks,s); row[f"overlap_dense_{k}"]=len(set(s)&set(range(10)))
    row["cov_random_mean"]=float(np.mean([coverage(qt,toks,s) for s in RND])); row["teacher_mean_pool"]=float(r.mean())
    per.append(row); sels[q]={k:[p["dids"][i] for i in s] for k,s in S.items()}; sels[q]["random"]=[[p["dids"][i] for i in s] for s in RND]
rng=np.random.default_rng(20261116)
def boot(v):
    v=np.asarray(v); b=rng.integers(0,len(v),(10000,len(v))); m=v[b].mean(1); return float(v.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))
rhos=[r["rho_cov_surprise"] for r in per if r["rho_cov_surprise"] is not None]
summ=dict(split=name,n=len(per),n_rho_defined=len(rhos),median_rho=float(np.median(rhos)),
          **{f"cos_{k}_minus_random":boot([r[f"cos_{k}"]-r["cos_random_mean"] for r in per]) for k in ["disco","dense","mmr","kcenter","dopt"]},
          **{f"mean_cos_{k}":float(np.mean([r[f"cos_{k}"] for r in per])) for k in ["disco","dense","mmr","kcenter","dopt","random_mean"]},
          **{f"mean_erank_{k}":float(np.mean([r[f"erank_{k}"] for r in per])) for k in ["disco","dense","mmr","kcenter","dopt","random_mean"]},
          **{f"mean_teacher_{k}":float(np.mean([r[f"teacher_mean_{k}"] for r in per])) for k in ["disco","dense","mmr","kcenter","dopt"]},
          mean_overlap_disco_dense=float(np.mean([r["overlap_dense_disco"] for r in per])))
mb,lo,hi=summ["cos_disco_minus_random"]; a_null=summ["median_rho"]<=0.10; b_null=(mb<0.05) or (lo<=0)
summ.update(D0_a_null=bool(a_null),D0_b_null=bool(b_null),D0_null_holds=bool(a_null and b_null),
            decision=("PARK_H8" if (a_null and b_null) else "D0_PASS_RESERVE_H8A") if name=="dev32" else "SMOKE_NO_DECISION",
            written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),qrels_read=False)
json.dump(per,open(f"{OUT}/d0_per_query_{name}.json","w"),indent=1); json.dump(sels,open(f"{OUT}/selections_{name}.json","w"))
json.dump(summ,open(f"{OUT}/D0_SUMMARY_{name}.json","w"),indent=1); print(json.dumps(summ,indent=1))
