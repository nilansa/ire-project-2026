"""H8-D0 on DEV32 (addendum v2). Qrel-free. Disco selections + marginal coverage from Modal coverage job."""
import sys, json, time, hashlib, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, __file__.rsplit("/",1)[0]); from h8common import *
D,Q,dids,qids=load_vectors(); qi={q:i for i,q in enumerate(qids)}; di={d:i for i,d in enumerate(dids)}
pools=load_pools(); T=load_teacher(); C=load_cov(); qs=load_split()["dev32"]
cos=lambda a,b: float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
per=[]
for q in qs:
    p=pools[q]; x=Q[qi[q]].astype(np.float64); Z=np.asarray(D[[di[d] for d in p["dids"]]],dtype=np.float64); r=np.array([T[(q,d)] for d in p["dids"]])
    S={"dense":sel_dense(50),"mmr":sel_mmr(x,Z),"kcenter":sel_kcenter(x,Z),"dopt":sel_dopt(x,Z),"disco":C[q]["disco_idx"]}
    RND=[sel_random(q,j,50) for j in range(20)]
    gM=grad0(x,Z,r); surprise=np.abs(softmax(r/TAU_T)-softmax(Z@x/TAU_S))
    f0=C[q]["first"]; others=[i for i in range(50) if i!=f0]; mg=np.array(C[q]["marg_given_first"])[others]
    rho=spearmanr(mg,surprise[others]).correlation if np.ptp(mg)>0 and np.ptp(surprise[others])>0 else float("nan")
    row=dict(qid=q,rho_cov_surprise=None if np.isnan(rho) else float(rho),cos_random_mean=float(np.mean([cos(grad0(x,Z[s],r[s]),gM) for s in RND])),
             erank_random_mean=float(np.mean([erank(x,Z[s]) for s in RND])))
    for k,s in S.items():
        row[f"cos_{k}"]=cos(grad0(x,Z[s],r[s]),gM); row[f"erank_{k}"]=erank(x,Z[s]); row[f"teacher_mean_{k}"]=float(r[s].mean()); row[f"overlap_dense_{k}"]=len(set(s)&set(range(10)))
    per.append(row)
rng=np.random.default_rng(20261116)
def boot(v):
    v=np.asarray(v); b=rng.integers(0,len(v),(10000,len(v))); m=v[b].mean(1); return [float(v.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))]
rhos=[r["rho_cov_surprise"] for r in per if r["rho_cov_surprise"] is not None]
A=["disco","dense","mmr","kcenter","dopt"]
summ=dict(split="dev32",n=len(per),n_rho_defined=len(rhos),median_rho=float(np.median(rhos)),
  cos_minus_random={k:boot([r[f"cos_{k}"]-r["cos_random_mean"] for r in per]) for k in A},
  mean_cos={**{k:float(np.mean([r[f"cos_{k}"] for r in per])) for k in A},"random":float(np.mean([r["cos_random_mean"] for r in per]))},
  mean_erank={**{k:float(np.mean([r[f"erank_{k}"] for r in per])) for k in A},"random":float(np.mean([r["erank_random_mean"] for r in per]))},
  mean_teacher_r_selected={k:float(np.mean([r[f"teacher_mean_{k}"] for r in per])) for k in A},
  mean_overlap_with_dense10={k:float(np.mean([r[f"overlap_dense_{k}"] for r in per])) for k in A})
mb,lo,hi=summ["cos_minus_random"]["disco"]; a_null=summ["median_rho"]<=0.10; b_null=(mb<0.05) or (lo<=0)
summ.update(D0_a_null=bool(a_null),D0_b_null=bool(b_null),D0_null_holds=bool(a_null and b_null),
  decision="PARK_H8" if (a_null and b_null) else "D0_PASS_RESERVE_H8A",written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),qrels_read=False,
  addendum_sha256="8adcd8043e2a8749f12a2cc76e0f51862c284efb411eaa5d67f4e933ff59b0b5",
  note="D0(a) uses the 49 pool docs other than the DISCo first pick (the conditioning doc has zero marginal gain by construction).")
json.dump(per,open(f"{RES}/d0_per_query_dev32.json","w"),indent=1); json.dump(summ,open(f"{RES}/D0_DECISION_dev32.json","w"),indent=1)
print(json.dumps(summ,indent=1)); print("SHA",hashlib.sha256(open(f"{RES}/D0_DECISION_dev32.json","rb").read()).hexdigest())
