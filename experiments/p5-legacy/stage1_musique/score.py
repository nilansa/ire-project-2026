import json,os,numpy as np
D=os.path.dirname(os.path.abspath(__file__))
P=json.load(open(f"{D}/predictions_top5.json")); G=json.load(open(f"{D}/gold_HIDDEN.json")); ids=sorted(P)
rng=np.random.default_rng(0); B=rng.integers(0,len(ids),(1000,len(ids))); out={"n":len(ids)}
for K in (2,3,5):
    fs=np.array([set(G[i])<=set(d["doc_id"] for d in P[i][:K]) for i in ids],float)
    rc=np.array([len(set(G[i])&set(d["doc_id"] for d in P[i][:K]))/2 for i in ids])
    wd=np.array([sum(d["words"] for d in P[i][:K]) for i in ids],float)
    ci=lambda a: [float(np.percentile(a[B].mean(1),2.5)),float(np.percentile(a[B].mean(1),97.5))]
    out[f"K{K}"]={"complete_support":float(fs.mean()),"cs_ci95":ci(fs),"cs_hits":int(fs.sum()),"support_recall":float(rc.mean()),"sr_ci95":ci(rc),"evidence_words_mean":float(wd.mean())}
json.dump(out,open(f"{D}/metrics.json","w"),indent=1); print(json.dumps(out))
