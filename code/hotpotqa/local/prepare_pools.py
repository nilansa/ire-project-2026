"""Build qrel-blind split + dense top-50 pools + teacher pair lists. Usage: <split: smoke|dev32>"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit("/",1)[0]); from h8common import *
name=sys.argv[1]; n={"smoke":8,"dev32":32}[name]
D,Q,dids,qids=load_vectors(); qi={q:i for i,q in enumerate(qids)}; order=split_order(qids)
json.dump(dict(order_first_400=order[:400],smoke=order[:8],dev32=order[:32]),open(f"{OUT}/SPLIT.json","w"),indent=1)
dt,qt=load_texts(); D=np.asarray(D); pools={}; pairs=[]
for q in order[:n]:
    s=D@Q[qi[q]]; top=np.argpartition(-s,M_POOL)[:M_POOL]; top=top[np.argsort(-s[top],kind="stable")]
    pools[q]=dict(dids=[dids[i] for i in top],scores=[float(s[i]) for i in top])
    pairs+= [dict(qid=q,did=dids[i],query=qt[q],doc=dt[dids[i]]) for i in top]
json.dump(pools,open(f"{OUT}/pools_{name}.json","w")); json.dump(pairs,open(f"{OUT}/pairs_{name}.json","w"))
print("POOLS",name,len(pools),"pairs",len(pairs))
