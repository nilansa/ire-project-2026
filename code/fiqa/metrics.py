"""Evaluation-only metrics; selector and adaptation never import this module."""
import math
import numpy as np

def metrics(ids, gold):
    if len(set(ids))!=len(ids) or any(not math.isfinite(float(v)) or float(v)<0 for v in gold.values()):
        raise ValueError('invalid ranking or qrels')
    positive={str(k) for k,v in gold.items() if float(v)>0}
    gains=sorted([float(v) for v in gold.values()],reverse=True)[:10]
    ideal=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(gains))
    dcg=sum((2**float(gold.get(d,0))-1)/math.log2(i+2) for i,d in enumerate(ids[:10]))
    rr=next((1/(i+1) for i,d in enumerate(ids[:10]) if d in positive),0.)
    return {'ndcg10':dcg/ideal if ideal else 0.,'mrr10':rr,
            'recall10':len(set(ids[:10])&positive)/len(positive) if positive else 0.,
            'recall100':len(set(ids[:100])&positive)/len(positive) if positive else 0.}

def paired(a,b,n=10000,seed=20261116):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim!=1 or a.shape!=b.shape or not len(a) or not np.isfinite(a-b).all(): raise ValueError('invalid paired inputs')
    diff=a-b; rng=np.random.default_rng(seed)
    boots=np.array([diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(n)])
    return {'delta':float(diff.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),
            'win_tie_loss':[int((diff>1e-12).sum()),int((abs(diff)<=1e-12).sum()),int((diff< -1e-12).sum())]}
