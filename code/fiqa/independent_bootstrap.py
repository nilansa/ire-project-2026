"""Independent paired-query bootstrap using Python's RNG; no model or selection code."""
import json,random,sys
from pathlib import Path

def main(run_dir):
    p=Path(run_dir); data=json.loads((p/'per_query_metrics.json').read_text()); ids=json.loads((p/'run_manifest.json').read_text())['qids']
    result={}
    for comparison in ('dense10','rankmatched_mean'):
        differences=[]
        for q in ids:
            m=data[q]['metrics']; a=m['coverage']['ndcg10']
            b=sum(m['rankmatched_'+str(i)]['ndcg10'] for i in range(3))/3 if comparison=='rankmatched_mean' else m[comparison]['ndcg10']
            differences.append(a-b)
        rng=random.Random(20261116); n=len(differences)
        boot=sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n for _ in range(10000))
        def quantile(frac):
            pos=frac*(len(boot)-1); lo=int(pos); hi=min(lo+1,len(boot)-1); return boot[lo]+(boot[hi]-boot[lo])*(pos-lo)
        result[comparison]={'delta':sum(differences)/n,'ci95':[quantile(.025),quantile(.975)],'rng':'Python random.Random; independent of primary numpy RNG','bootstrap_samples':10000,'seed':20261116}
    with (p/'independent_bootstrap.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main(*sys.argv[1:])
