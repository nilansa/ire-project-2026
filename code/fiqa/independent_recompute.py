"""Independent standard-library metric recomputation after the GPU run only."""
import hashlib,json,math,sys
from pathlib import Path

def main(run_dir,qrels_path):
    p=Path(run_dir); q=Path(qrels_path)
    if hashlib.sha256(q.read_bytes()).hexdigest()!='0d9458899c35efb08c2c3d6330995d0b76851222127e557a6f5b930c88f86261': raise ValueError('qrels mismatch')
    gold=json.loads(q.read_text()); rankings=json.loads((p/'rankings.json').read_text())
    recorded=json.loads((p/'per_query_metrics.json').read_text()); manifest=json.loads((p/'run_manifest.json').read_text())
    if len(manifest['qids'])!=200 or set(recorded)!=set(manifest['qids']): raise ValueError('query mismatch')
    worst=0.; count=0
    for qid in manifest['qids']:
        relevant={d for d,v in gold[qid].items() if float(v)>0}
        ideal=sorted((float(v) for v in gold[qid].values()),reverse=True)[:10]
        idcg=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(ideal))
        for arm,ids in rankings[qid].items():
            dcg=sum((2**float(gold[qid].get(d,0))-1)/math.log2(i+2) for i,d in enumerate(ids[:10]))
            rr=next((1/(i+1) for i,d in enumerate(ids[:10]) if d in relevant),0.)
            values={'ndcg10':dcg/idcg if idcg else 0.,'mrr10':rr,
                    'recall10':len(set(ids[:10])&relevant)/len(relevant) if relevant else 0.,
                    'recall100':len(set(ids[:100])&relevant)/len(relevant) if relevant else 0.}
            for key,value in values.items(): worst=max(worst,abs(value-recorded[qid]['metrics'][arm][key]));count+=1
    if worst>1e-12: raise ValueError('independent metric mismatch '+str(worst))
    result={'status':'pass','queries':len(manifest['qids']),'metric_values_checked':count,'max_absolute_difference':worst}
    with (p/'independent_recompute.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__': main(*sys.argv[1:])
