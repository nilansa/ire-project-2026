#!/usr/bin/env python3
"""Faithful reconstruction of the independent transient audit, not its original code."""
import csv, gzip, json, math, random
from pathlib import Path
from collections import defaultdict
D=Path(__file__).parent/'results/20261115-baseline'
ids=[line.rstrip('\n').split('\t')[1] for line in (D/'corpus-row-ids.tsv').open()]
qrels=defaultdict(dict)
with (D/'selected-qrels.tsv').open() as f:
 for r in csv.DictReader(f,delimiter='\t'): qrels[r['query-id']][r['corpus-id']]=int(r['score'])
arms=['dense','deep100','rocchio:0.05','rerank20:medcpt','rerank50:medcpt','weighted:medcpt:1:t4','tour:medcpt:0.8:stop0:t4','h2:medcpt:0.8:10:t4']
inner_diffs=[]
for split,n in [('dev',405),('inner',100)]:
 with gzip.open(D/f'{split}-score/per_query.csv.gz','rt') as f: scores={(r['qid'],r['arm']):r for r in csv.DictReader(f) if r['arm'] in arms}
 checked=defaultdict(int); pairs={}; cov=[0,0,0,0,0]
 with gzip.open(D/f'{split}-predict/predictions.jsonl.gz','rt') as f:
  for line in f:
   p=json.loads(line); q=p['qid']; arm=p['arm']
   if arm not in arms:continue
   rel=qrels[q]; positive=[v for v in rel.values() if v>0];assert positive
   ranks=[rel.get(ids[x],0) for x in p['ranked']]
   dcg=lambda xs: sum((2**v-1)/math.log2(i+2) for i,v in enumerate(xs[:10]))
   ndcg=dcg(ranks)/dcg(sorted(positive,reverse=True));rec=sum(v>0 for v in ranks[:20])/len(positive)
   row=scores[q,arm];assert abs(ndcg-float(row['ndcg10']))<1e-10 and abs(rec-float(row['recall20']))<1e-10
   checked[arm]+=1;pairs[q,arm]=ndcg
   if arm=='deep100':
    a=sum(rel.get(ids[x],0)>0 for x in p['initial50']);b=sum(rel.get(ids[x],0)>0 for x in p['ranked'][:100]);cov[0]+=len(positive);cov[1]+=a;cov[2]+=b;cov[3]+=not a;cov[4]+=not b
 assert all(checked[a]==n for a in arms),checked
 diffs=[pairs[q,arms[-1]]-pairs[q,arms[-2]] for q in sorted({q for q,a in pairs})]
 if split=='inner': inner_diffs=diffs
 print(split,'checked',dict(checked),'coverage total,50,100,zero50,zero100:',cov,'H2-TOUR',sum(diffs)/len(diffs))
r=random.Random(20261115); boot=sorted(sum(r.choices(inner_diffs,k=len(inner_diffs)))/len(inner_diffs) for _ in range(2000))
print('paired INNER bootstrap order statistics 49/1949:',boot[49],boot[1949])
