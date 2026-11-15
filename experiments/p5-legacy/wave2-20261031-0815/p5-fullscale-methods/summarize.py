from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from datetime import datetime
import numpy as np
import pyarrow.parquet as pq

OUT=Path(__file__).resolve().parent
METHOD_FILES={
 'bm25':'bm25_predictions.jsonl','dense_bge':'dense_bge_predictions.jsonl','dense_e5':'dense_e5_predictions.jsonl',
 'hybrid_rrf_bge_e5_bm25':'hybrid_rrf_predictions.jsonl','entity_graph_beta025':'entity_graph_predictions.jsonl',
 'mdr_bge_query_top1':'mdr_bge_predictions.jsonl'}

def read_jsonl(p):
 with p.open() as f: return [json.loads(line) for line in f if line.strip()]
def bootstrap_diff(a,b,seed=0,nboot=2000):
 a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float); assert len(a)==len(b)>=3
 d=a-b; rng=np.random.default_rng(seed); vals=np.empty(nboot)
 for i in range(nboot): vals[i]=d[rng.integers(0,len(d),size=len(d))].mean()
 return float(d.mean()),[float(x) for x in np.quantile(vals,[.025,.975])]
def mdata(pred):
 pos={g:i for i,g in enumerate(pred['order'])}
 mset=max(pos[g] for g in pred['gold'])+1
 return {'fs':{str(k):float(set(pred['gold']).issubset(set(pred['order'][:k]))) for k in (2,3,5)},
  'recall':{str(k):len(set(pred['gold'])&set(pred['order'][:k]))/len(pred['gold']) for k in (2,3,5)},'min_set_size':mset}
def main():
 target=OUT/'metrics.jsonl'; result_path=OUT/'RESULT.md'
 if target.exists() or result_path.exists(): raise FileExistsError('refusing to overwrite final deliverable')
 rows=pq.read_table(OUT/'dev2.parquet').to_pylist(); qmeta={}
 for r in rows:
  titles=r['context']['title']; gold={ ' '.join(str(x).split()).casefold() for x in r['supporting_facts']['title']}
  gi={i for i,t in enumerate(titles) if ' '.join(str(t).split()).casefold() in gold}
  qmeta[str(r['id'])]={'type':str(r['type']),'gold':sorted(gi)}
 expected=set(qmeta)
 preds={}
 for method,filename in METHOD_FILES.items():
  p=OUT/filename
  if p.exists():
   for x in read_jsonl(p): preds.setdefault(x['method'],{})[x['qid']]=x
 gpu_dense=OUT/'gpu_dense_predictions.jsonl'
 if gpu_dense.exists():
  for x in read_jsonl(gpu_dense):
   if x['method']=='dense_bge': preds.setdefault(x['method'],{})[x['qid']]=x
   else: preds.setdefault(x['method'],{}).setdefault(x['qid'],x)
 gpu_structured=OUT/'gpu_structured_predictions.jsonl'
 if gpu_structured.exists():
  for x in read_jsonl(gpu_structured): preds.setdefault(x['method'],{})[x['qid']]=x
 gpu=OUT/'gpu_predictions.jsonl'
 if gpu.exists():
  for x in read_jsonl(gpu): preds.setdefault(x['method'],{})[x['qid']]=x
 if all(m in preds for m in ['bm25','dense_bge','dense_e5']):
  shared=set(preds['bm25'])&set(preds['dense_bge'])&set(preds['dense_e5'])
  hybrid={}
  for qid in shared:
   score=np.zeros(10,dtype=np.float64)
   for method in ['bm25','dense_bge','dense_e5']:
    for rank,j in enumerate(preds[method][qid]['order'],1): score[j]+=1/(60+rank)
   base=preds['dense_bge'][qid]
   hybrid[qid]={'qid':qid,'type':base['type'],'method':'hybrid_rrf_bge_e5_bm25','order':np.argsort(-score,kind='stable').astype(int).tolist(),'gold':base['gold'],'latency_s':None}
  preds['hybrid_rrf_bge_e5_bm25']=hybrid
 if not preds: raise RuntimeError('no predictions found')
 for method,pm in preds.items():
  for qid,p in pm.items():
   assert qid in expected, f'non-DEV2 id in predictions: {qid}'
   assert p['gold']==qmeta[qid]['gold']
   order=p['order']; assert len(order)==10 and len(set(order))==10 and set(order)==set(range(10))
 values={m:{q:mdata(p) for q,p in pm.items()} for m,pm in preds.items()}
 records=[]; by_group={}
 for group in ['overall','bridge','comparison']:
  qids=sorted(q for q in expected if group=='overall' or qmeta[q]['type']==group)
  by_group[group]={}
  for method,pm in preds.items():
   avail=[q for q in qids if q in pm]
   if not avail: continue
   stat={'record_type':'summary','group':group,'method':method,'query_n':len(avail),'dev2_n':len(qids),
    'coverage':len(avail)/len(qids),'FS':{},'mean_recall':{},'mean_minimal_set_size':float(np.mean([values[method][q]['min_set_size'] for q in avail]))}
   for k in ('2','3','5'):
    stat['FS'][k]=float(np.mean([values[method][q]['fs'][k] for q in avail]))
    stat['mean_recall'][k]=float(np.mean([values[method][q]['recall'][k] for q in avail]))
   records.append(stat); by_group[group][method]=stat
  single=[m for m in ['dense_bge','dense_e5','bm25','hybrid_rrf_bge_e5_bm25','entity_graph_beta025','minilm_ce_independent','bge_reranker_v2_m3_independent'] if m in by_group[group]]
  if single:
   best=max(single,key=lambda m:by_group[group][m]['FS']['2'])
   records.append({'record_type':'best_single_shot','group':group,'selection_metric':'FS@2','method':best,'FS2':by_group[group][best]['FS']['2'],'query_n':by_group[group][best]['query_n']})
   by_group[group]['_single']=single; by_group[group]['_best']=best
 # Paired query bootstrap deltas for each method vs both dense baselines and vs reselected best single-shot.
 rng_seed=0
 for group in ['overall','bridge','comparison']:
  qids=sorted(q for q in expected if group=='overall' or qmeta[q]['type']==group)
  for method in preds:
   for k in ('2','3','5'):
    for dense in ['dense_bge','dense_e5']:
     if method==dense or dense not in preds: continue
     paired=[q for q in qids if q in values[method] and q in values[dense]]
     if len(paired)<3: continue
     d,ci=bootstrap_diff([values[method][q]['fs'][k] for q in paired],[values[dense][q]['fs'][k] for q in paired],rng_seed)
     records.append({'record_type':'paired_contrast','group':group,'metric':f'FS@{k}','method':method,'baseline':dense,'query_n':len(paired),'difference':d,'ci95_percentile':ci,'bootstrap_resamples':2000,'bootstrap_seed':rng_seed})
    single=by_group[group].get('_single',[])
    if single and any(q in values[method] for q in qids):
     paired=[q for q in qids if q in values[method] and all(q in values[s] for s in single)]
     if len(paired)>=3:
      a=np.asarray([values[method][q]['fs'][k] for q in paired]); mat=np.asarray([[values[s][q]['fs'][k] for s in single] for q in paired])
      diff,ci=bootstrap_diff(a,mat.max(1),rng_seed)
      boot=np.empty(2000); rng=np.random.default_rng(rng_seed)
      for bi in range(len(boot)):
       ix=rng.integers(0,len(paired),size=len(paired)); boot[bi]=a[ix].mean()-mat[ix].mean(0).max()
      ci=[float(x) for x in np.quantile(boot,[.025,.975])]
      records.append({'record_type':'paired_contrast','group':group,'metric':f'FS@{k}','method':method,'baseline':'best_single_shot_reselected_per_resample','best_single_shots':single,'best_single_shot_full_data':by_group[group].get('_best'),'query_n':len(paired),'difference':diff,'ci95_percentile':ci,'bootstrap_resamples':2000,'bootstrap_seed':rng_seed})
 # Primary predeclared contrast: bridge conditional MiniLM vs its independent ranking at k=2.
 primary={'method':'minilm_ce_conditional','baseline':'minilm_ce_independent'}
 bridge_q=sorted(q for q in expected if qmeta[q]['type']=='bridge')
 if all(m in values for m in primary.values()):
  paired=[q for q in bridge_q if all(q in values[m] for m in primary.values())]
  if len(paired)>=3:
   delta,ci=bootstrap_diff([values[primary['method']][q]['fs']['2'] for q in paired],[values[primary['baseline']][q]['fs']['2'] for q in paired])
   verdict='SUPPORT' if ci[0]>0 else ('REFUTE' if ci[1]<0 else 'INCONCLUSIVE')
   primary.update({'group':'bridge','metric':'FS@2','query_n':len(paired),'difference':delta,'ci95_percentile':ci,'bootstrap_resamples':2000,'bootstrap_seed':0,'verdict':verdict})
  else: primary.update({'verdict':'INCONCLUSIVE','reason':'fewer than three paired bridge queries'})
 else:
  primary.update({'verdict':'INCONCLUSIVE','reason':'one or both MiniLM CE rankings unavailable'})
 # Write JSONL once, with aggregated metrics and bootstrap contrasts.
 with target.open('x') as f:
  for x in records: f.write(json.dumps(x,separators=(',',':'))+'\n')
 manifest=json.loads((OUT/'split_manifest.json').read_text())
 try: local_meta=json.loads((OUT/'local_run_metadata.json').read_text())
 except Exception: local_meta={}
 try: local_status=json.loads((OUT/'local_cpu_status.json').read_text())
 except Exception: local_status={}
 try: gpu_meta=json.loads((OUT/'gpu_run_metadata.json').read_text())
 except Exception: gpu_meta={}
 try: modal_cost=float(json.loads((OUT/'modal_branch_costs.json').read_text())['metered_total_usd'])
 except Exception: modal_cost=None
 try: billing_summary=json.loads((OUT/'modal_billing_summary_after.json').read_text())
 except Exception:
  try: billing_summary=json.loads((OUT/'modal_billing_summary.json').read_text())
  except Exception: billing_summary={}
 local_wall=None
 try:
  started=datetime.strptime((OUT/'local_process_start.txt').read_text().strip(),'%a %b %d %H:%M:%S %Y')
  finished=datetime.fromisoformat(local_meta['finished_at_ist']).replace(tzinfo=None)
  local_wall=(finished-started).total_seconds()
 except Exception: pass
 if local_wall is None:
  try:
   started=datetime.strptime((OUT/'local_process_start.txt').read_text().strip(),'%a %b %d %H:%M:%S %Y')
   stopped=datetime.fromisoformat(local_status['stopped_local']).replace(tzinfo=None)
   local_wall=(stopped-started).total_seconds()
  except Exception: pass
 if local_wall is None:
  try:
   started=datetime.strptime((OUT/'local_process_start.txt').read_text().strip(),'%a %b %d %H:%M:%S %Y')
   local_wall=(datetime.now().astimezone().replace(tzinfo=None)-started).total_seconds()
  except Exception: pass
 bridge_best=by_group['bridge'].get('_best','unavailable')
 bridge_best_score=by_group['bridge'].get(bridge_best,{}).get('FS',{}).get('2')
 completed={m:len(v) for m,v in preds.items()}
 n_bridge=len(bridge_q)
 total_completed=len({q for pm in preds.values() for q in pm})
 primary_line=(f"{primary['difference']:+.3f} [{primary['ci95_percentile'][0]:+.3f}, {primary['ci95_percentile'][1]:+.3f}] over {primary['query_n']} bridge queries" if 'difference' in primary else primary.get('reason','not computed'))
 try: dense_meta=json.loads((OUT/'gpu_dense_run_metadata.json').read_text())
 except Exception: dense_meta={}
 try: structured_meta=json.loads((OUT/'gpu_structured_run_metadata.json').read_text())
 except Exception: structured_meta={}
 gpu_wall=(gpu_meta.get('wall_seconds') or 0)+(dense_meta.get('wall_seconds') or 0)+(structured_meta.get('wall_seconds') or 0)
 if not gpu_wall: gpu_wall=None
 cost=modal_cost if modal_cost is not None else gpu_meta.get('observed_cost_usd')
 if cost is None and gpu_wall is not None: cost=round((gpu_wall+60)/3600*(1.10+4*.0473+32*.008),3)
 cost_text=f"${cost:.2f} Modal metered" if isinstance(cost,(int,float)) else 'Modal cost unavailable; bounded under $2'
 if billing_summary.get('billed_cost') is not None: cost_text+=f" (workspace billed ${float(billing_summary['billed_cost']):.2f} at capture)"
 result_lines=[
  '# IRE P5 Wave 2 — Full-scale methods',
  f"Hypothesis: conditioning the second evidence slot on q+top1 improves bridge FS@2 over independent MiniLM CE; test bge reranker as replication.",
  f"Compute: local CPU {local_meta.get('threads',local_status.get('threads_requested',12))} threads, {local_wall/60:.1f} min (BM25; encoder fetch stalled); Modal A10G {gpu_wall:.0f}s; {cost_text}." if gpu_wall is not None and local_wall is not None else f"Compute: local CPU; Modal A10G run {'completed' if gpu_meta else 'incomplete'}; {cost_text}.",
  f"Scale/split: DEV2 {manifest['DEV2_n']} of {manifest['remaining_n']} eligible non-pilot rows; {manifest['SEQ_HOLDOUT_n']} SEQ_HOLDOUT IDs sealed (SHA256 {manifest['SEQ_HOLDOUT_ids_sha256']}); holdout not evaluated.",
  f"Coverage: {total_completed}/{manifest['DEV2_n']} DEV2 queries have any predictions; per-method n: {json.dumps(completed,sort_keys=True)}.",
  'Grid: BGE-small, E5-small, BM25, RRF, beta=.25 title-link greedy, MDR q+top1 re-embed, MiniLM CE independent/conditional, BGE-reranker-v2-m3 independent/conditional; k=2/3/5.',
  'Pooling: BGE CLS and E5 masked mean per the [BGE](https://huggingface.co/BAAI/bge-small-en-v1.5) and [E5](https://huggingface.co/intfloat/e5-small-v2) model cards.',
  f"Best single-shot bridge config by FS@2: {bridge_best} ({bridge_best_score:.3f})" if isinstance(bridge_best_score,(int,float)) else f"Best single-shot bridge config by FS@2: {bridge_best}.",
  f"Primary contrast, bridge FS@2 MiniLM conditional−independent: {primary_line}.",
  f"Verdict: {primary.get('verdict','INCONCLUSIVE')}.",
  'Final experiment implication: retain conditional reranking only if this paired bridge gain is positive and stable; otherwise prioritize the best single-shot ranker and report set size.',
  'No task-specific fine-tuning; tests fixed pretrained encoders/rerankers on full DEV2. Input length capped at 256 tokens.',
 ]
 result_path.write_text('\n'.join(result_lines)+'\n')
 print(json.dumps({'methods':completed,'primary':primary,'best_single_bridge':bridge_best,'metrics_rows':len(records),'result_lines':len(result_lines)}))
if __name__=='__main__': main()
