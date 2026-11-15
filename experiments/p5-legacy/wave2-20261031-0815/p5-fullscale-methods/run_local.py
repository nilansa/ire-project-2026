from __future__ import annotations
import hashlib, json, math, os, platform, re, sys, time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pyarrow.parquet as pq

OUT=Path(__file__).resolve().parent
DATA=OUT/'dev2.parquet'
CACHE=OUT/'.hf-cache'
MAX_LEN=256
BATCH=96
THREADS=min(12,os.cpu_count() or 8)

def sha_file(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def norm(s): return ' '.join(str(s).split()).casefold()
def rows_load():
 raw=pq.read_table(DATA).to_pylist(); out=[]
 for r in raw:
  titles=r['context']['title']; groups=r['context']['sentences']
  paras=[(str(t).strip()+' '+' '.join(str(s).strip() for s in g)).strip() for t,g in zip(titles,groups)]
  gold={norm(x) for x in r['supporting_facts']['title']}
  gi={i for i,t in enumerate(titles) if norm(t) in gold}
  assert len(paras)==10 and len(gi)==2
  out.append({'qid':str(r['id']),'type':str(r['type']),'question':str(r['question']),'titles':[str(t) for t in titles],'paragraphs':paras,'gold':sorted(gi)})
 return out
def tokens(s): return re.findall(r'[a-z0-9]+',s.casefold())
def bm25(q,paras):
 docs=[tokens(x) for x in paras]; lens=[len(x) for x in docs]; avg=sum(lens)/len(lens)
 df=Counter(t for d in docs for t in set(d)); scores=np.zeros(len(docs),dtype=np.float64)
 for term in tokens(q):
  n=df.get(term,0)
  if not n: continue
  idf=math.log(1+(len(docs)-n+.5)/(n+.5))
  for i,d in enumerate(docs):
   tf=d.count(term)
   if tf:
    den=tf+1.2*(1-.75+.75*lens[i]/max(avg,1e-12)); scores[i]+=idf*tf*2.2/den
 return np.argsort(-scores,kind='stable').astype(int).tolist()
def exact_link(src,dst_title):
 if not dst_title.strip(): return False
 return re.search(r'(?<!\w)'+re.escape(dst_title.strip().casefold())+r'(?!\w)',src.casefold()) is not None
def save_pred(path,method,rows,orders,latencies=None):
 if path.exists(): raise FileExistsError(f'refuse overwrite: {path}')
 with path.open('x') as f:
  for i,(r,o) in enumerate(zip(rows,orders)):
   f.write(json.dumps({'qid':r['qid'],'type':r['type'],'method':method,'order':[int(x) for x in o],
    'gold':r['gold'],'latency_s':None if latencies is None else float(latencies[i])},separators=(',',':'))+'\n')
def encode_model(model_id,rows,texts,role):
 os.environ.setdefault('HF_HUB_DISABLE_XET','1')
 from huggingface_hub import snapshot_download
 import torch
 from transformers import AutoModel,AutoTokenizer
 torch.set_num_threads(THREADS)
 snap=Path(snapshot_download(repo_id=model_id,cache_dir=str(CACHE),allow_patterns=['config.json','model.safetensors','*.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','sentencepiece.bpe.model','spiece.model']))
 tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True)
 model=AutoModel.from_pretrained(str(snap),local_files_only=True,use_safetensors=True).to('cpu').eval()
 maxlen=MAX_LEN
 def batch_encode(inputs, prefix, desc):
  encs=[]; trunc=0
  for start in range(0,len(inputs),BATCH):
   batch=[prefix+x for x in inputs[start:start+BATCH]]
   raw=tok(batch,add_special_tokens=True,truncation=False,padding=False)['input_ids']
   trunc+=sum(len(a)>maxlen for a in raw)
   z=tok(batch,padding=True,truncation=True,max_length=maxlen,return_tensors='pt')
   with torch.inference_mode():
    h=model(**z).last_hidden_state
    mask=z['attention_mask'].unsqueeze(-1).expand(h.size()).float()
    v=(h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
    v=torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy().astype(np.float32)
   encs.append(v)
   if (start//BATCH)%20==0: print(f'{model_id} {desc}: {min(start+BATCH,len(inputs))}/{len(inputs)}',flush=True)
  return np.concatenate(encs),trunc
 para_vec,pt=batch_encode(texts,'passage: ' if model_id.startswith('intfloat/e5') else '', 'paragraphs')
 qprefix='query: ' if model_id.startswith('intfloat/e5') else 'Represent this sentence for searching relevant passages: '
 qvec,qt=batch_encode([r['question'] for r in rows],qprefix,'queries')
 info={'model':model_id,'revision':snap.name,'max_length':maxlen,'device':'cpu','threads':THREADS,'paragraphs_unique':len(texts),'paragraph_truncations':pt,'query_truncations':qt,'files':[]}
 for p in sorted(snap.rglob('*')):
  if p.is_file(): info['files'].append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha_file(p)})
 del model,tok
 return para_vec,qvec,info

def main():
 assert DATA.exists()
 rows=rows_load(); print(f'DEV2 loaded: {len(rows)}',flush=True)
 # Fixed cutoff leaves time to write and summarize before 09:15 IST.
 cutoff=datetime.now(ZoneInfo('Asia/Kolkata')).replace(hour=9,minute=5,second=0,microsecond=0)
 texts=list(dict.fromkeys(p for r in rows for p in r['paragraphs']))
 look={t:i for i,t in enumerate(texts)}
 if not (OUT/'bm25_predictions.jsonl').exists():
  start=time.perf_counter(); orders=[bm25(r['question'],r['paragraphs']) for r in rows]
  save_pred(OUT/'bm25_predictions.jsonl','bm25',rows,orders,[0.0]*len(rows))
  print(f'BM25 done {len(rows)} in {time.perf_counter()-start:.1f}s',flush=True)
 metadata={'python':sys.version.split()[0],'platform':platform.platform(),'cpu':platform.processor() or platform.machine(),'cpu_count':os.cpu_count(),'threads':THREADS,'input_sha256':sha_file(DATA),'input_n':len(rows),'cutoff_ist':cutoff.isoformat(),'models':{},'failures':[]}
 for model_id,method in [('BAAI/bge-small-en-v1.5','dense_bge'),('intfloat/e5-small-v2','dense_e5')]:
  if datetime.now(ZoneInfo('Asia/Kolkata'))>=cutoff: break
  pred_path=OUT/f'{method}_predictions.jsonl'
  try:
   pv,qv,info=encode_model(model_id,rows,texts,'local')
   ranks=[]; lat=[]; start_all=time.perf_counter()
   for i,r in enumerate(rows):
    tic=time.perf_counter(); ix=np.asarray([look[p] for p in r['paragraphs']],dtype=int)
    scores=pv[ix]@qv[i]; order=np.argsort(-scores,kind='stable').astype(int).tolist(); ranks.append(order); lat.append(time.perf_counter()-tic)
   save_pred(pred_path,method,rows,ranks,lat); metadata['models'][method]=info
   print(f'{method} rankings done {len(rows)}',flush=True)
   # BGE-derived greedy title-link expansion.
   if method=='dense_bge' and not (OUT/'entity_graph_predictions.jsonl').exists():
    graph_orders=[]
    for qi,(r,order) in enumerate(zip(rows,ranks)):
     selected=[order[0]]
     base_scores={j:float(pv[look[r['paragraphs'][j]]]@qv[qi]) for j in order}
     while len(selected)<10:
      cand=[j for j in range(10) if j not in selected]
      scored=[]
      for j in cand:
       linked=any(exact_link(r['paragraphs'][s],r['titles'][j]) for s in selected)
       scored.append((base_scores[j]+(.25 if linked else 0.0),-j,j))
      selected.append(max(scored)[2])
     graph_orders.append(selected)
    save_pred(OUT/'entity_graph_predictions.jsonl','entity_graph_beta025',rows,graph_orders)
    # RRF over BM25, BGE and E5 is completed after E5.
   if method=='dense_bge':
    # One MDR-style query+top1 re-embedding, batched over the full DEV2 set.
    conditioned=[r['question']+' '+r['paragraphs'][ranks[i][0]] for i,r in enumerate(rows)]
    _,cv,ci=encode_model_queries_only(model_id,conditioned)
    mdr=[]
    for i,r in enumerate(rows):
     ix=np.asarray([look[p] for p in r['paragraphs']],dtype=int); scores=pv[ix]@cv[i]
     order=[ranks[i][0]]+[j for j in np.argsort(-scores,kind='stable').astype(int).tolist() if j!=ranks[i][0]]
     mdr.append(order)
    save_pred(OUT/'mdr_bge_predictions.jsonl','mdr_bge_query_top1',rows,mdr)
    metadata['models']['mdr_query_reembed']=ci
  except Exception as e:
   metadata['failures'].append({'model':model_id,'error':f'{type(e).__name__}: {e}'})
   print(f'MODEL FAILED {model_id}: {type(e).__name__}: {e}',flush=True)
 # RRF only when all three base rankings exist.
 files=[OUT/'bm25_predictions.jsonl',OUT/'dense_bge_predictions.jsonl',OUT/'dense_e5_predictions.jsonl']
 if all(p.exists() for p in files) and not (OUT/'hybrid_rrf_predictions.jsonl').exists():
  maps=[]
  for p in files:
   with p.open() as f: maps.append({x['qid']:x for x in map(json.loads,f)})
  orders=[]
  for r in rows:
   score=np.zeros(10,dtype=np.float64)
   for mp in maps:
    for rank,j in enumerate(mp[r['qid']]['order'],1): score[j]+=1/(60+rank)
   orders.append(np.argsort(-score,kind='stable').astype(int).tolist())
  save_pred(OUT/'hybrid_rrf_predictions.jsonl','hybrid_rrf_bge_e5_bm25',rows,orders)
 metadata['finished_at_ist']=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat()
 metadata['complete_base_methods']=[p.name for p in sorted(OUT.glob('*_predictions.jsonl'))]
 p=OUT/'local_run_metadata.json'
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(metadata,indent=2)+'\n')
 print(json.dumps({'methods':metadata['complete_base_methods'],'failures':metadata['failures'],'finished':metadata['finished_at_ist']}),flush=True)

def encode_model_queries_only(model_id,inputs):
 import torch
 from transformers import AutoModel,AutoTokenizer
 from huggingface_hub import snapshot_download
 snap=Path(snapshot_download(repo_id=model_id,cache_dir=str(CACHE),allow_patterns=['config.json','model.safetensors','*.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','sentencepiece.bpe.model','spiece.model']))
 tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True)
 model=AutoModel.from_pretrained(str(snap),local_files_only=True,use_safetensors=True).to('cpu').eval(); torch.set_num_threads(THREADS)
 prefix='query: ' if model_id.startswith('intfloat/e5') else 'Represent this sentence for searching relevant passages: '
 out=[]; trunc=0
 for st in range(0,len(inputs),BATCH):
  b=[prefix+x for x in inputs[st:st+BATCH]]; raw=tok(b,truncation=False,padding=False)['input_ids']; trunc+=sum(len(x)>MAX_LEN for x in raw)
  z=tok(b,padding=True,truncation=True,max_length=MAX_LEN,return_tensors='pt')
  with torch.inference_mode():
   h=model(**z).last_hidden_state; mask=z['attention_mask'].unsqueeze(-1).expand(h.size()).float(); v=(h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
   out.append(torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy().astype(np.float32))
  if st//BATCH%20==0: print(f'MDR query re-embedding {st+BATCH}/{len(inputs)}',flush=True)
 del model,tok
 info={'model':model_id,'revision':snap.name,'device':'cpu','max_length':MAX_LEN,'conditioned_query_truncations':trunc}
 return None,np.concatenate(out),info

if __name__=='__main__': main()
