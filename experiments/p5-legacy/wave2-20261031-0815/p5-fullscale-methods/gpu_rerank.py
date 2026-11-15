from __future__ import annotations
import hashlib, json, time
from pathlib import Path
import modal

OUT=Path(__file__).resolve().parent
DATA=OUT/'dev2.parquet'
REMOTE_DATA='/data/dev2.parquet'
APP_NAME='ire-exp-p5-fullscale-20261031'
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.7.1','transformers==4.53.2','huggingface-hub==0.36.2','pyarrow==19.0.1','numpy==1.26.4')
 .add_local_file(str(DATA),REMOTE_DATA))
app=modal.App(APP_NAME)

@app.function(image=image,gpu='A10G',cpu=4,memory=32768,timeout=1800,startup_timeout=600,retries=0,max_containers=1)
def run_rerank():
 import os, torch, pyarrow.parquet as pq, numpy as np
 from huggingface_hub import snapshot_download
 from transformers import AutoModelForSequenceClassification,AutoTokenizer
 os.environ['HF_HUB_DISABLE_XET']='1'
 device='cuda'; torch.set_num_threads(4)
 raw=pq.read_table(REMOTE_DATA).to_pylist(); rows=[]
 def norm(s): return ' '.join(str(s).split()).casefold()
 for r in raw:
  titles=r['context']['title']; groups=r['context']['sentences']
  paras=[(str(t).strip()+' '+' '.join(str(s).strip() for s in g)).strip() for t,g in zip(titles,groups)]
  gold={norm(x) for x in r['supporting_facts']['title']}; gi={i for i,t in enumerate(titles) if norm(t) in gold}
  assert len(gi)==2 and len(paras)==10
  rows.append({'qid':str(r['id']),'type':str(r['type']),'question':str(r['question']),'paragraphs':paras,'gold':sorted(gi)})
 print(f'GPU DEV2 loaded n={len(rows)}; device={torch.cuda.get_device_name(0)}',flush=True)
 all_out=[]; model_metadata=[]
 for model_id,tag in [('cross-encoder/ms-marco-MiniLM-L6-v2','minilm_ce'),('BAAI/bge-reranker-v2-m3','bge_reranker_v2_m3')]:
  started=time.time()
  snap=Path(snapshot_download(repo_id=model_id,cache_dir='/tmp/hf-cache',allow_patterns=['config.json','model.safetensors','*.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','sentencepiece.bpe.model','spiece.model']))
  tokenizer=AutoTokenizer.from_pretrained(str(snap),local_files_only=True)
  model=AutoModelForSequenceClassification.from_pretrained(str(snap),local_files_only=True,use_safetensors=True,torch_dtype=torch.float16).to(device).eval()
  revision=snap.name
  def score_pairs(pairs):
   scores=[]
   for st in range(0,len(pairs),64):
    batch=pairs[st:st+64]
    z=tokenizer([x[0] for x in batch],[x[1] for x in batch],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
     y=model(**z).logits.float().reshape(-1)
    scores.extend(y.cpu().numpy().tolist())
    if st%4096==0: print(f'{tag} pairs {min(st+64,len(pairs))}/{len(pairs)}',flush=True)
   return scores
  independent_pairs=[(r['question'],p) for r in rows for p in r['paragraphs']]
  ind=score_pairs(independent_pairs)
  independent=[]; cond_pairs=[]
  for qi,r in enumerate(rows):
   s=np.asarray(ind[qi*10:(qi+1)*10],dtype=np.float64)
   order=np.argsort(-s,kind='stable').astype(int).tolist(); independent.append(order)
   top=order[0]
   q_cond=r['question']+' '+r['paragraphs'][top]
   cond_pairs.extend((q_cond,r['paragraphs'][j]) for j in range(10) if j!=top)
  cond_scores=score_pairs(cond_pairs)
  conditional=[]; ptr=0
  for qi,r in enumerate(rows):
   top=independent[qi][0]; chunk=cond_scores[ptr:ptr+9]; ptr+=9
   candidates=[j for j in range(10) if j!=top]
   local=np.argsort(-np.asarray(chunk,dtype=np.float64),kind='stable').tolist()
   conditional.append([top]+[candidates[k] for k in local])
  for method,orders in [(tag+'_independent',independent),(tag+'_conditional',conditional)]:
   for r,order in zip(rows,orders):
    all_out.append({'qid':r['qid'],'type':r['type'],'method':method,'order':order,'gold':r['gold'],'latency_s':None})
  files=[]
  for p in sorted(snap.rglob('*')):
   if p.is_file():
    h=hashlib.sha256()
    with p.open('rb') as f:
     for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
  model_metadata.append({'model':model_id,'revision':revision,'max_length':256,'device':device,'precision':'float16','files':files,'wall_seconds':time.time()-started})
  del model,tokenizer,ind,independent,conditional,cond_scores
  torch.cuda.empty_cache()
 return {'predictions':all_out,'models':model_metadata,'n':len(rows),'wall_seconds':time.time()-_APP_START}

_APP_START=time.time()

@app.local_entrypoint()
def main():
 import json
 out_path=OUT/'gpu_predictions.jsonl'; meta_path=OUT/'gpu_run_metadata.json'
 if out_path.exists() or meta_path.exists(): raise FileExistsError('GPU result exists; refusing overwrite')
 print(f'Launching single bounded A10G run; cap < USD 2 by 10m startup + 30m execution and one container',flush=True)
 result=run_rerank.remote()
 with out_path.open('x') as f:
  for item in result['predictions']: f.write(json.dumps(item,separators=(',',':'))+'\n')
 meta={'app_name':APP_NAME,'gpu':'A10G','gpu_rate_usd_hour':1.10,'timeout_seconds':1800,'startup_timeout_seconds':600,'retries':0,'max_containers':1,'cost_ceiling_usd_estimate':2.0,'n':result['n'],'models':result['models'],'wall_seconds':result['wall_seconds']}
 with meta_path.open('x') as f: f.write(json.dumps(meta,indent=2)+'\n')
 print(json.dumps({'n':result['n'],'methods':[x['model'] for x in result['models']],'wall_seconds':result['wall_seconds']}),flush=True)
