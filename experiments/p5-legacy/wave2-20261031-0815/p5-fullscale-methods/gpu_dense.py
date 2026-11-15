from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import modal
OUT=Path(__file__).resolve().parent
DATA=OUT/'dev2.parquet'
REMOTE_DATA='/data/dev2.parquet'
APP_NAME='ire-exp-p5-fullscale-dense-20261031'
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.7.1','transformers==4.53.2','huggingface-hub==0.36.2','pyarrow==19.0.1','numpy==1.26.4')
 .add_local_file(str(DATA),REMOTE_DATA))
app=modal.App(APP_NAME)

@app.function(image=image,gpu='A10G',cpu=4,memory=32768,timeout=900,startup_timeout=600,retries=0,max_containers=1)
def run_dense():
 import os,torch,numpy as np,pyarrow.parquet as pq
 from huggingface_hub import snapshot_download
 from transformers import AutoModel,AutoTokenizer
 os.environ['HF_HUB_DISABLE_XET']='1'; torch.set_num_threads(4)
 raw=pq.read_table(REMOTE_DATA).to_pylist(); rows=[]
 def norm(s): return ' '.join(str(s).split()).casefold()
 for r in raw:
  titles=r['context']['title']; groups=r['context']['sentences']
  paras=[(str(t).strip()+' '+' '.join(str(s).strip() for s in g)).strip() for t,g in zip(titles,groups)]
  gold={norm(x) for x in r['supporting_facts']['title']}; gi={i for i,t in enumerate(titles) if norm(t) in gold}
  assert len(gi)==2 and len(paras)==10
  rows.append({'qid':str(r['id']),'type':str(r['type']),'question':str(r['question']),'paragraphs':paras,'gold':sorted(gi)})
 texts=list(dict.fromkeys(p for r in rows for p in r['paragraphs'])); look={t:i for i,t in enumerate(texts)}
 all_predictions=[]; model_info=[]; task_start=time.time()
 for model_id,method in [('BAAI/bge-small-en-v1.5','dense_bge'),('intfloat/e5-small-v2','dense_e5')]:
  started=time.time(); snap=Path(snapshot_download(repo_id=model_id,cache_dir='/tmp/hf-cache',allow_patterns=['config.json','model.safetensors','*.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','sentencepiece.bpe.model','spiece.model']))
  tok=AutoTokenizer.from_pretrained(str(snap),local_files_only=True)
  model=AutoModel.from_pretrained(str(snap),local_files_only=True,use_safetensors=True,torch_dtype=torch.float16).to('cuda').eval()
  tr={'paragraph':0,'query':0}
  def encode(inputs,prefix,role):
   chunks=[]
   for st in range(0,len(inputs),128):
    b=[prefix+x for x in inputs[st:st+128]]
    raw_ids=tok(b,truncation=False,padding=False,add_special_tokens=True)['input_ids']
    tr[role]+=sum(len(x)>256 for x in raw_ids)
    z=tok(b,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
     h=model(**z).last_hidden_state
     if model_id.startswith('BAAI/'):
      v=h[:,0]
     else:
      mask=z['attention_mask'].unsqueeze(-1).expand(h.size()).to(h.dtype)
      v=(h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
     chunks.append(torch.nn.functional.normalize(v,p=2,dim=1).float().cpu().numpy())
    if st%4096==0: print(f'{method} {role} {min(st+128,len(inputs))}/{len(inputs)}',flush=True)
   return np.concatenate(chunks)
  pp='passage: ' if model_id.startswith('intfloat/e5') else ''
  qp='query: ' if model_id.startswith('intfloat/e5') else 'Represent this sentence for searching relevant passages: '
  pv=encode(texts,pp,'paragraph'); qv=encode([r['question'] for r in rows],qp,'query')
  for i,r in enumerate(rows):
   ix=np.asarray([look[p] for p in r['paragraphs']],dtype=int); order=np.argsort(-(pv[ix]@qv[i]),kind='stable').astype(int).tolist()
   all_predictions.append({'qid':r['qid'],'type':r['type'],'method':method,'order':order,'gold':r['gold'],'latency_s':None})
  files=[]
  for p in sorted(snap.rglob('*')):
   if p.is_file():
    h=hashlib.sha256()
    with p.open('rb') as f:
     for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
  model_info.append({'model':model_id,'method':method,'revision':snap.name,'max_length':256,'device':'cuda','precision':'float16','unique_paragraphs':len(texts),'truncations':tr,'files':files,'wall_seconds':time.time()-started})
  del model,tok,pv,qv; torch.cuda.empty_cache()
 return {'predictions':all_predictions,'models':model_info,'n':len(rows),'wall_seconds':time.time()-task_start}

@app.local_entrypoint()
def main():
 out=OUT/'gpu_dense_predictions.jsonl'; meta=OUT/'gpu_dense_run_metadata.json'
 if out.exists() or meta.exists(): raise FileExistsError('refusing to overwrite GPU dense output')
 r=run_dense.remote()
 with out.open('x') as f:
  for x in r['predictions']: f.write(json.dumps(x,separators=(',',':'))+'\n')
 data={'app_name':APP_NAME,'gpu':'A10G','gpu_rate_usd_hour':1.10,'timeout_seconds':900,'startup_timeout_seconds':600,'retries':0,'max_containers':1,'cost_ceiling_usd_estimate':1.25,'n':r['n'],'models':r['models'],'wall_seconds':r['wall_seconds']}
 meta.write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps({'n':r['n'],'wall_seconds':r['wall_seconds'],'models':[x['method'] for x in r['models']]}),flush=True)
