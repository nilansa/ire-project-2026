from __future__ import annotations
import hashlib,json,re,time
from pathlib import Path
import modal
OUT=Path(__file__).resolve().parent
DATA=OUT/'dev2.parquet'
REMOTE_DATA='/data/dev2.parquet'
APP_NAME='ire-exp-p5-fullscale-structured-20261031'
image=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.7.1','transformers==4.53.2','huggingface-hub==0.36.2','pyarrow==19.0.1','numpy==1.26.4')
 .add_local_file(str(DATA),REMOTE_DATA))
app=modal.App(APP_NAME)

@app.function(image=image,gpu='A10G',cpu=4,memory=32768,timeout=900,startup_timeout=600,retries=0,max_containers=1)
def run_structured():
 import os,torch,numpy as np,pyarrow.parquet as pq
 from huggingface_hub import snapshot_download
 from transformers import AutoModel,AutoTokenizer
 os.environ['HF_HUB_DISABLE_XET']='1'; torch.set_num_threads(4)
 raw=pq.read_table(REMOTE_DATA).to_pylist(); rows=[]
 def norm(s): return ' '.join(str(s).split()).casefold()
 for r in raw:
  titles=[str(t) for t in r['context']['title']]; groups=r['context']['sentences']
  paras=[(t.strip()+' '+' '.join(str(s).strip() for s in g)).strip() for t,g in zip(titles,groups)]
  gold={norm(x) for x in r['supporting_facts']['title']}; gi={i for i,t in enumerate(titles) if norm(t) in gold}
  assert len(gi)==2 and len(paras)==10
  rows.append({'qid':str(r['id']),'type':str(r['type']),'question':str(r['question']),'titles':titles,'paragraphs':paras,'gold':sorted(gi)})
 texts=list(dict.fromkeys(p for r in rows for p in r['paragraphs'])); look={t:i for i,t in enumerate(texts)}
 model_id='BAAI/bge-small-en-v1.5'; started=time.time()
 snap=Path(snapshot_download(repo_id=model_id,cache_dir='/tmp/hf-cache',allow_patterns=['config.json','model.safetensors','*.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt']))
 tokenizer=AutoTokenizer.from_pretrained(str(snap),local_files_only=True)
 model=AutoModel.from_pretrained(str(snap),local_files_only=True,use_safetensors=True,torch_dtype=torch.float16).to('cuda').eval()
 trunc={'paragraph':0,'query':0,'conditioned_query':0}
 def encode(inputs,role):
  vectors=[]
  prefix='Represent this sentence for searching relevant passages: '
  for st in range(0,len(inputs),128):
   b=[prefix+x for x in inputs[st:st+128]]
   raw_ids=tokenizer(b,truncation=False,padding=False,add_special_tokens=True)['input_ids']
   trunc[role]+=sum(len(x)>256 for x in raw_ids)
   batch=tokenizer(b,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
    h=model(**batch).last_hidden_state; v=h[:,0]
    vectors.append(torch.nn.functional.normalize(v,p=2,dim=1).float().cpu().numpy())
   if st%4096==0: print(f'{role} {min(st+128,len(inputs))}/{len(inputs)}',flush=True)
  return np.concatenate(vectors)
 pv=encode(texts,'paragraph'); qv=encode([r['question'] for r in rows],'query')
 graph=[]; mdr=[]; conditioned=[]
 title_patterns=[[re.compile(r'(?<!\w)'+re.escape(t.strip().casefold())+r'(?!\w)') for t in r['titles']] for r in rows]
 for i,r in enumerate(rows):
  ix=np.asarray([look[p] for p in r['paragraphs']],dtype=int); scores=pv[ix]@qv[i]
  dense=np.argsort(-scores,kind='stable').astype(int).tolist()
  selected=[dense[0]]
  while len(selected)<10:
   scored=[]
   for j in range(10):
    if j in selected: continue
    linked=any(title_patterns[i][j].search(r['paragraphs'][s].casefold()) for s in selected)
    scored.append((float(scores[j])+(0.25 if linked else 0.0),-j,j))
   selected.append(max(scored)[2])
  graph.append(selected)
  conditioned.append(r['question']+' '+r['paragraphs'][dense[0]])
 cv=encode(conditioned,'conditioned_query')
 for i,r in enumerate(rows):
  ix=np.asarray([look[p] for p in r['paragraphs']],dtype=int); scores=pv[ix]@cv[i]
  dense=np.argsort(- (pv[ix]@qv[i]),kind='stable').astype(int).tolist(); top=dense[0]
  order=[top]+[j for j in np.argsort(-scores,kind='stable').astype(int).tolist() if j!=top]
  mdr.append(order)
 out=[]
 for method,orders in [('entity_graph_beta025',graph),('mdr_bge_query_top1',mdr)]:
  for r,order in zip(rows,orders): out.append({'qid':r['qid'],'type':r['type'],'method':method,'order':order,'gold':r['gold'],'latency_s':None})
 files=[]
 for p in sorted(snap.rglob('*')):
  if p.is_file():
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(1<<20),b''): h.update(b)
   files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 return {'predictions':out,'n':len(rows),'wall_seconds':time.time()-started,'model':{'id':model_id,'revision':snap.name,'pooling':'CLS','max_length':256,'precision':'float16','truncations':trunc,'files':files}}

@app.local_entrypoint()
def main():
 out=OUT/'gpu_structured_predictions.jsonl'; meta=OUT/'gpu_structured_run_metadata.json'
 if out.exists() or meta.exists(): raise FileExistsError('refusing to overwrite structured GPU output')
 r=run_structured.remote()
 with out.open('x') as f:
  for x in r['predictions']: f.write(json.dumps(x,separators=(',',':'))+'\n')
 data={'app_name':APP_NAME,'gpu':'A10G','gpu_rate_usd_hour':1.10,'timeout_seconds':900,'startup_timeout_seconds':600,'retries':0,'max_containers':1,'cost_ceiling_usd_estimate':1.25,'n':r['n'],'model':r['model'],'wall_seconds':r['wall_seconds']}
 meta.write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps({'n':r['n'],'wall_seconds':r['wall_seconds'],'methods':['entity_graph_beta025','mdr_bge_query_top1']}),flush=True)
