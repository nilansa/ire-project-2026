"""Explicit encode, shard, merge. No qrels are read by encode or shard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from benchmark import digest, _model_files, _jsonlines, batched_teacher, shard_ids

ARCHIVE_SHA = '96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1'


def atomic_new(path, writer):
    target = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.'+target.name+'.', dir=target.parent)
    os.close(fd)
    try:
        writer(tmp)
        os.link(tmp, target)
    finally:
        os.unlink(tmp)


def atomic_json(path, value):
    def write(p):
        with open(p,'w') as f: json.dump(value,f,allow_nan=False); f.write('\n')
    atomic_new(path,write)


def source(archive):
    if digest(archive) != ARCHIVE_SHA: raise ValueError('unexpected archive')
    with zipfile.ZipFile(archive) as z:
        corpus = list(_jsonlines(z,'scidocs/corpus.jsonl'))
        queries = list(_jsonlines(z,'scidocs/queries.jsonl'))
    docs = {x['_id']:(x.get('title','')+' '+x.get('text','')).strip() for x in corpus}
    if len(docs)!=25657 or len(queries)!=1000 or len({q['_id'] for q in queries})!=1000: raise ValueError('invalid corpus or query IDs')
    return docs, queries


def encoder(model_path, device):
    tok=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModel.from_pretrained(model_path,local_files_only=True).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    def embed(texts):
        result=[]
        with torch.inference_mode():
            for i in range(0,len(texts),8):
                t=tok(texts[i:i+8],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
                x=model(**t).last_hidden_state; mask=t['attention_mask'].unsqueeze(-1)
                result.append(((x*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu())
        return torch.cat(result)
    return embed


def encode(args):
    docs,queries=source(args.archive); ids=sorted(docs); em=encoder(args.retriever,torch.device(args.device))
    start=time.perf_counter(); dv=em([docs[d] for d in ids]); qv=em([q['text'] for q in queries])
    if dv.shape!=(25657,768) or qv.shape!=(1000,768): raise ValueError('unexpected embedding shape')
    value={'doc_ids':ids,'query_ids':[q['_id'] for q in queries], 'docs':dv,'queries':qv,
           'archive_sha256':ARCHIVE_SHA,'retriever_files':_model_files(args.retriever),
           'script_sha256':digest(__file__),'encode_seconds':time.perf_counter()-start}
    atomic_new(args.out,lambda p:torch.save(value,p))


def refine(q, docs, scores):
    if not torch.isfinite(scores).all() or (scores.max()-scores.min()).item()==0: raise ValueError('invalid teacher scores')
    target=torch.softmax(((scores-scores.min())/(scores.max()-scores.min()))/2,dim=0).detach()
    x=q.detach().clone().requires_grad_(True)
    for _ in range(100):
        s=docs@x; span=s.max()-s.min()
        if not torch.isfinite(span) or span.item()==0: raise ValueError('invalid student scores')
        loss=torch.nn.functional.kl_div(torch.log_softmax((s-s.min())/span,dim=0),target,reduction='sum')
        g,=torch.autograd.grad(loss,x); x=(x-0.005*g).detach().requires_grad_(True)
    return x.detach()


def shard(args):
    docs,queries=source(args.archive); data=torch.load(args.cache,map_location='cpu',weights_only=True)
    ids=sorted(docs); qids=[q['_id'] for q in queries]
    if data['archive_sha256']!=ARCHIVE_SHA or ids!=data['doc_ids'] or qids!=data['query_ids']: raise ValueError('cache mapping mismatch')
    if data['retriever_files']!=_model_files(args.retriever): raise ValueError('retriever hash mismatch')
    device=torch.device(args.device); vectors=data['docs'].to(device); selected=set(shard_ids(qids,args.shard,4)); cache_hash=digest(args.cache)
    tok=AutoTokenizer.from_pretrained(args.teacher,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.teacher,local_files_only=True).to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    if getattr(model.config,'sbert_ce_default_activation_function',None) != 'torch.nn.modules.linear.Identity': raise ValueError('unverified activation')
    def batch(q,passages):
        t=tok([q]*len(passages),passages,padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
        with torch.inference_mode(): return model(**t).logits.reshape(-1).cpu().tolist()
    results={}; feedback={}; start=time.perf_counter()
    for query,qcpu in zip(queries,data['queries']):
        qid=query['_id']
        if qid not in selected: continue
        tick=time.perf_counter(); q=qcpu.to(device); sims=vectors@q
        if qid in docs: sims[ids.index(qid)]=-torch.inf
        order=torch.argsort(sims,descending=True,stable=True)[:125]; slate=[ids[i] for i in order.tolist()]
        vals=batched_teacher(query['text'],[docs[d] for d in slate],batch)
        if len(vals)!=125: raise ValueError('teacher alignment')
        feedback[qid]={'docids':slate,'scores':vals}
        adjusted=refine(q,vectors[order[:100]].detach(),torch.tensor(vals[:100],device=device)); sim2=vectors@adjusted
        if qid in docs: sim2[ids.index(qid)]=-torch.inf
        ranking=[ids[i] for i in torch.argsort(sim2,descending=True,stable=True)[:100].tolist()]
        reranked=sorted(range(125),key=lambda i:(-vals[i],i))[:100]
        results[qid]={'base':slate[:100],'rerank125':[slate[i] for i in reranked],'refit':ranking,'elapsed_seconds':time.perf_counter()-tick}
    if set(results)!=selected: raise ValueError('missing shard queries')
    atomic_json(args.out,{'shard':args.shard,'shards':4,'cache_sha256':cache_hash,'archive_sha256':ARCHIVE_SHA,
       'teacher_files':_model_files(args.teacher),'script_sha256':digest(__file__),'device':str(device),
       'elapsed_seconds':time.perf_counter()-start,'logical_pairs_per_query':{'base':0,'rerank125':125,'refit':100},
       'physical_teacher_pairs':len(results)*125,'results':results,'teacher_feedback':feedback})


def merge(args):
    _,queries=source(args.archive); qids={q['_id'] for q in queries}; combined={}; seen=set(); records=[]
    for path in args.inputs:
        with open(path) as f: a=json.load(f)
        if a['shard'] in seen or a['shards']!=4 or set(combined)&set(a['results']): raise ValueError('duplicate shard/query')
        seen.add(a['shard']); combined.update(a['results']); records.append(a)
    for key in ('cache_sha256','archive_sha256','teacher_files','script_sha256'):
        if any(x[key]!=records[0][key] for x in records): raise ValueError('mismatched shard provenance: '+key)
    if seen!={0,1,2,3} or set(combined)!=qids: raise ValueError('incomplete query coverage')
    atomic_json(args.out,{'inputs':{str(p):digest(p) for p in args.inputs},'results':combined,'physical_teacher_pairs':sum(a['physical_teacher_pairs'] for a in records)})


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    e=s.add_parser('encode');e.add_argument('--archive',required=True);e.add_argument('--retriever',required=True);e.add_argument('--out',required=True);e.add_argument('--device',default='cuda:0')
    q=s.add_parser('shard');q.add_argument('--archive',required=True);q.add_argument('--retriever',required=True);q.add_argument('--teacher',required=True);q.add_argument('--cache',required=True);q.add_argument('--shard',type=int,required=True);q.add_argument('--out',required=True);q.add_argument('--device',default='cuda:0')
    m=s.add_parser('merge');m.add_argument('--archive',required=True);m.add_argument('--inputs',nargs=4,required=True);m.add_argument('--out',required=True)
    a=p.parse_args();{'encode':encode,'shard':shard,'merge':merge}[a.command](a)

if __name__=='__main__':main()
