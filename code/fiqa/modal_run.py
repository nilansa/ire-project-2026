"""FiQA200: one capped GPU container; qrels are never read here."""
import os, json, time, hashlib
from pathlib import Path
import modal

HERE=Path(__file__).resolve().parent
app=modal.App('fiqa200-complementarity-v1-20261116')
vol=modal.Volume.from_name('ire-fiqa-tttembed-20261116',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.7.1','transformers==4.53.2','huggingface_hub==0.33.4','numpy==1.26.4','safetensors==0.5.3')
       .add_local_file(str(HERE/'selection_arms.py'),'/root/selection_arms.py')
       .add_local_file(str(HERE/'fit_querywise.py'),'/root/fit_querywise.py')
       .add_local_file(str(HERE/'run_guard.py'),'/root/run_guard.py')
       .add_local_file(str(HERE/'query_ids.json'),'/root/query_ids.json')
       .add_local_file(str(HERE/'modal_run.py'),'/root/modal_run.py')
       .add_local_file(str(HERE/'FROZEN_INPUTS.json'),'/root/FROZEN_INPUTS.json'))
DEST=Path('/vol/fiqa200-complementarity-v1/run')
INST='Given a web search query, retrieve relevant passages that answer the query'
PRE=('<|im_start|>system\nJudge whether the Document meets the requirements based on '
     'the Query and the Instruct provided. Note that the answer can only be '
     '"yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUF='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def save(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,sort_keys=True)); os.replace(tmp,path)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

@app.function(image=image,gpu='H100',cpu=4,memory=32768,timeout=1800,
              retries=0,max_containers=1,scaledown_window=3,volumes={'/vol':vol})
def run():
    import sys
    sys.path.insert(0,'/root')
    import numpy as np, torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    from selection_arms import select
    from fit_querywise import fit
    from run_guard import begin_once,validate_selection
    t0=time.monotonic(); DEST.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(Path('/root/FROZEN_INPUTS.json').read_text())
    for name in ('modal_run.py','selection_arms.py','fit_querywise.py','run_guard.py','query_ids.json'):
        if digest('/root/'+name)!=frozen['files'][name]: raise ValueError('frozen code/input mismatch: '+name)
    # Fail closed on a second invocation: the previous GPU time is unknowable if killed.
    begin_once(DEST/'EXECUTION_STARTED.json',{'budget_gpu_seconds':1800,'timestamp_unix':time.time(),'freeze_manifest_sha256':digest('/root/FROZEN_INPUTS.json')}); vol.commit()
    def progress(stage,**kw):
        d={'stage':stage,'elapsed_seconds':round(time.monotonic()-t0,2),'gpu':torch.cuda.get_device_name(0),**kw}
        save(DEST/'PROGRESS.json',d); vol.commit(); print('PROGRESS',json.dumps(d),flush=True)
    def guard():
        if time.monotonic()-t0>1740: raise TimeoutError('stopping before 1800-second container limit')
    hashes={
      '/root/query_ids.json':'5db9e2b464fae5b730e62e83cab1f68f70ac4a17aaedbd175d85e842df6632d8',
      '/vol/input/corpus.jsonl':'f2460adb84a9943775f492ffda829d589548f2f4a5067a7f7608cdaac7215745',
      '/vol/out/fiqa_doc_ids.json':'0768cd2fdb2aa94e751d6a857cd82895607d3dc5c5382fb86133dc14e53f8674',
      '/vol/out/fiqa_documents_qwen3_0.6b_f32.npy':'bcc7c48e2b104090eb5863d3436117f9d77e35ebe395a412b2801e52197c0b37',
      '/vol/out/fiqa_queries_qwen3_0.6b_f32.npy':'a8740a29cbd78aa8f792c712e98cec18dba2339945b082023b8906bcb1f940ae',
      '/vol/out/fiqa_top50.json':'9182c53155fbb144186f5ad1d969f02966f0da01d43d08947c02775d88f8420c',
      '/vol/out/fiqa_qwen3_reranker_top10_teacher.jsonl':'808534d6c2e826fa4ba62f1aed2924530f7592f7a91ed6024cf8fd0e05533555',
      '/vol/models/rerank/tokenizer.json':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
      '/vol/models/rerank/tokenizer_config.json':'253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0',
      '/vol/models/rerank/vocab.json':'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
      '/vol/models/rerank/merges.txt':'8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
      '/vol/models/rerank/config.json':'38bff5eac700032a185745e4076eccad7aa453473cafc2a27de412cdb7b79e19',
      '/vol/models/rerank/special_tokens_map.json':'76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd',
      '/vol/fiqa200-complementarity-v1/models/bert/model.safetensors':'68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3',
      '/vol/fiqa200-complementarity-v1/models/bert/config.json':'7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20',
      '/vol/fiqa200-complementarity-v1/models/bert/tokenizer.json':'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98',
      '/vol/fiqa200-complementarity-v1/models/bert/tokenizer_config.json':'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10',
      '/vol/fiqa200-complementarity-v1/models/bert/vocab.txt':'07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'}
    for path,expected in hashes.items():
        if digest(path)!=expected: raise ValueError('input hash mismatch: '+path)
    rr=json.loads(Path('/vol/models/rerank/MODEL_STAGE_MANIFEST.json').read_text())
    assert rr['model']=='Qwen/Qwen3-Reranker-4B' and rr['revision']=='22e683669bc0f0bd69640a1354a6d0aebcfeede5'
    expected_weights={
      'model-00001-of-00002.safetensors':'cf2e87cbf71fa628961532232e04dd6c19702a0a057f5e2aff95ea1aca4fd488',
      'model-00002-of-00002.safetensors':'78946d22b7f6456ea7a5358dbdf3982de36c5bac1f166a5fd58e18e31db8048a'}
    if {x['name']:x['sha256'] for x in rr['weights']}!=expected_weights or {p.name for p in Path('/vol/models/rerank').glob('*.safetensors')}!=set(expected_weights):
        raise ValueError('unexpected reranker model weight inventory')
    for name,expected in expected_weights.items():
        if digest('/vol/models/rerank/'+name)!=expected: raise ValueError('reranker weight hash mismatch: '+name)
    qids=json.loads(Path('/root/query_ids.json').read_text()); assert len(qids)==len(set(qids))==200
    top=json.loads(Path('/vol/out/fiqa_top50.json').read_text()); rows={str(r['query_id']):(i,r) for i,r in enumerate(top[:648])}
    dids=json.loads(Path('/vol/out/fiqa_doc_ids.json').read_text()); assert len(dids)==len(set(dids))==57638
    dmap={d:i for i,d in enumerate(dids)}
    corpus={str(o['id']):o['text'] for o in (json.loads(line) for line in Path('/vol/input/corpus.jsonl').open())}
    assert len(corpus)==57638 and set(corpus)==set(dids)
    assert all(q in rows for q in qids)
    for q in qids:
        c=rows[q][1]['candidates']; assert len(c)==50 and len({v['doc_id'] for v in c})==50 and all(x['doc_id'] in dmap for x in c)
    # Check the mounted volume itself before running either model.
    xa=np.load('/vol/out/fiqa_documents_qwen3_0.6b_f32.npy',mmap_mode='r')
    qa=np.load('/vol/out/fiqa_queries_qwen3_0.6b_f32.npy',mmap_mode='r')
    if xa.shape!=(57638,1024) or qa.shape!=(1148,1024): raise ValueError('vector shape mismatch')
    for i in range(0,len(xa),2048):
        chunk=xa[i:i+2048]
        if not np.isfinite(chunk).all() or np.max(np.abs(np.linalg.norm(chunk,axis=1)-1))>.01: raise ValueError('invalid document vectors')
    if not np.isfinite(qa).all() or np.max(np.abs(np.linalg.norm(qa,axis=1)-1))>.01: raise ValueError('invalid query vectors')
    for q in qids[:5]:
        exact=xa@qa[rows[q][0]]; ii=np.argsort(-exact,kind='stable')[:50]
        if [dids[k] for k in ii]!=[c['doc_id'] for c in rows[q][1]['candidates']]: raise ValueError('preflight top50 mismatch')
    del xa,qa
    selpath=DEST/'selections.json'
    if selpath.exists(): raise RuntimeError('unexpected old selection checkpoint')
    selections={}
    if len(selections)<200:
        progress('coverage',complete=len(selections))
        bertdir='/vol/fiqa200-complementarity-v1/models/bert'
        assert Path(bertdir,'model.safetensors').is_file()
        tk=AutoTokenizer.from_pretrained(bertdir,local_files_only=True)
        model=AutoModel.from_pretrained(bertdir,local_files_only=True,torch_dtype=torch.float32).cuda().eval()
        specials={tk.cls_token_id,tk.sep_token_id,tk.pad_token_id}
        def tokens(texts,limit):
            b=tk(texts,padding=True,truncation=True,max_length=limit,return_tensors='pt').to('cuda')
            with torch.no_grad(): h=model(**b).last_hidden_state
            keep=b['attention_mask'].bool()
            for s in specials: keep &= b['input_ids']!=s
            result=[F.normalize(h[i][keep[i]],dim=-1) for i in range(len(texts))]
            if any(len(x)==0 for x in result): raise ValueError('empty retained BERT token set')
            return result
        # Qrel-free end-to-end smoke before the full coverage and teacher stages.
        sq=qids[0]; sr=rows[sq][1]; sd=sr['candidates'][0]['doc_id']
        qt=tokens([sr['query_text']],64)[0]; dt=tokens([corpus[sd]],256)[0]
        assert torch.isfinite(qt@dt.T).all()
        st=AutoTokenizer.from_pretrained('/vol/models/rerank',local_files_only=True,padding_side='left')
        sm=AutoModelForCausalLM.from_pretrained('/vol/models/rerank',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        spr=st.encode(PRE,add_special_tokens=False); ssu=st.encode(SUF,add_special_tokens=False)
        body=st.encode(f"<Instruct>: {INST}\n<Query>: {sr['query_text']}\n<Document>: {corpus[sd]}",add_special_tokens=False)
        assert len(spr)+len(body)+len(ssu)<=8192
        batch=st.pad({'input_ids':[spr+body+ssu]},padding=True,return_tensors='pt').to('cuda')
        save(DEST/'teacher_attempt.json',{'kind':'smoke','state':'in_flight','pairs':[[sq,sd]]}); vol.commit()
        with torch.inference_mode(): lg=sm(**batch).logits[:,-1,:].float()
        py=float(torch.softmax(lg[:,[st.convert_tokens_to_ids('yes'),st.convert_tokens_to_ids('no')]],-1)[0,0])
        assert 0<=py<=1
        save(DEST/'teacher_attempt.json',{'kind':'smoke','state':'finished','pairs':[[sq,sd]],'p_yes':py}); vol.commit()
        sx=torch.tensor([[1.,0.,0.]],device='cuda'); sz=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]],device='cuda')
        sv=fit(sx,sz,torch.tensor([[0.,1.]],device='cuda'),steps=30)
        assert torch.isfinite(sv).all() and float(sv[0,1])>0
        smoke_ids=[c['doc_id'] for c in sr['candidates']]; smoke_docs=[]
        for offset in (0,25): smoke_docs.extend(tokens([corpus[d] for d in smoke_ids[offset:offset+25]],256))
        smoke_sims=torch.stack([(qt@d.T).max(1).values for d in smoke_docs]).cpu().numpy()
        smoke_select=select(sq,smoke_ids,smoke_sims)
        assert all(len(set(x))==10 and set(x)<=set(smoke_ids) for x in smoke_select.values())
        old={}
        for line in Path('/vol/out/fiqa_qwen3_reranker_top10_teacher.jsonl').open():
            record=json.loads(line)
            if record['split']=='test' and str(record['query_id'])==sq:
                if record['doc_id'] in old: raise ValueError('duplicate smoke teacher key')
                old[record['doc_id']]=float(record['teacher_reward'])
        dense=smoke_select['dense10']; assert set(dense)<=set(old)
        delta=abs(py-old[sd])
        if delta>.05: raise ValueError('teacher smoke mismatch from cached probability: '+str(delta))
        xx=np.load('/vol/out/fiqa_documents_qwen3_0.6b_f32.npy',mmap_mode='r')
        qq=np.load('/vol/out/fiqa_queries_qwen3_0.6b_f32.npy',mmap_mode='r')
        smoke_x=torch.from_numpy(qq[rows[sq][0]].copy()).cuda().unsqueeze(0)
        smoke_z=torch.from_numpy(np.stack([xx[dmap[d]] for d in dense])).cuda().unsqueeze(0)
        smoke_v=fit(smoke_x,smoke_z,torch.tensor([[old[d] for d in dense]],device='cuda'))
        smoke_docs_gpu=torch.from_numpy(np.array(xx)).cuda()
        with torch.no_grad(): smoke_ix=torch.argsort(F.normalize(smoke_x+.5*smoke_v,dim=-1)@smoke_docs_gpu.T,dim=-1,descending=True,stable=True)[:,:100]
        assert smoke_ix.shape==(1,100) and torch.isfinite(smoke_v).all()
        save(DEST/'SMOKE.json',{'status':'pass','query_id':sq,'qrels_read':False,'candidate_count':50,
                                'selected_per_arm':10,'teacher_diagnostic_pairs':1,'full_corpus_size':len(dids),
                                'ranking_size':100,'teacher_p_yes':py,'cached_p_yes':old[sd],
                                'teacher_absolute_difference':delta,'tolerance':.05,'input_hashes':hashes,
                                'elapsed_seconds':time.monotonic()-t0})
        vol.commit(); guard()
        del smoke_docs_gpu,xx,qq,smoke_z,smoke_x,smoke_v
        del sm,st; torch.cuda.empty_cache()
        progress('smoke_qrel_free_passed',queries=1,bert_q_tokens=len(qt),bert_doc_tokens=len(dt),teacher_p_yes=py,diagnostic_teacher_pairs=1)
        for q in qids:
            if q in selections: continue
            guard(); row=rows[q][1]; ids=[c['doc_id'] for c in row['candidates']]
            qt=tokens([row['query_text']],64)[0]; dv=[]
            for i in (0,25): dv.extend(tokens([corpus[d] for d in ids[i:i+25]],256))
            sims=torch.stack([(qt@d.T).max(1).values for d in dv]).cpu().numpy()
            arms=select(q,ids,sims)
            validate_selection(ids,arms)
            qlen=len(tk(row['query_text'],add_special_tokens=True,truncation=False)['input_ids'])
            dlens=[len(x) for x in tk([corpus[d] for d in ids],add_special_tokens=True,truncation=False)['input_ids']]
            selections[q]={'arms':arms,'query_tokens':int(len(qt)),'document_tokens':sum(len(x) for x in dv),
                           'query_truncated':qlen>64,'documents_truncated':sum(x>256 for x in dlens)}
            save(selpath,selections); vol.commit()
            if len(selections)%20==0: progress('coverage',complete=len(selections))
        del model,tk; torch.cuda.empty_cache()
    assert set(selections)==set(qids)
    names=['dense10','coverage','bert_maxsim','rankmatched_0','rankmatched_1','rankmatched_2','uniform']
    allpairs={(q,d) for q in qids for a in names for d in selections[q]['arms'][a]}
    assert len(allpairs)<=10000 and all(len(selections[q]['arms'][a])==10 and len(set(selections[q]['arms'][a]))==10 for q in qids for a in names)
    existing={}
    for line in Path('/vol/out/fiqa_qwen3_reranker_top10_teacher.jsonl').open():
        r=json.loads(line)
        if r['split']=='test' and (str(r['query_id']),r['doc_id']) in allpairs:
            key=(str(r['query_id']),r['doc_id']); value=float(r['teacher_reward'])
            if key in existing or not 0<=value<=1: raise ValueError('duplicate or invalid cached teacher score')
            existing[key]=value
    scorepath=DEST/'teacher_new.jsonl'; new={}
    if scorepath.exists():
        for line in scorepath.open():
            r=json.loads(line); key=(r['qid'],r['did']); assert key not in new; new[key]=r['p_yes']
    assert set(new)<=allpairs and not (set(existing)&set(new))
    missing=sorted(allpairs-set(existing)-set(new)); progress('teacher',cached=len(existing),new=len(new),remaining=len(missing),union=len(allpairs))
    if missing:
        tk=AutoTokenizer.from_pretrained('/vol/models/rerank',local_files_only=True,padding_side='left')
        model=AutoModelForCausalLM.from_pretrained('/vol/models/rerank',local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        pr=tk.encode(PRE,add_special_tokens=False); su=tk.encode(SUF,add_special_tokens=False)
        yes,no=tk.convert_tokens_to_ids('yes'),tk.convert_tokens_to_ids('no'); assert yes!=no
        texts=[f"<Instruct>: {INST}\n<Query>: {rows[q][1]['query_text']}\n<Document>: {corpus[d]}" for q,d in missing]
        enc=tk(texts,add_special_tokens=False,truncation=False)['input_ids']; lengths=[len(x)+len(pr)+len(su) for x in enc]
        if max(lengths)>8192: raise ValueError('teacher input exceeds baseline context; no truncation permitted')
        order=sorted(range(len(missing)),key=lambda i:lengths[i]); pos=0
        while pos<len(order):
            guard(); first=lengths[order[pos]]; bs=24 if first<=384 else 12 if first<=768 else 6 if first<=1536 else 2
            ii=order[pos:pos+bs]
            while len(ii)>1 and max(lengths[j] for j in ii)>max(512,2*first): ii=ii[:-1]
            batch=tk.pad({'input_ids':[pr+enc[j]+su for j in ii]},padding=True,return_tensors='pt').to('cuda')
            save(DEST/'teacher_attempt.json',{'kind':'full','state':'in_flight','pairs':[missing[j] for j in ii], 'previous_new':len(new)}); vol.commit()
            with torch.inference_mode(): logits=model(**batch).logits[:,-1,:]
            probs=torch.softmax(logits[:,[yes,no]].float(),dim=-1)[:,0].cpu().tolist()
            if any(not 0<=x<=1 for x in probs): raise ValueError('nonfinite teacher probability')
            with scorepath.open('a') as f:
                for j,p in zip(ii,probs):
                    q,d=missing[j]; new[(q,d)]=p; f.write(json.dumps({'qid':q,'did':d,'p_yes':p})+'\n')
                f.flush(); os.fsync(f.fileno())
            pos+=len(ii); vol.commit()
            save(DEST/'teacher_attempt.json',{'kind':'full','state':'finished','pairs':[missing[j] for j in ii], 'new_total':len(new)}); vol.commit()
            if pos//400>(pos-len(ii))//400 or pos==len(order): progress('teacher',cached=len(existing),new=len(new),remaining=len(order)-pos,union=len(allpairs))
        del model,tk; torch.cuda.empty_cache()
    teacher=existing|new; assert set(teacher)==allpairs
    progress('fit',pairs_cached=len(existing),pairs_new=len(new),union=len(allpairs))
    X=np.load('/vol/out/fiqa_documents_qwen3_0.6b_f32.npy',mmap_mode='r')
    Q=np.load('/vol/out/fiqa_queries_qwen3_0.6b_f32.npy',mmap_mode='r')
    assert X.shape==(57638,1024) and Q.shape==(1148,1024) and np.isfinite(X).all() and np.isfinite(Q).all()
    docs=torch.from_numpy(np.array(X)).cuda(); qvec=torch.from_numpy(np.stack([Q[rows[q][0]] for q in qids])).cuda()
    # Stable ties follow the original corpus row order.
    with torch.no_grad(): raw=torch.argsort(qvec@docs.T,dim=-1,descending=True,stable=True)[:,:100].cpu().numpy()
    for i,q in enumerate(qids):
        assert [dids[k] for k in raw[i,:50]]==[c['doc_id'] for c in rows[q][1]['candidates']], 'cached top50 mismatch'
    output={q:{'dense':[dids[k] for k in raw[i]]} for i,q in enumerate(qids)}
    vs=[]
    for a in names:
        guard(); selected=[selections[q]['arms'][a] for q in qids]
        Z=docs[torch.tensor([[dmap[d] for d in s] for s in selected],device='cuda')]
        rewards=torch.tensor([[teacher[(q,d)] for d in s] for q,s in zip(qids,selected)],device='cuda')
        v=fit(qvec,Z,rewards); vs.append(v.cpu().numpy())
        with torch.no_grad(): ix=torch.argsort(F.normalize(qvec+.5*v,dim=-1)@docs.T,dim=-1,descending=True,stable=True)[:,:100].cpu().numpy()
        for i,q in enumerate(qids): output[q][a]=[dids[k] for k in ix[i]]
        np.save(DEST/('vectors_'+a+'.npy'),v.cpu().numpy())
        save(DEST/('rankings_'+a+'.json'),{q:output[q][a] for q in qids}); vol.commit()
        progress('fit',finished_arm=a)
    for q in qids:
        rank={c['doc_id']:c['rank'] for c in rows[q][1]['candidates']}
        for a in ('dense10','coverage'):
            output[q]['rerank_'+a]=sorted(selections[q]['arms'][a],key=lambda d:(-teacher[(q,d)],rank[d],d))
    np.save(DEST/'vectors.npy',np.stack(vs)); save(DEST/'rankings.json',output)
    outputs={name:digest(DEST/name) for name in ('selections.json','rankings.json','vectors.npy','SMOKE.json')}
    if scorepath.exists(): outputs['teacher_new.jsonl']=digest(scorepath)
    save(DEST/'run_manifest.json',{'qids':qids,'arms':names,'physical_new_teacher_pairs':len(new),'diagnostic_teacher_pairs':1,'cached_teacher_pairs':len(existing),'union_pairs':len(allpairs),'gpu':torch.cuda.get_device_name(0),'elapsed_seconds':time.monotonic()-t0,'qrels_read':False,'outputs_sha256':outputs,'freeze_manifest_sha256':digest('/root/FROZEN_INPUTS.json')})
    progress('gpu_complete',queries=200); return {'stage':'gpu_complete','pairs_new':len(new),'elapsed_seconds':round(time.monotonic()-t0,2)}

@app.local_entrypoint()
def main():
    print('RESULT',json.dumps(run.remote()),flush=True)
