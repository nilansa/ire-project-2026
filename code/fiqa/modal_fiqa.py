"""Paper-aligned FiQA2018 embeddings and Qwen3 reward-score preparation.
This is the original TTT-Embed top-10 candidate baseline, not DISCo.
test query feedback is IN-SAMPLE as in the main paper; extra train is separate.
"""
from __future__ import annotations
import hashlib, json, os, random, time
from pathlib import Path
import modal

TASK="ire-disco-tttembed-selection-20261116-0700"
NAME="ire-fiqa-tttembed-20261116"
EMB="Qwen/Qwen3-Embedding-0.6B"
EMB_REV="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
RR="Qwen/Qwen3-Reranker-4B"
RR_REV="22e683669bc0f0bd69640a1354a6d0aebcfeede5"
INSTRUCT="Given a web search query, retrieve relevant passages that answer the query"
N=57638
app=modal.App("ire-fiqa-tttembed-pipeline-20261116")
vol=modal.Volume.from_name(NAME,create_if_missing=False)
image=modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.7.1","transformers==4.53.2","huggingface_hub==0.33.4",
    "numpy==1.26.4","safetensors==0.5.3","tqdm==4.67.1")
mount={"/vol":vol}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def write_json(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=str(path)+".tmp"
    with open(tmp,"w") as f: json.dump(obj,f,sort_keys=True,indent=2)
    os.replace(tmp,path)

def corpus():
    return [json.loads(x) for x in open("/vol/input/corpus.jsonl")]

def progress(stage,worker,done,total,t0,**extra):
    obj=dict(task_id=TASK,stage=stage,worker=worker,done=int(done),total=int(total),
             elapsed_seconds=round(time.monotonic()-t0,2),
             updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),**extra)
    write_json(f"/vol/progress/{stage}_{worker}.json",obj)
    vol.commit()
    print("TQDM_PROGRESS",json.dumps(obj,sort_keys=True),flush=True)

def load_embedding():
    import torch
    from transformers import AutoTokenizer,AutoModel
    p="/vol/models/embed"
    assert Path(p,"MODEL_STAGE_MANIFEST.json").is_file(),"model not prefetched"
    tk=AutoTokenizer.from_pretrained(p,local_files_only=True,padding_side="left")
    model=AutoModel.from_pretrained(p,local_files_only=True,torch_dtype=torch.bfloat16,
              attn_implementation="sdpa").cuda().eval()
    torch.set_num_threads(4)
    return tk,model

def embed_texts(texts,tk,model,callback=None):
    import torch, torch.nn.functional as F, numpy as np
    enc=tk(texts,add_special_tokens=True,padding=False,truncation=False)["input_ids"]
    if not enc: return np.empty((0,1024),dtype=np.float32)
    mx=max(map(len,enc))
    if mx>4096: raise ValueError(f"would truncate text with {mx} tokens")
    out=np.empty((len(enc),1024),dtype=np.float32)
    order=sorted(range(len(enc)),key=lambda i:len(enc[i]))
    seen=0
    while seen<len(order):
        n0=len(enc[order[seen]])
        bs=32 if n0<=384 else 16 if n0<=768 else 8 if n0<=1536 else 2
        ids=order[seen:seen+bs]
        while len(ids)>1 and max(len(enc[x]) for x in ids)>max(512,2*n0):
            ids=ids[:-1]
        batch=tk.pad({"input_ids":[enc[i] for i in ids]},padding=True,return_tensors="pt")
        if "attention_mask" not in batch:raise RuntimeError("missing padding attention mask")
        with torch.inference_mode():
            hidden=model(input_ids=batch["input_ids"].cuda(),
                         attention_mask=batch["attention_mask"].cuda()).last_hidden_state[:,-1,:]
            vec=F.normalize(hidden.float(),p=2,dim=-1).cpu().numpy()
        out[ids]=vec
        seen+=len(ids)
        if callback and (seen//768>(seen-len(ids))//768 or seen==len(order)):
            callback(seen,len(order))
    return out

@app.function(image=image,gpu="H100",cpu=8,memory=32768,timeout=3600,
              max_containers=4,scaledown_window=3,volumes=mount)
def encode(part:int):
    import numpy as np, torch
    t0=time.monotonic()
    rows=corpus()
    assert len(rows)==N
    lo=N*part//4; hi=N*(part+1)//4
    outpath=Path(f"/vol/out/encode_shard_{part}.npz")
    meta=outpath.with_suffix(".json")
    if outpath.is_file() and meta.is_file():
        info=json.loads(meta.read_text())
        if info.get("sha256")==sha(outpath) and info.get("lo")==lo and info.get("hi")==hi:
            print("SHARD_REUSED",part,flush=True);return info
    progress("encode",part,0,hi-lo,t0,gpu=torch.cuda.get_device_name(0))
    tk,model=load_embedding()
    vec=embed_texts([r["text"] for r in rows[lo:hi]],tk,model,
                    callback=lambda done,total:progress("encode",part,done,total,t0))
    if vec.shape!=(hi-lo,1024) or not np.isfinite(vec).all() or \
       np.max(np.abs(np.linalg.norm(vec,axis=1)-1))>.01:
        raise RuntimeError("invalid dense embedding shard")
    outpath.parent.mkdir(parents=True,exist_ok=True)
    np.savez(outpath,indices=np.arange(lo,hi,dtype=np.int64),
             ids=np.array([r["id"] for r in rows[lo:hi]]),vec=vec)
    info=dict(part=part,lo=lo,hi=hi,rows=hi-lo,bytes=outpath.stat().st_size,
              sha256=sha(outpath),model=EMB,revision=EMB_REV,
              elapsed_seconds=round(time.monotonic()-t0,2))
    write_json(meta,info);vol.commit()
    progress("encode",part,hi-lo,hi-lo,t0,complete=True,sha256=info["sha256"])
    print("EMBED_PART_COMPLETED",json.dumps(info),flush=True)
    return info

@app.function(image=image,gpu="H100",cpu=8,memory=32768,timeout=2400,
              max_containers=1,scaledown_window=3,volumes=mount)
def merge_and_retrieve(train_count:int=500):
    import numpy as np, torch
    t0=time.monotonic()
    rows=corpus();assert len(rows)==N
    doc_ids=[r["id"] for r in rows]
    if len(set(doc_ids))!=N:raise RuntimeError("duplicated document IDs")
    parts=[]
    for part in range(4):
        p=Path(f"/vol/out/encode_shard_{part}.npz")
        info=json.loads(p.with_suffix(".json").read_text())
        if sha(p)!=info["sha256"]:raise RuntimeError("corrupt shard "+str(part))
        with np.load(p) as ds:
            lo=N*part//4;hi=N*(part+1)//4
            if not np.array_equal(ds["indices"],np.arange(lo,hi)):raise RuntimeError("row order mismatch")
            if list(ds["ids"])!=doc_ids[lo:hi]:raise RuntimeError("doc-id mismatch")
            parts.append(ds["vec"])
    vectors=np.concatenate(parts,axis=0).astype(np.float32)
    if vectors.shape!=(N,1024) or not np.isfinite(vectors).all():raise RuntimeError("bad merged embeddings")
    out=Path("/vol/out")
    docpath=out/"fiqa_documents_qwen3_0.6b_f32.npy"
    np.save(docpath,vectors)
    write_json(out/"fiqa_doc_ids.json",doc_ids)
    qs_test=json.load(open("/vol/input/queries_test.json"))
    qs_train=json.load(open("/vol/input/queries_train.json"))
    assert len(qs_test)==648 and len(qs_train)==5500
    rng=random.Random(1006)
    qs=([dict(split="test",**x) for x in qs_test]+
        [dict(split="train_extra",**x) for x in rng.sample(qs_train,train_count)])
    progress("merge",0,0,len(qs),t0)
    tk,model=load_embedding()
    prompts=[f"Instruct: {INSTRUCT}\nQuery:{x['text']}" for x in qs]
    qvec=embed_texts(prompts,tk,model,
                     callback=lambda d,n:progress("merge",0,d,n,t0))
    np.save(out/"fiqa_queries_qwen3_0.6b_f32.npy",qvec)
    doc_gpu=torch.from_numpy(vectors).cuda()
    q_gpu=torch.from_numpy(qvec).cuda()
    top=[]
    for i in range(0,len(qs),96):
        cos=q_gpu[i:i+96] @ doc_gpu.T
        vals,ix=torch.topk(cos,k=50,dim=-1,largest=True,sorted=True)
        for q,ids,ss in zip(qs[i:i+96],ix.cpu().numpy(),vals.cpu().numpy()):
            top.append(dict(split=q["split"],query_id=q["id"],query_text=q["text"],
                candidates=[dict(doc_id=doc_ids[int(idx)],rank=j+1,cosine=float(score))
                            for j,(idx,score) in enumerate(zip(ids,ss))]))
    write_json(out/"fiqa_top50.json",top)
    manifest=dict(task_id=TASK,corpus_sha256=sha("/vol/input/corpus.jsonl"),
        n_docs=N,dim=1024,dtype="float32",model=EMB,model_revision=EMB_REV,
        query_instruction=INSTRUCT,document_token_cap=4096,
        n_test=648,n_extra_train=train_count,topk=50,feedback_depth=10,
        doc_matrix_sha256=sha(docpath),doc_ids_sha256=sha(out/"fiqa_doc_ids.json"),
        query_matrix_sha256=sha(out/"fiqa_queries_qwen3_0.6b_f32.npy"),
        top50_sha256=sha(out/"fiqa_top50.json"),paper_test_feedback_is_in_sample=True)
    write_json(out/"embedding_manifest.json",manifest);vol.commit()
    progress("merge",0,len(qs),len(qs),t0,complete=True)
    print("FULL_CORPUS_RETRIEVAL_DONE",json.dumps(manifest,sort_keys=True),flush=True)
    return manifest

def load_reranker():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    p="/vol/models/rerank"
    assert Path(p,"MODEL_STAGE_MANIFEST.json").is_file(),"reranker not prefetched"
    tk=AutoTokenizer.from_pretrained(p,local_files_only=True,padding_side="left")
    m=AutoModelForCausalLM.from_pretrained(p,local_files_only=True,
          torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    return tk,m

@app.function(image=image,gpu="H100",cpu=8,memory=32768,timeout=3600,
              max_containers=4,scaledown_window=3,volumes=mount)
def score(part:int):
    import numpy as np, torch
    t0=time.monotonic()
    qs=json.load(open("/vol/out/fiqa_top50.json"))
    lo=len(qs)*part//4; hi=len(qs)*(part+1)//4
    qs=qs[lo:hi]
    meta=Path(f"/vol/out/score_part_{part}.json")
    scored=Path(f"/vol/out/score_part_{part}.jsonl")
    if scored.is_file() and meta.is_file():
        info=json.loads(meta.read_text())
        if info.get("sha256")==sha(scored) and info.get("query_count")==len(qs):
            print("SCORES_REUSED",part,flush=True);return info
    docs={r["id"]:r["text"] for r in corpus()}
    tk,model=load_reranker()
    prefix=('<|im_start|>system\nJudge whether the Document meets the requirements based on '
           'the Query and the Instruct provided. Note that the answer can only be '
           '"yes" or "no".<|im_end|>\n<|im_start|>user\n')
    suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    pr=tk.encode(prefix,add_special_tokens=False)
    su=tk.encode(suffix,add_special_tokens=False)
    yes,no=tk.convert_tokens_to_ids("yes"),tk.convert_tokens_to_ids("no")
    if yes==no:raise RuntimeError("Yes and No tokens collide")
    rows=[];txt=[]
    for q in qs:
        for c in q["candidates"][:10]:
            rows.append(dict(split=q["split"],query_id=q["query_id"],
                doc_id=c["doc_id"],base_rank=c["rank"],base_cosine=c["cosine"]))
            txt.append(f"<Instruct>: {INSTRUCT}\n<Query>: {q['query_text']}\n<Document>: {docs[c['doc_id']]}")
    enc=tk(txt,add_special_tokens=False,truncation=False)["input_ids"]
    lens=[len(z)+len(pr)+len(su) for z in enc]
    if max(lens)>8192:raise RuntimeError("Too many reranker tokens")
    order=sorted(range(len(rows)),key=lambda i:lens[i])
    rewards=np.empty(len(rows),np.float32)
    progress("score",part,0,len(rows),t0)
    n=0
    while n<len(order):
        first=lens[order[n]]
        bs=24 if first<=384 else 12 if first<=768 else 6 if first<=1536 else 2
        ix=order[n:n+bs]
        while len(ix)>1 and max(lens[j] for j in ix)>max(512,2*first):
            ix=ix[:-1]
        b=tk.pad({"input_ids":[pr+enc[j]+su for j in ix]},padding=True,return_tensors="pt")
        with torch.inference_mode():
            ll=model(input_ids=b["input_ids"].cuda(),attention_mask=b["attention_mask"].cuda()).logits[:,-1,:]
            probs=torch.softmax(ll[:,[yes,no]].float(),dim=-1)[:,0].cpu().numpy()
        rewards[ix]=probs
        n+=len(ix)
        if n//240>(n-len(ix))//240 or n==len(order):
            progress("score",part,n,len(rows),t0)
    if not np.isfinite(rewards).all() or (rewards<0).any() or (rewards>1).any():
        raise RuntimeError("Invalid Yes/No scores")
    with open(scored,"w") as f:
        for x,y in zip(rows,rewards):
            f.write(json.dumps(dict(x,teacher_reward=float(y)))+"\n")
    info=dict(part=part,query_count=len(qs),pair_count=len(rows),
        model=RR,revision=RR_REV,score_semantics="P(yes|query,document)",
        sha256=sha(scored),bytes=scored.stat().st_size)
    write_json(meta,info);vol.commit()
    progress("score",part,len(rows),len(rows),t0,complete=True)
    print("TEACHER_PART_COMPLETED",json.dumps(info),flush=True)
    return info

@app.function(image=image,cpu=4,memory=8192,timeout=900,volumes=mount)
def validate():
    import numpy as np
    out=Path("/vol/out")
    em=json.loads((out/"embedding_manifest.json").read_text())
    embp=out/"fiqa_documents_qwen3_0.6b_f32.npy"
    m=np.load(embp,mmap_mode="r")
    if m.shape!=(N,1024) or sha(embp)!=em["doc_matrix_sha256"]:
        raise RuntimeError("Merged embedding does not match manifest")
    allr=[]
    for part in range(4):
        p=out/f"score_part_{part}.jsonl"
        info=json.loads(p.with_suffix(".json").read_text())
        if sha(p)!=info["sha256"]:raise RuntimeError("Corrupt reranker shard")
        allr.extend(json.loads(line) for line in open(p))
    counts={}
    keys=set()
    for r in allr:
        counts[r["split"]]=counts.get(r["split"],0)+1
        k=(r["split"],r["query_id"],r["doc_id"])
        if k in keys:raise RuntimeError("Duplicate q-doc teacher score")
        keys.add(k)
    expected={"test":6480,"train_extra":em["n_extra_train"]*10}
    if counts!=expected:raise RuntimeError(f"Pair counts {counts} != expected {expected}")
    allr.sort(key=lambda r:(r["split"],r["query_id"],r["base_rank"]))
    merged=out/"fiqa_qwen3_reranker_top10_teacher.jsonl"
    with open(merged,"w") as f:
        for r in allr: f.write(json.dumps(r)+"\n")
    final=dict(task_id=TASK,embedding_sha256=em["doc_matrix_sha256"],
        teacher_sha256=sha(merged),teacher_pairs=len(allr),
        pairs_by_split=counts,verified=True,teacher_model=RR,teacher_revision=RR_REV,
        baseline="TTT-Embed depth10 independent dense top10",
        scientific_caveat="Test-feedback 648 queries in-sample to paper's main evaluation; separate train feedback is heldout.")
    write_json(out/"FINAL_MANIFEST.json",final)
    vol.commit()
    print("FINAL_VALIDATED",json.dumps(final,sort_keys=True),flush=True)
    return final

@app.local_entrypoint()
def main(stage:str="all"):
    if stage in ("all","embed"):
        for result in encode.map(range(4),order_outputs=False):
            print("ENCODED_SHARD",result,flush=True)
    if stage in ("all","merge"):
        print("MERGED",merge_and_retrieve.remote(500),flush=True)
    if stage in ("all","score"):
        for result in score.map(range(4),order_outputs=False):
            print("RERANKED_SHARD",result,flush=True)
    if stage in ("all","validate"):
        print("COMPLETE",validate.remote(),flush=True)
    print("STAGE_COMPLETE",stage,flush=True)
