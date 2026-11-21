"""H8 teacher (Qwen3-Reranker-4B Eq.7) + DISCo-style coverage on Modal rsai-r. Task ceous-disco-ttt-20261116-0820.
Single L4 container per run. No qrels on volume. Resumable chunked outputs."""
import os, json, time, hashlib
import modal
STAGE=os.environ.get("H8_STAGE","teacher_smoke")
app=modal.App(f"h8-{STAGE.replace('_','-')}-rsai-r-20261116")
vol=modal.Volume.from_name("h8-disco-ttt-rsai-r-20261116",create_if_missing=False)
img=(modal.Image.debian_slim(python_version="3.11")
     .pip_install("torch==2.7.1","transformers==4.53.2","huggingface_hub==0.33.4","numpy<2.3"))
RR=("Qwen/Qwen3-Reranker-4B","22e683669bc0f0bd69640a1354a6d0aebcfeede5"); BERT=("google-bert/bert-base-uncased","86b5e0934494bd15c9632b12f734a8a67f723594")
INST="Given a web search query, retrieve relevant passages that answer the query"
PRE="<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
SUF="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

def _fetch(repo,rev,dst):
    from huggingface_hub import snapshot_download
    if not os.path.exists(f"{dst}/config.json"):
        snapshot_download(repo_id=repo,revision=rev,local_dir=dst,token=False,allow_patterns=["*.safetensors","*.json","*.txt","*.jinja","vocab.*","merges.txt"]); vol.commit()

@app.function(image=img,gpu="L4",cpu=4,memory=32768,volumes={"/vol":vol},timeout=7200,max_containers=1)
def teacher(qids_key:str, out_prefix:str, check:bool=False, bs:int=16):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0=time.time(); _fetch(*RR,"/vol/models/rerank")
    split=json.load(open("/vol/out/SPLIT.json"))
    if qids_key=="rest": keep=set(split["order_first_400"]+split["confirm"])-set(split["smoke"])
    else: keep=set(split[qids_key])
    P=[p for p in json.load(open("/vol/out/pairs_all1000_top50.json")) if p["qid"] in keep]
    tok=AutoTokenizer.from_pretrained("/vol/models/rerank",padding_side="left")
    m=AutoModelForCausalLM.from_pretrained("/vol/models/rerank",torch_dtype=torch.float16).cuda().eval()
    yes,no=tok.convert_tokens_to_ids("yes"),tok.convert_tokens_to_ids("no"); tload=time.time()-t0
    txt=[f"{PRE}<Instruct>: {INST}\n<Query>: {p['query']}\n<Document>: {p['doc']}{SUF}" for p in P]
    def score(idx,b_):
        out={}
        for s in range(0,len(idx),b_):
            ii=idx[s:s+b_]; b=tok([txt[i] for i in ii],padding=True,truncation=True,max_length=8192,return_tensors="pt").to("cuda")
            with torch.no_grad(): lg=m(**b).logits[:,-1,:].float()
            ly,ln=lg[:,yes],lg[:,no]; py=torch.softmax(torch.stack([ln,ly],1),1)[:,1]
            for j,i in enumerate(ii): out[i]=(float(ly[j]),float(ln[j]),float(py[j]))
        return out
    order=sorted(range(len(txt)),key=lambda i:len(txt[i])); ntr=sum(len(x)>8192 for x in tok(txt)["input_ids"])
    t1=time.time(); res=score(order,bs); tscore=time.time()-t1
    rec=dict(pairs=len(P),physical_batches=(len(P)+bs-1)//bs,batch_size=bs,truncated=int(ntr),load_s=round(tload,1),score_s=round(tscore,1),
             pairs_per_s=round(len(P)/tscore,2),gpu=torch.cuda.get_device_name(0),finite=all(all(map(lambda z:z==z,v)) for v in res.values()))
    if check:
        sub=order[::max(1,len(order)//48)][:48]; r1=score(sub,1); r2=score(sub,bs)
        rec["bs1_vs_bs_max_abs_r"]=max(abs(r1[i][2]-res[i][2]) for i in sub); rec["repeat_max_abs_r"]=max(abs(r2[i][2]-res[i][2]) for i in sub)
    fn=f"/vol/out/{out_prefix}.jsonl"
    with open(fn,"w") as f:
        for i in range(len(P)): f.write(json.dumps(dict(qid=P[i]["qid"],did=P[i]["did"],logit_yes=res[i][0],logit_no=res[i][1],r=res[i][2]))+"\n")
    rec["sha256"]=hashlib.sha256(open(fn,"rb").read()).hexdigest(); rec["total_s"]=round(time.time()-t0,1)
    json.dump(rec,open(f"/vol/out/{out_prefix}.manifest.json","w"),indent=1); vol.commit(); return rec

@app.function(image=img,gpu="L4",cpu=4,memory=32768,volumes={"/vol":vol},timeout=5400,max_containers=1)
def coverage(B:int=10):
    """bert-base-uncased fp32 contextual tokens; exact greedy coverage F(S)=sum_q max(0,max_t<q,t>); outputs selections + D0 quantities."""
    import torch, numpy as np
    from transformers import AutoTokenizer, AutoModel
    t0=time.time(); _fetch(*BERT,"/vol/models/bert")
    tok=AutoTokenizer.from_pretrained("/vol/models/bert"); m=AutoModel.from_pretrained("/vol/models/bert",torch_dtype=torch.float32).cuda().eval()
    pairs=json.load(open("/vol/out/pairs_all1000_top50.json")); pools=json.load(open("/vol/out/pools_all1000_top50.json"))
    qtext={p["qid"]:p["query"] for p in pairs}; dtext={p["did"]:p["doc"] for p in pairs}
    sp={tok.cls_token_id,tok.sep_token_id,tok.pad_token_id}
    def toks(texts,maxlen):
        b=tok(texts,padding=True,truncation=True,max_length=maxlen,return_tensors="pt").to("cuda")
        with torch.no_grad(): h=m(**b).last_hidden_state
        keep=b["attention_mask"].bool()
        for s in sp: keep&=(b["input_ids"]!=s)
        return [torch.nn.functional.normalize(h[j][keep[j]],dim=-1) for j in range(len(texts))]
    out={}; ntok=0
    for q,p in pools.items():
        qt=toks([qtext[q]],64)[0]; dv=[]
        for s in range(0,50,25): dv+=toks([dtext[d] for d in p["dids"][s:s+25]],256)
        ntok+=sum(len(x) for x in dv)+len(qt)
        sims=[ (qt@d.T).max(1).values for d in dv ]  # per-doc best sim per query token
        Smat=torch.stack(sims)  # 50 x nq
        def cov(best): return float(torch.clamp(best,min=0).sum())
        standalone=[cov(Smat[i]) for i in range(50)]
        S=[]; best=torch.full((Smat.shape[1],),-1e9,device="cuda"); gains_first=None
        while len(S)<B:
            base=cov(best) if S else 0.0
            g=[(cov(torch.maximum(best,Smat[i]))-base) if i not in S else -1e18 for i in range(50)]
            mx=max(g); i=min(j for j in range(50) if g[j]==mx); S.append(i); best=torch.maximum(best,Smat[i])
            if len(S)==1:
                first=i; c0=cov(Smat[i]); gains_first=[cov(torch.maximum(Smat[i],Smat[j]))-c0 for j in range(50)]
        out[q]=dict(disco_idx=S,disco_dids=[p["dids"][i] for i in S],coverage=cov(best),standalone=standalone,first=first,marg_given_first=gains_first,n_qtok=int(Smat.shape[1]))
    json.dump(out,open("/vol/out/disco_coverage_all1000.json","w"))
    rec=dict(queries=len(out),bert_tokens_encoded=int(ntok),total_s=round(time.time()-t0,1),gpu=torch.cuda.get_device_name(0),
             sha256=hashlib.sha256(open("/vol/out/disco_coverage_all1000.json","rb").read()).hexdigest())
    json.dump(rec,open("/vol/out/disco_coverage_all1000.manifest.json","w"),indent=1); vol.commit(); return rec

@app.local_entrypoint()
def main():
    if STAGE=="teacher_smoke": r=teacher.remote("smoke","teacher_smoke8",True)
    elif STAGE=="teacher_rest": r=teacher.remote("rest","teacher_rest992",False)
    elif STAGE=="coverage": r=coverage.remote()
    print("H8_RESULT",json.dumps(r))
