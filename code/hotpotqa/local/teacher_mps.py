"""Qwen3-Reranker-4B yes/no teacher on MPS (TTT-Embed Eq.7, generic instruction, 8192 limit). Usage: pairs.json out.jsonl"""
import os, sys, json, time, hashlib
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
RR=os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-4B/snapshots/22e683669bc0f0bd69640a1354a6d0aebcfeede5")
INST="Given a web search query, retrieve relevant passages that answer the query"
pre="<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suf="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
P=json.load(open(sys.argv[1])); t0=time.time()
tok=AutoTokenizer.from_pretrained(RR,padding_side="left"); m=AutoModelForCausalLM.from_pretrained(RR,torch_dtype=torch.float16).to("mps").eval()
yes,no=tok.convert_tokens_to_ids("yes"),tok.convert_tokens_to_ids("no"); tload=time.time()-t0
txt=[f"{pre}<Instruct>: {INST}\n<Query>: {p['query']}\n<Document>: {p['doc']}{suf}" for p in P]
order=sorted(range(len(txt)),key=lambda i:len(txt[i])); res=[None]*len(txt); nb=0; ntr=0; BS=10
for s in range(0,len(order),BS):
    idx=order[s:s+BS]; batch=[txt[i] for i in idx]; ntr+=sum(len(x)>8192 for x in tok(batch)["input_ids"])
    b=tok(batch,padding=True,truncation=True,max_length=8192,return_tensors="pt").to("mps")
    with torch.no_grad(): lg=m(**b).logits[:,-1,:].float()
    ly,ln=lg[:,yes],lg[:,no]; py=torch.softmax(torch.stack([ln,ly],1),1)[:,1]
    for j,i in enumerate(idx): res[i]=dict(qid=P[i]["qid"],did=P[i]["did"],logit_yes=float(ly[j]),logit_no=float(ln[j]),r=float(py[j]))
    nb+=1
with open(sys.argv[2],"w") as f:
    for r in res: f.write(json.dumps(r)+"\n")
print("TEACHER_DONE",json.dumps(dict(pairs=len(res),physical_batches=nb,batch_size=BS,truncated=ntr,load_s=round(tload,1),total_s=round(time.time()-t0,1),
  pairs_per_s_excl_load=round(len(res)/(time.time()-t0-tload),2),sha256=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest())),flush=True)
