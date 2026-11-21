"""bert-base-uncased contextual token vectors for queries + pool docs (DISCo-style coverage space). Usage: <split>"""
import os, sys, json, numpy as np
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch; from transformers import AutoTokenizer, AutoModel
sys.path.insert(0, __file__.rsplit("/",1)[0]); from h8common import S, OUT, load_texts
name=sys.argv[1]; pools=json.load(open(f"{OUT}/pools_{name}.json")); dt,qt=load_texts()
tok=AutoTokenizer.from_pretrained("bert-base-uncased"); m=AutoModel.from_pretrained("bert-base-uncased").to("mps").eval()
def toks(texts,maxlen):
    b=tok(texts,padding=True,truncation=True,max_length=maxlen,return_tensors="pt").to("mps")
    with torch.no_grad(): h=m(**b).last_hidden_state
    out=[]
    for j in range(len(texts)):
        ids=b["input_ids"][j]; keep=(b["attention_mask"][j]==1)&(ids!=tok.cls_token_id)&(ids!=tok.sep_token_id)&(ids!=tok.pad_token_id)
        v=torch.nn.functional.normalize(h[j][keep].float(),dim=-1).cpu().numpy().astype(np.float32); out.append(v)
    return out
os.makedirs(f"{OUT}/bert_{name}",exist_ok=True)
for q,p in pools.items():
    qv=toks([qt[q]],64)[0]; dv=[]
    for s in range(0,len(p["dids"]),25): dv+=toks([dt[d] for d in p["dids"][s:s+25]],256)
    off=np.cumsum([0]+[len(v) for v in dv]); np.savez(f"{OUT}/bert_{name}/{q}.npz",q=qv,d=np.concatenate(dv),off=off)
print("BERT_DONE",name,len(pools))
