"""Bounded, read-only FiQA/Qwen3 H100 sanity: 256 corpus docs, no cached outputs."""
import json,time,modal
app=modal.App("ire-fiqa-qwen3-h100-sanity-20261116")
volume=modal.Volume.from_name("ire-fiqa-tttembed-20261116",create_if_missing=False)
image=modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.7.1","transformers==4.53.2","huggingface_hub==0.33.4","numpy==1.26.4","safetensors==0.5.3","tqdm==4.67.1")
@app.function(image=image,gpu="H100",cpu=8,memory=32768,volumes={"/vol":volume},timeout=600,max_containers=1,scaledown_window=3)
def bench():
 import torch,numpy as np,torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModel
 p="/vol/models/embed"
 with open("/vol/input/corpus.jsonl") as f:
  docs=[json.loads(next(f))['text'] for _ in range(256)]
 print("LOADING_GPU_MODEL",flush=True)
 st=time.monotonic()
 tok=AutoTokenizer.from_pretrained(p,padding_side="left",local_files_only=True)
 model=AutoModel.from_pretrained(p,torch_dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).cuda().eval()
 print("MODEL_LOADED",round(time.monotonic()-st,2),"gpu",torch.cuda.get_device_name(0),flush=True)
 texts=docs[:]
 # Sort only for the benchmark. No truncation.
 texts=sorted(texts,key=len)
 st=time.monotonic()
 done=0
 for j in range(0,len(texts),16):
  chunk=texts[j:j+16]
  tokens=tok(chunk,padding=True,truncation=False,return_tensors="pt")
  if tokens["input_ids"].shape[1]>4096:raise ValueError("unexpected sequence")
  tokens={k:v.cuda() for k,v in tokens.items()}
  with torch.inference_mode():
   vec=F.normalize(model(**tokens).last_hidden_state[:,-1,:].float(),p=2,dim=-1).cpu().numpy()
  assert np.isfinite(vec).all() and vec.shape[1]==1024
  done+=len(chunk)
  if done%64==0: print("SANITY_PROGRESS",done,len(texts),round(time.monotonic()-st,3),flush=True)
 torch.cuda.synchronize()
 elapsed=time.monotonic()-st
 print("SANITY_COMPLETE",dict(documents=done,seconds=round(elapsed,3),docs_per_second=round(done/elapsed,3),gpu=torch.cuda.get_device_name(0),no_truncation=True),flush=True)
 return round(done/elapsed,3)
@app.local_entrypoint()
def main():
 print("H100_BENCH_RESULT_DOCS_PER_SEC",bench.remote(),flush=True)
