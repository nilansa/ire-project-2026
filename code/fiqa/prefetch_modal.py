"""CPU-only, private, bounded Modal model prefetch for FiQA TTT-Embed."""
import hashlib
import os
import time
import modal

NAME = "ire-fiqa-tttembed-20261116"
app = modal.App("ire-fiqa-tttembed-model-prefetch-20261116")
volume = modal.Volume.from_name(NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub==0.33.4","hf_xet")

MODEL = {
    "embed": ("Qwen/Qwen3-Embedding-0.6B","97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "rerank": ("Qwen/Qwen3-Reranker-4B","22e683669bc0f0bd69640a1354a6d0aebcfeede5")
}

@app.function(image=image,cpu=4,memory=8192,volumes={"/vol":volume},timeout=2400,max_containers=2,scaledown_window=3)
def prefetch(kind:str):
    from huggingface_hub import snapshot_download
    from pathlib import Path
    repo,rev = MODEL[kind]
    out = f"/vol/models/{kind}"
    t=time.monotonic()
    print("MODEL_DOWNLOAD_STARTED",kind,repo,rev,flush=True)
    p=snapshot_download(repo_id=repo,revision=rev,local_dir=out,token=False,
             allow_patterns=["*.safetensors","*.json","*.txt","*.jinja","*.model","*.md","vocab.*","merges.txt"],
             max_workers=4)
    data=sorted(Path(out).rglob("*.safetensors"))
    if not data:
        raise RuntimeError("Model weight not downloaded: "+repr(repo))
    manifest=[]
    for f in data:
        hh=hashlib.sha256()
        with f.open("rb") as stream:
            for chunk in iter(lambda: stream.read(16*1024*1024),b""):
                hh.update(chunk)
        manifest.append(dict(name=f.name,bytes=f.stat().st_size,sha256=hh.hexdigest()))
    import json
    Path(out,"MODEL_STAGE_MANIFEST.json").write_text(json.dumps(dict(model=repo,revision=rev,weights=manifest),indent=2))
    volume.commit()
    print("MODEL_STAGE_VERIFIED",kind,len(data),"elapsed",round(time.monotonic()-t,2),flush=True)
    return kind

@app.local_entrypoint()
def main():
    for item in prefetch.map(["embed","rerank"],order_outputs=False):
        print("MODEL_READY",item,flush=True)
