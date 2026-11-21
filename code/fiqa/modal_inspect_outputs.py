"""Read-only remote byte inventory after GPU completion; does not read qrels."""
import modal,json,hashlib,tarfile
from pathlib import Path

app=modal.App('fiqa200-output-inventory-20261116')
vol=modal.Volume.from_name('ire-fiqa-tttembed-20261116',create_if_missing=False)

@app.function(image=modal.Image.debian_slim(python_version='3.11'),cpu=1,memory=1024,timeout=120,max_containers=1,retries=0,volumes={'/vol':vol})
def inventory():
    root=Path('/vol/fiqa200-complementarity-v1/run')
    result=[]
    for p in sorted(root.iterdir()):
        if not p.is_file(): continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
        result.append({'name':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    bundle=root.parent/'gpu-output-v1.tar'
    if bundle.exists(): raise RuntimeError('output bundle already exists; reconcile before retry')
    with tarfile.open(bundle,'x') as t:
        for entry in result: t.add(root/entry['name'],arcname=entry['name'],recursive=False)
    vol.commit()
    h=hashlib.sha256()
    with bundle.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return {'files':result,'bundle':{'name':bundle.name,'bytes':bundle.stat().st_size,'sha256':h.hexdigest()}}

@app.local_entrypoint()
def main(): print('OUTPUT_INVENTORY_JSON',json.dumps(inventory.remote(),sort_keys=True))
