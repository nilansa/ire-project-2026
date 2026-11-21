"""Ceous H8 isolated 30-second Modal GPU permission and compute probe. No model or data downloads."""
import modal
app=modal.App("h8-gpu-access-check-20261116")
@app.function(image=modal.Image.debian_slim(),gpu="L4",timeout=30,cpu=1,memory=512)
def check():
    return {"ok": True}
@app.local_entrypoint()
def main():
    print("GPU_CHECK",check.remote())
