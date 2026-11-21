"""H8 DISCo x TTT-Embed — Modal GPU stages (profile rsai-tris only). Task ceous-disco-ttt-20261116-0820.
Stages: prefetch (CPU) | timing (L4, 256 docs) | encode (L4, full corpus + all queries) | teacher (L4, pair list).
No qrels ever uploaded. Weights frozen."""
import os, json, time, hashlib
import modal
STAGE = os.environ.get("H8_STAGE", "prefetch")
app = modal.App(f"h8-disco-ttt-{STAGE}-20261116")
vol = modal.Volume.from_name("h8-disco-ttt-20261116", create_if_missing=False)
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.7.1", "transformers==4.53.2", "huggingface_hub==0.33.4", "numpy<2.3", "pyarrow==20.0.0"))
EMB = ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
RR = ("Qwen/Qwen3-Reranker-4B", "22e683669bc0f0bd69640a1354a6d0aebcfeede5")
QPROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
INST = "Given a web search query, retrieve relevant passages that answer the query"

@app.function(image=img, cpu=4, memory=8192, volumes={"/vol": vol}, timeout=1500)
def prefetch():
    from huggingface_hub import snapshot_download
    out = {}
    for k, (repo, rev) in {"embed": EMB, "rerank": RR}.items():
        t = time.time()
        snapshot_download(repo_id=repo, revision=rev, local_dir=f"/vol/models/{k}", token=False,
                          allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "vocab.*", "merges.txt"])
        out[k] = round(time.time() - t, 1)
    vol.commit(); return out

def _corpus_texts():
    import pyarrow.parquet as pq
    c = pq.read_table("/vol/data/corpus.parquet").to_pydict()
    ids = [str(x) for x in c["_id"]]
    texts = [(f"{ti} {tx}".strip() if ti else tx) for ti, tx in zip(c["title"], c["text"])]
    return ids, texts

def _embedder():
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("/vol/models/embed", padding_side="left")
    m = AutoModel.from_pretrained("/vol/models/embed", torch_dtype=torch.float16).cuda().eval()
    return tok, m

def _encode(tok, m, texts, bs=128, maxlen=512):
    import torch, numpy as np
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), m.config.hidden_size), dtype=np.float32); ntrunc = 0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; batch = [texts[i] for i in idx]
        full = tok(batch, add_special_tokens=True)["input_ids"]; ntrunc += sum(len(f) > maxlen for f in full)
        b = tok(batch, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = m(**b).last_hidden_state[:, -1].float()
        out[idx] = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
    return out, ntrunc

@app.function(image=img, gpu="L4", memory=32768, volumes={"/vol": vol}, timeout=600)
def timing():
    import torch
    ids, texts = _corpus_texts(); tok, m = _embedder()
    _encode(tok, m, texts[:128]); torch.cuda.synchronize(); t = time.time()
    e, nt = _encode(tok, m, texts[:256 * 8][::8]); torch.cuda.synchronize(); s = time.time() - t
    return dict(docs=256, s=round(s, 3), docs_per_s=round(256 / s, 1), est_full_s=round(len(texts) / (256 / s)),
                gpu=torch.cuda.get_device_name(0), finite=bool((e == e).all()))

@app.function(image=img, gpu="L4", memory=49152, volumes={"/vol": vol}, timeout=3000)
def encode():
    import torch, numpy as np, pyarrow.parquet as pq
    t0 = time.time(); ids, texts = _corpus_texts(); tok, m = _embedder()
    D, nt = _encode(tok, m, texts); t_corpus = time.time() - t0
    q = pq.read_table("/vol/data/queries.parquet").to_pydict()
    qids = [str(x) for x in q["_id"]]; Q, ntq = _encode(tok, m, [QPROMPT + x for x in q["text"]], bs=256)
    os.makedirs("/vol/out", exist_ok=True)
    np.save("/vol/out/corpus_qwen3emb06b_f32.npy", D); np.save("/vol/out/queries_qwen3emb06b_f32.npy", Q)
    json.dump(ids, open("/vol/out/corpus_ids.json", "w")); json.dump(qids, open("/vol/out/query_ids.json", "w"))
    man = dict(corpus_shape=list(D.shape), query_shape=list(Q.shape), corpus_trunc=int(nt), query_trunc=int(ntq),
               finite=bool(np.isfinite(D).all() and np.isfinite(Q).all()),
               norm_minmax=[float(np.linalg.norm(D, axis=1).min()), float(np.linalg.norm(D, axis=1).max())],
               corpus_encode_s=round(t_corpus, 1), total_s=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0),
               sha256={f: hashlib.sha256(open(f"/vol/out/{f}", "rb").read()).hexdigest() for f in
                       ["corpus_qwen3emb06b_f32.npy", "queries_qwen3emb06b_f32.npy", "corpus_ids.json", "query_ids.json"]})
    json.dump(man, open("/vol/out/ENCODE_MANIFEST.json", "w"), indent=1); vol.commit(); return man

@app.function(image=img, gpu="L4", memory=32768, volumes={"/vol": vol}, timeout=1500)
def teacher(pairs_file: str, out_file: str, bs: int = 16):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    t0 = time.time(); P = json.load(open(f"/vol/{pairs_file}"))  # [{qid,did,query,doc}]
    tok = AutoTokenizer.from_pretrained("/vol/models/rerank", padding_side="left")
    m = AutoModelForCausalLM.from_pretrained("/vol/models/rerank", torch_dtype=torch.float16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suf = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    txt = [f"{pre}<Instruct>: {INST}\n<Query>: {p['query']}\n<Document>: {p['doc']}{suf}" for p in P]
    order = sorted(range(len(txt)), key=lambda i: len(txt[i])); res = [None] * len(txt); nb = 0; ntr = 0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; batch = [txt[i] for i in idx]
        ntr += sum(len(x) > 8192 for x in tok(batch)["input_ids"])
        b = tok(batch, padding=True, truncation=True, max_length=8192, return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = m(**b).logits[:, -1, :].float()
        ly, ln = lg[:, yes], lg[:, no]; py = torch.softmax(torch.stack([ln, ly], 1), 1)[:, 1]
        for j, i in enumerate(idx):
            res[i] = dict(qid=P[i]["qid"], did=P[i]["did"], logit_yes=float(ly[j]), logit_no=float(ln[j]), r=float(py[j]))
        nb += 1
    with open(f"/vol/{out_file}", "w") as f:
        for r in res: f.write(json.dumps(r) + "\n")
    vol.commit()
    return dict(pairs=len(res), physical_batches=nb, truncated=ntr, s=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(0),
                sha256=hashlib.sha256(open(f"/vol/{out_file}", "rb").read()).hexdigest())

@app.local_entrypoint()
def main(pairs_file: str = "", out_file: str = ""):
    fn = {"prefetch": prefetch, "timing": timing, "encode": encode}.get(STAGE)
    r = teacher.remote(pairs_file, out_file) if STAGE == "teacher" else fn.remote()
    print("H8_RESULT", json.dumps(r))
