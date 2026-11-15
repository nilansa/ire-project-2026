import json,os,re,time,datetime
from rank_bm25 import BM25Okapi
D=os.path.dirname(os.path.abspath(__file__)); t0=time.time()
log={"pid":os.getpid(),"start":datetime.datetime.now().isoformat()}
docs=[json.loads(l) for l in open(f"{D}/corpus.jsonl")]
tok=lambda s: re.findall(r"\w+",s.lower())
bm=BM25Okapi([tok(d["title"]+" "+d["text"]) for d in docs]); log["index_s"]=round(time.time()-t0,2)
qs=json.load(open(f"{D}/sample_questions.json")); preds={}; lat=[]
import numpy as np
for q in qs:
    s=time.time(); sc=bm.get_scores(tok(q["question"])); top=np.argsort(-sc)[:5]
    preds[q["id"]]=[{"doc_id":docs[i]["doc_id"],"score":float(sc[i]),"words":len((docs[i]["title"]+" "+docs[i]["text"]).split())} for i in top]; lat.append(time.time()-s)
json.dump(preds,open(f"{D}/predictions_top5.json","w"))
log.update(end=datetime.datetime.now().isoformat(),total_s=round(time.time()-t0,2),n_queries=len(qs),n_docs=len(docs),latency_mean_s=float(np.mean(lat)),latency_p95_s=float(np.percentile(lat,95)))
json.dump(log,open(f"{D}/retrieve_log.json","w"),indent=1); print(json.dumps(log))
