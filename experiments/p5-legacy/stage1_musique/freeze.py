import json,hashlib,re,os,sys
D=os.path.dirname(os.path.abspath(__file__)); src=f"{D}/data/musique_ans_v1.0_dev.jsonl"
rows=[json.loads(l) for l in open(src)]
h=lambda s: hashlib.sha1(s.encode()).hexdigest()[:16]
corpus={}
for r in rows:
    for p in r["paragraphs"]:
        did=h(p["title"]+"\n"+p["paragraph_text"]); corpus.setdefault(did,{"doc_id":did,"title":p["title"],"text":p["paragraph_text"]})
elig=[];reasons={"not_2hop":0,"not_answerable":0,"support_ne_2":0,"support_not_distinct":0}
for r in rows:
    if not r["id"].startswith("2hop__"): reasons["not_2hop"]+=1; continue
    if not r.get("answerable",True): reasons["not_answerable"]+=1; continue
    sup=[p for p in r["paragraphs"] if p["is_supporting"]]
    if len(sup)!=2: reasons["support_ne_2"]+=1; continue
    ids={h(p["title"]+"\n"+p["paragraph_text"]) for p in sup}
    if len(ids)!=2: reasons["support_not_distinct"]+=1; continue
    elig.append(r)
key=lambda r: hashlib.sha256(("p5-stage1-20261031:"+r["id"]).encode()).hexdigest()
elig.sort(key=key)
if len(elig)<150: print("BLOCKED eligible",len(elig)); sys.exit(2)
samp=elig[:150]
json.dump([{"id":r["id"],"question":r["question"]} for r in samp],open(f"{D}/sample_questions.json","w"),indent=0)
json.dump({r["id"]:sorted(h(p["title"]+"\n"+p["paragraph_text"]) for p in r["paragraphs"] if p["is_supporting"]) for r in samp},open(f"{D}/gold_HIDDEN.json","w"))
with open(f"{D}/corpus.jsonl","w") as f:
    for did in sorted(corpus): f.write(json.dumps(corpus[did])+"\n")
# exclusion check vs HotpotQA dev questions
import pyarrow.parquet as pq
hp=f"{D}/../pilot/data/hotpot_validation.parquet"; norm=lambda s: " ".join(re.findall(r"\w+",s.lower()))
hq={norm(q) for q in pq.read_table(hp,columns=["question"]).column("question").to_pylist()}
overlap=sum(norm(r["question"]) in hq for r in samp)
info={"rows":len(rows),"eligible":len(elig),"excluded":reasons,"sampled":len(samp),"corpus_docs":len(corpus),"hotpot_question_exact_overlap":overlap,"hotpot_parquet_sha256":hashlib.sha256(open(hp,"rb").read()).hexdigest()}
json.dump(info,open(f"{D}/freeze_info.json","w"),indent=1); print(json.dumps(info))
