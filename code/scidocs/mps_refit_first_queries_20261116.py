#!/usr/bin/env python3
"""Source-locked ReFIT SCIDOCS real smoke run: first N queries, full 25,657-doc corpus.

Not the paper's full 1,000-query reproduction. Qrels are loaded only AFTER
raw, rerank125, and ReFIT predictions have all been computed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from benchmark import batched_teacher, read_qrels, score, _model_files, digest
from sharded import source, refine

def main():
    ap=argparse.ArgumentParser()
    for x in ("archive","cache","teacher","out"):
        ap.add_argument("--"+x,required=True)
    ap.add_argument("--num-queries",type=int,default=5)
    ap.add_argument("--device",default="mps")
    a=ap.parse_args()
    assert 1<=a.num_queries<=20
    docs,queries=source(a.archive)
    x=torch.load(a.cache,map_location="cpu",weights_only=True)
    docids=sorted(docs)
    assert x["doc_ids"]==docids and x["query_ids"]==[q["_id"] for q in queries]
    device=torch.device(a.device)
    vectors=x["docs"].to(device)
    tok=AutoTokenizer.from_pretrained(a.teacher,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(a.teacher,local_files_only=True).to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    preds={};feedback={};timings={}
    started=time.monotonic()
    for q,q_cpu in zip(queries[:a.num_queries],x["queries"][:a.num_queries]):
        t0=time.monotonic()
        qid=q["_id"]
        qvec=q_cpu.to(device)
        scores=vectors@qvec
        if qid in docs:scores[docids.index(qid)]=-torch.inf
        idx=torch.argsort(scores,descending=True,stable=True)[:125]
        slate=[docids[int(j)] for j in idx.cpu().tolist()]
        def scorer(query,pas):
            enc=tok([query]*len(pas),pas,truncation=True,padding=True,max_length=512,return_tensors="pt").to(device)
            with torch.inference_mode():
                return model(**enc).logits.reshape(-1).cpu().tolist()
        tc=batched_teacher(q["text"],[docs[d] for d in slate],scorer,batch_size=8)
        assert len(tc)==125 and all(isinstance(f,(int,float)) for f in tc)
        newvec=refine(qvec,vectors[idx[:100]].detach(),torch.tensor(tc[:100],device=device))
        rec=vectors@newvec
        if qid in docs:rec[docids.index(qid)]=-torch.inf
        newidx=torch.argsort(rec,descending=True,stable=True)[:100]
        rr=sorted(range(125),key=lambda i:(-tc[i],i))[:100]
        preds[qid]={"base":slate[:100],"rerank125":[slate[i] for i in rr],
                    "refit":[docids[int(j)] for j in newidx.cpu().tolist()]}
        feedback[qid]={"candidate_ids":slate,"reranker_scores":tc}
        timings[qid]=time.monotonic()-t0
        print("SCORED_REAL_QUERY",len(preds),a.num_queries,"elapsed_s",round(timings[qid],2),flush=True)
    output_path=Path(a.out)
    assert not output_path.exists()
    output={"status":"PARTIAL_REAL_FULL_CORPUS_SEARCH_NOT_FINAL_TEST_SET",
            "dataset":"BEIR SCIDOCS","selected_queries":"first_N_in_original_dataset_order",
            "total_dataset_queries":1000,"evaluated_queries":a.num_queries,
            "full_corpus_size":len(docids),
            "retriever_checkpoint_sha256":x["retriever_files"],
            "teacher_checkpoint_sha256":_model_files(a.teacher),
            "frozen_vector_cache_sha256":digest(a.cache),
            "source_script_sha256":digest(__file__),
            "teacher_pairs_physical":len(preds)*125,
            "teacher_pairs_logical_for_refit":len(preds)*100,
            "prediction_per_query":preds,"teacher_feedback_per_query":feedback,
            "latency_s_per_query":timings,
            "elapsed_s":time.monotonic()-started}
    # Relevance judgments are inaccessible to prediction and vector update.
    # Separate assessment loads qrels only after all candidate sets are frozen.
    qrels=read_qrels(a.archive)
    output["qrel_conditional_recall_at_100"]={}
    for name in ("base","rerank125","refit"):
        output["qrel_conditional_recall_at_100"][name]=score(
            {qid:row[name] for qid,row in preds.items()},
            {qid:qrels[qid] for qid in preds},
            100)["mean_recall"]
    output_path.write_text(json.dumps(output,indent=2)+"\n")
    print("REAL_SCIDOCS_REFIT_SMOKE",output["qrel_conditional_recall_at_100"],
          "run_sha256",digest(output_path),"out",str(output_path),flush=True)

if __name__=="__main__":main()
