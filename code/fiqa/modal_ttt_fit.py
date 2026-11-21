"""TTT-Embed task-wise KL baseline on FiQA2018, without DISCo selection.
Run only after full Qwen3 teacher scoring passed validation.
Reports (1) paper-style in-sample rewarded test queries and
(2) genuinely held-out transfer from separately rewarded FiQA train queries.
No qrels are used for pair selection, reward scoring or optimization.
"""
from __future__ import annotations
import json,math,hashlib,time,os
from pathlib import Path
import modal

app=modal.App("ire-fiqa-tttembed-baseline-fit-20261116")
volume=modal.Volume.from_name("ire-fiqa-tttembed-20261116",create_if_missing=False)
image=modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.7.1","transformers==4.53.2","huggingface_hub==0.33.4",
    "numpy==1.26.4","safetensors==0.5.3","tqdm==4.67.1")

@app.function(image=image,gpu="H100",cpu=8,memory=32768,
              timeout=1200,scaledown_window=3,max_containers=1,
              volumes={"/vol":volume})
def fit_and_evaluate():
    import torch
    import torch.nn.functional as F
    import numpy as np
    t=time.monotonic()
    output=Path("/vol/out")
    m=json.loads((output/"FINAL_MANIFEST.json").read_text())
    assert m["verified"] and m["pairs_by_split"]=={"test":6480,"train_extra":5000}
    documents=np.load(output/"fiqa_documents_qwen3_0.6b_f32.npy")
    doc_ids=json.loads((output/"fiqa_doc_ids.json").read_text())
    queries=np.load(output/"fiqa_queries_qwen3_0.6b_f32.npy")
    top50=json.loads((output/"fiqa_top50.json").read_text())
    qrels=json.load(open("/vol/input/qrels_test.json"))
    scores=[json.loads(x) for x in open(output/"fiqa_qwen3_reranker_top10_teacher.jsonl")]
    assert documents.shape==(57638,1024) and queries.shape==(1148,1024)
    assert len(doc_ids)==57638 and len(top50)==1148 and len(qrels)==648
    dmap={d:i for i,d in enumerate(doc_ids)}
    reward_by_id={(s["split"],s["query_id"],s["doc_id"]):float(s["teacher_reward"]) for s in scores}
    if len(reward_by_id)!=11480:raise ValueError("Duplicate/missing feedback")
    q_meta={}
    for j,q in enumerate(top50):
        q_meta[(q["split"],q["query_id"])]=(j,q)
    out_results={}
    gpu_documents=torch.from_numpy(documents).cuda()
    test_rows=[x for x in top50 if x["split"]=="test"]
    assert len(test_rows)==648

    def ndcg(rank_list, gold):
        max_rel=sorted([int(v) for v in gold.values()],reverse=True)[:10]
        ideal=sum((2**r-1)/math.log2(i+2) for i,r in enumerate(max_rel))
        if ideal==0:return 0.0
        dcg=sum((2**int(gold.get(d,0))-1)/math.log2(i+2) for i,d in enumerate(rank_list[:10]))
        return dcg/ideal

    def retrieve(qemb):
        q_t=torch.from_numpy(qemb.astype(np.float32)).cuda()
        ids=[]
        for i in range(0,q_t.shape[0],96):
            _,di=torch.topk(q_t[i:i+96]@gpu_documents.T,k=10,dim=1)
            for row in di.cpu().numpy():
                ids.append([doc_ids[int(j)] for j in row])
        return ids

    test_indices=[q_meta[("test",q["query_id"])][0] for q in test_rows]
    x_test=queries[test_indices]
    baseline_lists=retrieve(x_test)
    raw_per_query=[ndcg(d,qrels[str(q["query_id"])]) for d,q in zip(baseline_lists,test_rows)]
    score_sorted=lambda row: sorted(row["candidates"][:10],
        key=lambda c:(-reward_by_id[(row["split"],row["query_id"],c["doc_id"])],c["rank"]))
    direct_lists=[[c["doc_id"] for c in score_sorted(row)] for row in test_rows]
    direct_per_query=[ndcg(d,qrels[str(q["query_id"])]) for d,q in zip(direct_lists,test_rows)]
    out_results["raw_retriever_ndcg10"]=float(np.mean(raw_per_query))
    out_results["direct_teacher_rerank_top10_ndcg10"]=float(np.mean(direct_per_query))

    for scope in ("test","train_extra"):
        rows=[q for q in top50 if q["split"]==scope]
        n=len(rows)
        original_ids=[q_meta[(q["split"],q["query_id"])][0] for q in rows]
        xs=torch.from_numpy(queries[original_ids].copy()).cuda()
        Z=torch.from_numpy(np.stack(
            [documents[[dmap[c["doc_id"]] for c in q["candidates"][:10]]] for q in rows])
            .astype(np.float32)).cuda()
        rewards=torch.tensor([[reward_by_id[(scope,q["query_id"],c["doc_id"])]
                         for c in q["candidates"][:10]] for q in rows],
                         device="cuda",dtype=torch.float32)
        pT=torch.softmax(rewards/0.02,dim=-1)
        vec=torch.zeros(1024,device="cuda",dtype=torch.float32,requires_grad=True)
        opt=torch.optim.Adam([vec],lr=0.01)
        loss_history=[]
        for step in range(300):
            opt.zero_grad(set_to_none=True)
            adapted=F.normalize(xs+vec[None,:],p=2,dim=-1)
            logits=torch.einsum("bd,bkd->bk",adapted,Z)/0.05
            logps=torch.log_softmax(logits,dim=-1)
            kl=torch.sum(pT*(torch.log(pT.clamp_min(1e-12))-logps),dim=-1).mean()
            loss=kl+0.3*vec.pow(2).sum()
            loss.backward()
            opt.step()
            if (step+1)%50==0:loss_history.append(dict(step=step+1,loss=float(loss.detach().cpu()),kl=float(kl.detach().cpu())))
        alpha=n/(n+1)
        corrected=F.normalize(torch.from_numpy(x_test.copy()).cuda()+alpha*vec.detach()[None,:],p=2,dim=-1).cpu().numpy()
        new_lists=retrieve(corrected)
        adapted_scores=[ndcg(d,qrels[str(q["query_id"])]) for d,q in zip(new_lists,test_rows)]
        vfile=output/f"ttt_fiqa_{scope}_v_f32.npy"
        np.save(vfile,vec.detach().cpu().numpy())
        out_results[scope]=dict(reward_queries=n,reward_pairs=n*10,
              alpha=alpha,steps=300,lr=0.01,lambda_l2=0.3,
              teacher_temperature=0.02,student_temperature=0.05,
              ttt_test_ndcg10=float(np.mean(adapted_scores)),
              gain_over_raw=float(np.mean(adapted_scores)-np.mean(raw_per_query)),
              loss_history=loss_history,adapted_vector_path=vfile.name,
              evaluation="IN_SAMPLE_REWARDED_TEST" if scope=="test" else "HELDOUT_TEST_FROM_DISJOINT_TRAIN")
        print("TTT_SCOPE_COMPLETED",scope,json.dumps(out_results[scope]),flush=True)
    manifest=dict(task_id="ire-disco-tttembed-selection-20261116-0700",
         model="Qwen3-Embedding-0.6B",teacher="Qwen3-Reranker-4B",
         eval_queries=648,docs=57638,scope="task-wise FiQA2018",
         optimization="KL(P_T || P_S) + 0.3*||v||^2; 300 Adam steps",
         results=out_results,elapsed_seconds=round(time.monotonic()-t,2),
         evaluated_with_test_qrels_only_after_vector_fit=True)
    (output/"TTT_BASELINE_RESULTS.json").write_text(json.dumps(manifest,indent=2))
    volume.commit()
    print("TTT_BASELINE_RESULTS",json.dumps(manifest,sort_keys=True),flush=True)
    return {k:v for k,v in out_results.items() if k!="test" and k!="train_extra"}

@app.local_entrypoint()
def main():
    print("TTT_FIQA_BASELINE",fit_and_evaluate.remote(),flush=True)
