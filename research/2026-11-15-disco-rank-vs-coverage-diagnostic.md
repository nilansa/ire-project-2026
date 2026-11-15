# Ceous / DISCo — complementary evidence, original ranks, and reranker-score alignment

**Date:** 2026-11-15  
**Coordination purpose:** Shared, publicly readable note for Claude Science, Codex, and the IRE final report.  
**Status:** `COMPLETED — exploratory read-only retrospective audit`; no new retrieval, teacher-scoring, training, or official DISCo benchmark started. No ongoing experiment or reserved owner is implied by this note.  
**Related live tracker:** [RESEARCH_STATUS.md](../RESEARCH_STATUS.md) (Codex remains its single writer).  
**Related literature note:** [DISCo paper analysis](literature/ire-retrieval-literature-20261115-01/notes/DISCo_detailed.md).

## The question and answer

The question is whether DISCo can promote complementary evidence that is very low under an ordinary retriever, and whether that mechanism can reconcile disagreements between an initial retriever and an expensive cross-encoder/reward-model reranker.

**Paper-supported answer:** DISCo is a separate first-stage *collective subset retriever*. It does not explicitly apply an earlier retriever's rank as a penalty; instead, its per-round relevance is **marginal contextual-token coverage given previously selected items**. A complementary passage at ordinary rank 60 can be selected early *if its representations are reachable by DISCo's own ANN index and survive approximation/pruning*. An ordinary top-20-only rerank cannot discover that passage. DISCo does **not** calibrate to teacher/reranker scores or train the first-stage retriever to match the teacher.

**Sources:** [ICLR 2026 paper](https://openreview.net/pdf?id=cUdODCFjUM), Sections 2–4, Eqs. 1–8, Algorithms 1–4, Figure 1; [authors' code](https://github.com/structlearning/DISCo).

## Exact transformations — ST100 toy

Query: “Which scientist born in Clamart edited Algorithmica?” A contextual encoder produces one dense vector **per query/document token** (not literal n-gram coverage), e.g., two imaginary orthogonal query-aspect vectors `q_identity=(1,0)`, `q_birthplace=(0,1)`.

Take the following **made-up** per-aspect token max similarities; these are *not* experimental scores.

| Corpus item | identity | birthplace | standalone sum |
|---|---:|---:|---:|
| A, repeated high-rank evidence | 0.95 | 0.30 | 1.25 |
| B, a near duplicate of A | 0.93 | 0.25 | 1.18 |
| D, complementary evidence | 0.10 | 0.90 | 1.00 |

If D is originally 60th because 58 other A-like duplicates outrank it, ordinary top-2 chooses A-like items. DISCo first selects A, recording best coverage `[0.95,0.30]`. A second A-like item has gain `max(0,.93-.95)+max(0,.25-.30)=0`; D has gain `max(0,.10-.95)+max(0,.90-.30)=.60`. D wins the second coverage step.

Formal objective: `F(S,Q)=sum_{q in Q} max_{x in union_{c in S} X_c} <q,x>`. Per-candidate gain is `Delta(c|S,Q)=sum_q max_{x in X_c} [<q,x>-F(S,{q})]_+`. The `max` over the document **union**, rather than summing standalone document scores, prevents duplicate evidence from earning duplicate credit.

**Lift:** `q_hat_S=[q;F(S,{q})]` and `x_hat=[x;-1]`. For the birthplace aspect, `q=(0,1)`, `F(S,{q})=.30`, and D's toy token `x=(.10,.90)`: `q_hat=(0,1,.30)`, `x_hat=(.10,.90,-1)`, giving `<q_hat,x_hat>=.90-.30=.60`; a duplicate `x=(.93,.25)` gives `-.05`, clipped to zero by `[z]_+=max(0,z)`.

**Random feature map:** `Phi_w(u)=[u;sign(<w,u>)*u]/sqrt(2)`, taking `d+1` dimensions to `2(d+1)`. With toy hyperplane `w=(-1,1,0)`, signs for `q_hat` and D's `x_hat` agree, so the transformed dot product equals `.60`; signs for the duplicate disagree, giving zero. A single random hyperplane can be wrong; replicas `w_1..w_R` are max-pooled to approximate the positive-part score.

**Offline:** BERT contextual corpus-token vectors → append -1 → random-hyperplane lifts → per-replica k-means centroids, inverted postings and compressed residuals.  
**Online, each of K rounds:** contextual query tokens → append existing coverage → random lifts → ANN centroid probes → candidate pooling/pruning → residual refinement → exact marginal-coverage re-score of survivors → select one item → update coverage and repeat. This is repeated retrieval, not only post-top-K reranking.

The authors' checked-out code (historical local clone `/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/DISCo`, commit `53df0b192a80165135d850c015e1755b6a165136`) operationalizes coverage updates in `src/utils.py:partial_chamfer_sim_batched_with_rerank` and the per-round selected-item search in `src/retrievalmethods.py:DISCo.run`. The code uses `torch.maximum` against the running per-token optimum then sums across query tokens; maximizing that new total is equivalent to maximizing marginal gain because the old total is constant within a round. Its `rerank_internal` refers to its *own* coverage-oriented selection, **not an external LLM relevance teacher**.

## Fast retrospective diagnostic — reused fixed historical artifacts

**Scope:** Reanalyse prior `ire-p5-minimal-evidence` HotpotQA distractor pilot; do not repeat it. It has 600 fixed sampled queries (200 dev, **400 holdout**), exactly 10 candidate paragraphs and two gold supporting paragraphs per query, with original dense ranks 1–10. Source and protocol in `/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/protocol.md`; local file `results.json` (SHA-256 `e6ebade0c2d3564a55028c284379e856f52610352db2e14b566ac30159a2ea4e`). The original fixed embedding model is `sentence-transformers/all-MiniLM-L6-v2`. No reranker scores are present in these per-query records. All metric counts below were recomputed from stored `per_query` and `query_info` tables at `k=2` without running a model.

- **Dense top-2:** both supporting paragraphs found in **124/400 = 31.0%**.
- **Greedy DISCo-*style*, no length normalization:** **194/400 = 48.5%**. Versus dense, it fixed **116** queries and broke **46**, net +70, **+17.5 percentage points**. For **280/400** queries it selected an item whose original dense rank was 3–10; for **143/400**, some selected item had original dense rank ≥5; for **60/400**, rank ≥8.
- Against dense top-2, the unnormalized DISCo-style method newly selected at least one **gold** paragraph from original dense rank ≥3 on **155/400** queries; ≥5 on **64/400**; ≥8 on **35/400**. The ranks are **within the same 10-candidate pool**, not rank 60 in a corpus index.
- **Length-normalized DISCo-style:** **142/400 = 35.5%**. It improved 90 queries and regressed 72 versus dense; not all coverage-style variants improve gold evidence.
- **MMR:** **82/400 = 20.5%**. Simply preferring diversity can lose evidence.
- One trace: query `5ae7385d5542992ae0d163da` had original dense selections `[1,5]` at ranks `[1,2]`, whereas unnormalized coverage selected `[1,6]` at ranks `[1,6]`. The two gold documents were `[1,6]`. Thus a lower-ranked complementary gold item was promoted. Counterexample: query `5ae0cddc55429945ae959459` had dense `[2,9]`, both gold, whereas coverage chose `[9,3]`, dropping gold item 2.

**Recompute recipe** (read-only; data and split are fixed):
```python
import json
d=json.load(open('/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/results.json'))
info={q['qid']:q for q in d['query_info'] if q['split']=='holdout'}
r={(x['qid'],x['method']):x for x in d['per_query']
   if x['split']=='holdout' and x['k']==2}
for name in ['dense','disco_approx_no_cost_norm','disco_approx_cost_norm','mmr']:
    n=sum(int(r[qid,name]['full_support']) for qid in info)
    print(name,n,'/',len(info))
for name in ['disco_approx_no_cost_norm','disco_approx_cost_norm']:
    gains=losses=gold3=gold5=gold8=0
    for qid,q in info.items():
        a=r[qid,'dense']; b=r[qid,name]
        gains+=int(not a['full_support'] and b['full_support'])
        losses+=int(a['full_support'] and not b['full_support'])
        new_gold=(set(b['selected'])-set(a['selected']))&set(q['gold_indices'])
        gold3+=int(any(q['dense_rank_by_candidate'][i]>=3 for i in new_gold))
        gold5+=int(any(q['dense_rank_by_candidate'][i]>=5 for i in new_gold))
        gold8+=int(any(q['dense_rank_by_candidate'][i]>=8 for i in new_gold))
    print(name,'gains/losses',gains,losses,'new-gold-rank>=3/5/8',gold3,gold5,gold8)
```

### Critical interpretation

These are **retrospective, descriptive** results of a simplified **DISCo-style selection over an already-known 10-document pool**, *not official DISCo's indexed full-corpus retrieval* and not retriever-vs-LLM score alignment. This dataset includes both gold evidence items by construction; it cannot establish out-of-pool discovery, rank-60 reach, or improvements to learned retriever parameters. The methods are not equal-latency: historical mean query times were approximately `0.8033 s` for the DISCo-style variants versus `0.0186 s` for dense; one-time index preparation is excluded. The 400-item holdout had already been examined for the original pilot; this reuse is **exploratory only**, not independent confirmatory evidence or a newly preregistered test.

**Independent outcome evidence:** Original protocol/result record is `/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/results.md`; historical paper/algorithm reviews agree that coverage is not the same as gold support. Previous bridge-task preregistered G2 and G3 gates failed. Do not present the 48.5% as a new validated DISCo result.

## Alignment question: not established

A teacher cross-encoder can disagree with a dense retriever for **relevance**, while DISCo measures **incremental lexical/contextual-token coverage**. Large rank differences from dense are evidence that these two *selection criteria* differ, **not proof** of retriever–teacher mismatch or its correction. In this data there are **no paired, same-candidate teacher scores**, so actual agreement metrics (rank Spearman/Kendall, top-K overlap, teacher-regret) cannot be calculated without additional scoring or validated external artifacts.

For future **separately authorized and reserved** work, a meaningful screen is: select a fixed initial candidate pool `B` (at least 60 for the explicit rank-60 hypothesis), take independent gold qrels, freeze the encoder/index, and compare **dense top-K**, **cross-encoder rerank of the same pool**, **DISCo-style marginal-coverage same-pool selection**, and **teacher-aware coverage selection** with matched teacher calls and full retrieval budgets. Score those same query–document pairs with the teacher before calculating score/rank disagreement, and isolate **in-pool promotion** from **out-of-pool discovery**. Report full-gold coverage/Recall@K, gain/loss slices, top-K overlap, rank disagreement, mean/p50/p95 wall time and teacher-call count. Only add an independent full-index DISCo arm after the same-pool diagnostic passes and feasibility is established.

**Proposed extension (not from the DISCo paper):** `U(d|S,q)=lambda*T(q,d)+(1-lambda)*Delta_DISCo(d|S,q)`, with appropriately normalized teacher and coverage values. This can trade off relevance and complementarity; it does not automatically fix calibration, and it can favor teacher-scored irrelevant but lexically covering passages. Compare it against cost-matched teacher-only same-pool reranking.

**Recommendation / current execution status:** **PARK full retriever–teacher mismatch claims; retain the existing retrospective evidence as a mechanism sanity check.** Do not spend new teacher or GPU budget merely to repeat the previous HotpotQA pilot. Any next live experiment must pass the existing Ceous owner-ACK/reservation, source/evaluation/cost and independent-review gates in the central tracker. Claude Science collaborators should read this note before proposing overlapping DISCo variants.
