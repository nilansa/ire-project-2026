# Ceous — TTT-Embed Predecessor Capability Matrix (2023–2026)

**Purpose:** A *historical, source-grounded comparison*, **not** a new Ceous research proposal, planned algorithm, or claim of novelty.  
**Date:** 2026-11-16 · **Task:** \`ire-ceous-training-free-rerank-20261116-0524\`  
**Endpoint:** [TTT-Embed, arXiv:2608.12569 (12 Aug 2026)](https://arxiv.org/abs/2608.12569)  
**Source boundaries:** Full primary text inspected for TOuR, ReFIT, FIRST, ColBERT-PRF, and Gera et al. 2026; IBM official abstract / publisher metadata only for SIGIR 2025; TTT-Embed primary arXiv abstract and available summaries. Do **not** pretend the inaccessible SIGIR 2025 full paper was read. All published results below are **author-reported**, not independent Ceous results.

## 1. Correct the central terminology first

Question: *Can the ranking/retrieval quality of an already trained dense retriever be improved using a black-box reranker's scores without updating its encoder weights?*

There are **three meanings** that earlier discussions accidentally conflated:

1. **Frozen encoder** = pretrained query and document network parameters remain untouched.
2. **Frozen index** = document embeddings/index entries do not change; we can still reissue search with an adapted query vector.
3. **No learning at all** = neither the encoder nor any temporary/external vectors are optimized. **TTT-Embed does NOT satisfy this strongest definition**: it *learns a residual vector* from scores, but does not update encoder parameters.

**Critical factual correction:** TOuR *already freezes both encoders* (see §3.1, p. 5733). TOuR, ReFIT and TTT-Embed do **not** require cross-encoder *hidden embeddings or internal weights* to calculate their retriever-side correction: the cross-encoder supplies relevance **scores**; the query/document vectors used in the update come from the **retriever**. Thus “TTT is first to use only a black-box reranker score” is false. The key distinction for TTT is *reusing a learned residual across queries/tasks*, subject to its sharing scope.

**Query-time computation correction:** TOuR may repeatedly retrieve, rerank, and update a query embedding over feedback rounds. ReFIT's *usual* setting uses **one teacher-scoring round**, up to 100 **cheap query-vector gradient updates**, then another retrieval. Do **not** mistake 100 optimization steps for 100 full cross-encoder forward-pass rounds. ReFIT's original Table 1 reports 400 ms GPU total for retrieval+rera nk versus 470 ms with ReFIT under their hardware/candidate setup (a relative 17.5% overhead), **not** an unqualified assertion that per-query feedback is prohibitively slow.

## 2. Historical capability matrix — seven predecessors, then TTT-Embed

Legend: ✓ = demonstrated/required; ✗ = not present under the reported method; — = not applicable; ? = **not established by the material inspected**. “Output only” refers to **access to the external teacher**, not to the retriever's own document vectors.

| Paper (chronological family) | Base retriever encoder weights frozen? | What teacher information does the method consume? Does it need hidden teacher features/weights? | What exactly is adapted? | Does a **new query** require teacher scoring? | Learned teacher state reusable across queries? | Corpus retrieval beyond original slate? |
|---|---|---|---|---|---|---|
| **ColBERT-PRF** (ACM TWEB, Jan 2023; earlier ICTIR 2021 preliminary) | ✓ no additional ColBERT training for PRF | **No external teacher**; consumes retriever's *token-level feedback-document embeddings* | Adds selected centroid/token vectors to **this query** | — (no teacher) | ✗ | ✓ |
| **TOuR** (ACL Findings 2023) | **✓ explicit (§3.1)** | **Cross-encoder scalar relevance scores**; **no teacher hidden features/weights** needed for query update | Gradient-descent update to **current query embedding**, hard or soft pseudo-labels, potentially multiple retrieval/feedback iterations | **✓** for adapted current test query | ✗ | ✓ |
| **ReFIT** (arXiv May 2023) | **✓ explicit (§3)** | **Cross-encoder scalar relevance scores**; **no teacher hidden features/weights** needed | KL-distillation into **current query embedding**; normally one feedback round with multiple gradient steps | **✓** for adapted current test query | ✗ | ✓ |
| **FIRST** (EMNLP 2024) | ✓ retriever frozen **during feedback experiments**; FIRST's **reranker itself is trained** | Stronger **listwise LLM ranking / first-identifier logits**; uses **rank ordering** rather than a scalar-score KL target for feedback | A *rank-loss*-adapted **current query embedding** in §4.4; FIRST also newly trains the teacher/reranker | **✓** for adapted current test query | ✗ | ✓ in feedback experiment |
| **Large-Scale Study of Reranker Relevance Feedback at Inference** (SIGIR 2025) | ✓ under the described test-time query-vector-distillation setup | Cross-encoder **output predictions/scores**; no hidden teacher-state requirement indicated in official abstract | ReFIT-type **query representation**; studies feedback candidates/iterations, domains, languages, modalities, latency | Feedback at inference for individual queries | **? no shared residual demonstrated in official abstract** | ✓ under studied retrieve/feedback/retrieve procedure |
| **Test-Time Compute for Dense Retrieval** (May 2026 preprint) | ✓ frozen embedding model | **No external teacher**: softmax-weighted centroid from initial top-K retriever outputs | Algebraic interpolation into **this query**; no learned teacher residual | — | ✗ | ✓ (adapted query search) |
| **Task-Adaptive Embedding Refinement via Test-time LLM Guidance** (May 2026 preprint) | ✓ frozen embedding model | Scores from a **generative LLM judge** over a sampled document set; **no hidden LLM activations required** by the documented KL algorithm | Gradient steps on **current query embedding**; paper states its main contribution is broader task setup, not a new refinement optimizer | **✓** for adapted current query | ✗ in documented algorithm | ✓ |
| **TTT-Embed** (arXiv August 2026) — endpoint | **✓** | **Only scalar ranking rewards** from black-box reranker or LLM judge, plus **retriever's own** stored embeddings; **no teacher internals** | Optimized **output-space residual vector**, shared by **query / task / global scope**; index unchanged | **Depends:** rewarded query requires scores; later queries with a learned **task/global** state can use it **without another teacher call** | **✓ task/global**; ✗ query-private state on a new query | ✓ adapted query re-search over frozen index |

**These papers should NOT be described as one uniformly comparable experimental lineage:** ColBERT-PRF requires a multi-vector/token index (not a single dense-vector ANN); the May 2026 frozen-compute study uses no reranker; FIRST develops a *trained LLM reranker* as its main contribution; Gera et al. uses an LLM judge and nuanced classification/search tasks. They are **boundary/negative controls**, not replacements for ReFIT/TOuR under exactly identical inputs.

### Five yes/no questions that capture the actual evolution

| Capability | ColBERT-PRF | TOuR | ReFIT | FIRST feedback | SIGIR 2025 | SoftCentroid 2026 | Gera et al. 2026 | **TTT-Embed** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Encoder weights unchanged during retrieval adaptation | ✓ | ✓ | ✓ | ✓ (feedback arm) | ✓ | ✓ | ✓ | **✓** |
| Can use a **score-only black-box cross-encoder** as teacher | — | ✓ | ✓ | Teacher provides ranking (not required scalar scores) | ✓ (study premise) | — | LLM score-based | **✓** |
| Needs **teacher's** hidden representations or weights | — | **✗** | **✗** | Logits are used to build FIRST; ordering is used in feedback | Not indicated | — | **✗** | **✗** |
| Optimizes/adapts the query for each rewarded query | ✓ local PRF | ✓ | ✓ | ✓ | ✓ | ✓ (algebraic) | ✓ | ✓ at query scope; **also shared alternatives** |
| Can deploy **previously learned teacher feedback** to **unscored future queries** | ✗ | ✗ | ✗ | ✗ | Not established in official abstract | — | ✗ in method | **✓** |
| No model-weight updates **and** no learned external state | ✓ | ✗ | ✗ | ✗ for FIRST teacher | ✗ | ✓ | ✗ | **✗** (learns residual) |

## 3. Not all incremental differences are guesses: previously published controlled comparisons

### ReFIT already compares TOuR (2023): Table 6

Published **Recall@100**, in percentage points, for selected shared tasks:

| Task | Raw retrieval | TOuR hard | TOuR soft | ReFIT |
|---|---:|---:|---:|---:|
| Natural Questions (NQ) | 86.1 | 87.0 | 87.2 | **87.6** |
| EntityQuestions | 70.1 | 71.9 | 72.5 | **72.6** |
| BEIR task average (as Table 6 reports) | 66.8 | 68.4 | 68.1 | **69.2** |
| Mr.TyDi (as Table 6 reports) | 87.0 | 88.7 | 88.1 | **89.7** |

**Provenance caution:** ReFIT marks TOuR's NQ and EntityQuestions cells with \`+\` (values carried from the original paper), so these rows are a *published cross-paper comparison*, not a fully new from-scratch matched rerun. ReFIT states it used the **same retriever and reranker** as TOuR for NQ and EntityQuestions, and explicitly analyzes why one teacher feedback round plus more query-vector steps is less expensive than TOuR's repeated rounds. This comparison already addresses the user's concern that test-time query feedback requires repeated expensive teacher calls.

### FIRST compares strength/type of teacher signal (2024): Table 4

All are **retrieval Recall@100 (%)** on BEIR for the **same** Contriever-retrieval family, judged *after query-vector feedback + second search*:

| Model of feedback | Mean BEIR Recall@100 |
|---|---:|
| Unadapted Contriever | **66.8** |
| Cross-encoder scalar feedback with KL | **69.0** |
| Listwise LLM ordered feedback with weighted RankNet | **71.2** |
| Combined cross-encoder + listwise LLM feedback | **72.0** |

FIRST uses 100 gradient updates at learning rate 0.005 in the cross-encoder feedback setting and 20 updates at 0.001 in the listwise LLM feedback setting. These are **not matched numbers of training steps**, though both are in the same paper. FIRST itself trains the LLM reranker with a ranking loss; that must be distinguished from freezing its retriever during test-time feedback.

### SIGIR 2025 is a systematic follow-up, not the original score-only mechanism

The [publisher/IBM publication abstract](https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference) explicitly says previous inference-time distillation had promising effects, then **studies generalization** across retrieval domains, languages and modalities, varying **distillation update count, number of feedback candidates, performance and latency**. This supports an incremental **evidence breadth/efficiency analysis**, not an original invention of black-box scalar feedback.

**Limit on source access:** The **SIGIR 2025 full text was not accessible** in this review (ACM DOI returned access error); do **not** assert it omitted reusable vectors or quote performance tables beyond the accessible primary abstract. The proper statement is *“cross-query persistent state not established by the inspected abstract,”* not *“SIGIR proves there is none.”*

### TTT-Embed (2026) — what changes at the endpoint

The published [arXiv abstract](https://arxiv.org/abs/2608.12569) says it optimizes a residual vector from **scalar rewards** while *both embedding-model weights and the index remain fixed*, uses **global / task / query** sharing, and reports gains **up to +8.36 nDCG@10** (authors' figure; do **not** compare this with Recall@100 rows above). Unlike a query-private correction, a residual trained from an earlier group of feedback queries can be applied to future *unscored* queries. TTT-Embed *does perform optimization of the external residual*; it is **parameter-efficient adaptation**, **not no-optimization inference**.

Its description does **not** establish that older ReFIT or TOuR demanded teacher hidden embeddings: they did not. Rather, TTT changes the **scope and persistence of the corrected representation**.

## 4. Conclusions limited to what the cited work already proves

**Established:** (1) Freeze weights and optimize test-query vectors from teacher score outputs (TOuR / ReFIT, 2023). (2) Reduce feedback rounds and measure latency (ReFIT). (3) Change from pointwise cross-encoder scores to listwise LLM rankings for stronger teacher supervision (FIRST, 2024). (4) Measure more tasks/languages/modalities and feedback-candidate/update/latency settings (SIGIR 2025). (5) Adapt query using no teacher at all via top-result centroid (May 2026). (6) Reuse learned feedback across unseen queries/tasks via scoped residuals with an unmodified encoder/index (TTT-Embed, Aug 2026).

**Not an established claim:** that TTT-Embed is the first weight-frozen method, first score-only black-box-reranker method, first method to re-search the full index, or first method to improve Recall@K. These have earlier precedents.

**Do not cross-compare incompatible published numbers:** ReFIT / FIRST report **Recall@100**, whereas TTT's headline is **nDCG@10**; they do not necessarily share encoder checkpoints, qrels, ANN retrieval depth or feedback budgets. This document reports *paper-level functionality and within-paper comparisons*, not a new benchmark, and proposes **no new problem statement**.

## 5. Primary sources checked

1. [ColBERT-PRF, Wang et al., ACM TWEB 2023](https://doi.org/10.1145/3572405), esp. abstract, §§4.3–4.5, conclusion; source explicitly says *no additional ColBERT training*.
2. [TOuR, Sung et al., ACL Findings 2023](https://aclanthology.org/2023.findings-acl.354.pdf), **p. 5733, §3.1** freezes encoders and defines teacher scalar \`s=phi(q,c)\`; methods §§3.2–3.5.
3. [ReFIT, Reddy et al., arXiv May 2023](https://arxiv.org/html/2305.11744), **§3.2** lists needed **retriever query/document vectors plus reranker scores**, **§4.2** latency, **§5.4 Table 6** controlled TOuR comparison.
4. [FIRST, Reddy et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.491.pdf), **p. 8642** logits/ranking-loss teacher; **§4.4 / Table 4 p. 8649** query-vector feedback comparisons.
5. [A Large-Scale Study of Reranker Relevance Feedback at Inference, Reddy et al., SIGIR 2025](https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference), official abstract (full paper restricted in this review), DOI: 10.1145/3726302.3730160.
6. [Test-Time Compute for Dense Retrieval, Xiao, arXiv 2605.11374](https://arxiv.org/abs/2605.11374), frozen model and teacher-free softmax centroid (abstract).
7. [Task-Adaptive Embedding Refinement via Test-time LLM Guidance, Gera et al., arXiv 2605.12487](https://arxiv.org/pdf/2605.12487), **p. 3 Algorithm 1**, §3, **Appendix B Table 4 / Table 5**, which explicitly compares ReFIT's query-refinement setup with LLM feedback.
8. [TTT-Embed, Chen & Wu, arXiv 2608.12569](https://arxiv.org/abs/2608.12569), original abstract (available full-paper summary separately); main endpoint is reuse of *learned external residual state* across query/task/global scopes.
