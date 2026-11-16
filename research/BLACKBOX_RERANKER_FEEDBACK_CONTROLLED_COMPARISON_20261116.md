# Ceous | Black-Box Reranker Feedback Without Retriever Weight Training
## Canonical 2023–2026 prior-work comparison and pre-registered controlled evaluation (v0.2)

**Date:** 2026-11-16  
**Task ID:** \`ire-ceous-training-free-rerank-20261116-0524\` (reuse existing Ceous cross-surface task; no new file-transfer workspace)  
**Status:** SOURCE-VERIFIED PROBLEM FORMULATION + EXPERIMENT DESIGN. **Not an experiment, no measured Ceous improvement, and no novelty claim.**  
**Parent:** [Initial candidate-repair specification](TRAINING_FREE_CANDIDATE_REPAIR_20261116.md)  
**Existing data header:** [candidate_repair_schema_v1.csv](datasets/candidate_repair_schema_v1.csv)  
**Coordination issue:** [Ceous research #3](https://github.com/nilansa/ire-project-2026/issues/3)

## 0. The exact statement, distilled from discussion

> Can a **frozen** dense retriever improve its *first-stage full-corpus ranking and Recall@K* using only scalar relevance/ranking rewards returned by a stronger **black-box** reranker or judge, without changing retriever/reranker weights or the document index, and with a **strict total budget of purchased reward scores**?

A broader *deployment* question asks whether these rewards can improve **later, unscored queries**, not only the query that triggered the reward. We compare (a) cheap non-teacher feedback, (b) per-query teacher-guided adaptation, and (c) reused teacher-guided state.

**Crucial clarification:** "No training" means **no changes to the underlying encoder or reranker weights**. The following may nevertheless occur: (i) a deterministic shift to the current query vector, (ii) a few gradient steps over a *temporary per-query vector*, or (iii) optimization of a small external correction vector/matrix reused across queries. These are different interventions and must not all be labeled "gradient-free." Optimizing a new residual is *learning of external state*, even while the pretrained model is frozen.

**Available access regimes (never silently conflate):**
- **A — rank-only reranker:** The judge returns ordered document IDs. Requires a predefined fixed rank-to-pseudo-score conversion for numeric-feedback methods. Evaluate separately.
- **B — score-only reranker, vector-search access:** We receive scalar scores of queried pairs, frozen query/document embeddings, and can submit a corrected query vector to the unchanged FAISS index. **This is the primary TTT-Embed-compatible setting.**
- **C — score-only reranker, text-only retriever API:** No embedding output and no ability to search by an edited vector. A score cannot change unseen first-stage candidates directly; one can rerank observed IDs or try query-text rewriting if supported. It is *not* the same experiment as B.
- **D — privileged/oracle:** We possess independently judged qrels or all candidate teacher scores for offline analysis. This defines diagnostic upper bounds, **not a permitted inference-time input**.

The target is **retriever improvement**, not a more powerful reranker trained from scratch, and not a claim that newly relevant documents can be found merely by permuting the initial top-K.

## 1. Narrow 2023–2026 literature, with explicit comparison boundaries

The third column means "not established or not held fixed in the inspected source; investigate against stronger prior art." **Absence of a particular comparison is not proof of an original idea.** Venue status and year are given where verified; the 2026 TTT-Embed and Q-REACT works are arXiv preprints as checked.

| Method / paper | What it **actually does**, and what is updated | What the paper does **not establish for our strict comparison** |
|---|---|---|
| **TOuR** — Sung et al., Findings of ACL **2023** ([paper](https://aclanthology.org/2023.findings-acl.354/), [code](https://github.com/dmis-lab/ToUR)) | Frozen query/document encoders; cross-encoder hard or soft pseudo-labels from initial candidates; **per-test-query embedding gradient optimization**, optionally repeated retrieval. | A shared reusable vector *across multiple queries*; matched-budget head-to-head with 2026 TTT-Embed and active query selection. TOuR proves that gradient-updated **query vectors**, not retriever weights, can help. |
| **ReFIT** — Reddy et al., **2023 preprint** ([paper](https://arxiv.org/abs/2305.11744)) | Frozen bi-encoder; scores from a stronger reranker; **per-query distillation into test query vector and a second retrieval** to improve recall beyond the initial reranker slate. | Rewards stored and reused for queries that were *not* scored; effects under a shared task/global budget compared with TTT-Embed, when all teacher pairs and costs are matched. Some per-dataset/iteration results are negative: do not assume universally monotonic improvement. |
| **FIRST** — Reddy et al., **EMNLP 2024** ([paper](https://aclanthology.org/2024.emnlp-main.491/)) | **Trains a listwise LLM reranker** to get ranking from first-token identifier logits and learning-to-rank loss; experiments also use its stronger reranker signals to improve a retriever at inference. | This is **not** a strictly training-free *construction of the reranker*; treat its trained model as a frozen **feedback-source alternative**, not an apples-to-apples no-training method. Separate feedback quality from teacher inference cost. |
| **Large-scale reranker-feedback study** — Reddy et al., **SIGIR 2025** ([IBM page](https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference); [DOI](https://doi.org/10.1145/3726302.3730160)) | Broadens reranker-guided query representation adaptation to **domains, languages, modalities, distillation update counts, candidate depths and latency**. | No established comparison in the available abstract between **task/global persistent residuals and per-query methods under the same selected reward pairs**. Treat as a mandatory quantitative baseline, **not** a distinct new feedback principle. |
| **GQR** — Uzan et al., **2025 preprint** ([paper](https://arxiv.org/abs/2510.05038); [code](https://github.com/IBM/test-time-hybrid-retrieval)) | Uses a **complementary retriever's** scores to optimize a primary query embedding for **visual document retrieval**. | Different feedback source/modality from a single text retriever + black-box cross-encoder; method is secondary rather than part of the primary text-only same-corpus comparison. |
| **Task-Adaptive Embedding Refinement via Test-time LLM Guidance** — Gera et al., **May 2026 preprint** ([paper](https://arxiv.org/abs/2605.12487)) | Uses generative-LLM guidance on small document groups to refine query embeddings across retrieval/classification settings. | Cost matching with **scalar cross-encoder scores** and TTT-Embed not established by the abstract; use as separate higher-cost judge control where feasible. |
| **Test-Time Compute for Dense Retrieval** — Xiao, **May 2026 preprint** ([paper](https://arxiv.org/abs/2605.11374)) | Searches over inference programs for a **frozen embedding API** with **no external teacher at query time**; weighted feedback centroid/Rocchio-like operations emerge as strong cheap controls. | Not a black-box-reranker-score method. Include a zero-teacher **centroid/PRF control** so our apparent gain is not merely extra query compute. |
| **DART** — Liu & Li, **May 2026 preprint / 2026 workshop** ([paper](https://arxiv.org/abs/2606.01070)) | Adapts a **bilinear reranking matrix at inference** from the retriever's own top/bottom pseudo-labels, without an external teacher. | It **updates the scoring matrix**, not primarily first-stage reretrieval from an unchanged inner-product index. It is outside the strict score-only external-reranker regime but a useful training-at-test-time contrast. |
| **TTT-Embed** — Chen & Wu, **August 2026 preprint**, Google DeepMind collaboration ([paper](https://arxiv.org/abs/2608.12569)) | **Most direct primary baseline.** Uses *only scalar ranking rewards* to learn a residual **in the output embedding space** of frozen retriever; query, task, or global sharing, with unchanged document index and second retrieval. Compares identical reward pairs and budgets against **direct reranking and its own scopes**. | Its evaluation explicitly treats **which query/doc pairs to reward as external**, using task-balanced nested water filling; active selection of query/doc targets is not itself optimized. Its listed primary baselines do **not include a matched TOuR/ReFIT implementation**. Decomposition of observed gains into (i) originally selected/rewarded docs and (ii) docs never in initial feedback slates is a target for explicit audited evaluation; this is **not** saying the method cannot retrieve unseen docs—it does. |
| **Q-REACT** — Li et al., **September 2026 preprint** ([paper](https://arxiv.org/abs/2609.27688)) | Direct follow-up in **visual document retrieval**: learns **query-dependent low-rank residuals shared across queries**, adds a document-context score, and normalizes student distributions **over the full task index**. Compares to TTT-Embed and GQR on ViDoRe V3 using matched scored candidates/budgets for the feedback-based methods. | Not a same-text-corpus head-to-head with ReFIT/TOuR and TTT under our planned fixed FAISS setup. Its gain must be decomposed into **learned residual vs document-context prior vs full-index normalization**, rather than credited entirely to feedback. Q-REACT already studies unscored pages—do **not** claim this has never been done. |

### Verified numeric evidence — all *authors' reported results*, never Ceous results

- **ReFIT / BEIR:** paper Table 7 reports mean Recall@100 for rerank-100 at **66.8**; its inference feedback with K=10 scored passages gives **67.6**, K=100 gives **69.0**, with different feedback budgets. This is *not* a shared-budget head-to-head against TTT-Embed.
- **TTT-Embed:** evaluates **15 specified MTEB tasks**, 12,263 queries and 1,661,208 corpus items total across tasks, using Qwen3-Reranker-4B as the default teacher. It compares the **same rewarded pairs**; at normalized b=10, mean \`Delta nDCG@10 x 100\` is **+6.36 task-sharing**, versus **+3.21 direct reranking**. At b=100 the query-wise column reaches **+8.36**, direct reranking **+7.80**. These are different budgets; never compare two cells across budgets as if matched.
- **Q-REACT / ViDoRe V3:** eight visual-document tasks, 14,514 queries; at b=10, five-backbone average nDCG@10 x 100: **54.73 base, 60.25 direct rerank, 56.81 TTT Global, 58.62 TTT Task, 61.14 Q-REACT Global, 62.61 Q-REACT Task**. These are **visual document numbers and method-specific scoring functions**, not text retriever benchmark results.

## 2. The exact scientific separation we need

**Three noninterchangeable metrics/outcomes:**
1. **Ordering:** NDCG@10 / MRR@10 on a set of already-retrieved document IDs. Reordering the same top-10 set **cannot increase the set-level Recall@10**; moving items within those ten may change nDCG/MRR.
2. **Candidate admission:** Relevant IDs existed in the initial broader top-B, B>K, but were missing from the initial top-K. Promoting them *can* increase Recall@K.
3. **New discovery:** The new search retrieves relevant IDs **outside the original top-B and beyond the set of teacher-scored candidate IDs**. This is the hardest test that a correction genuinely improves first-stage retrieval rather than only rearranges its local slate.

**Four offline diagnostic ceilings; none may be input to the live policy:** (a) oracle ordering of exactly the original top-K, (b) oracle selection of K from initial top-B, (c) oracle selection of K from explicitly evaluated expanded candidates, (d) best teacher-scored ordering (teacher preference proxy, **not ground truth**). With sparse qrels, these are only **qrel-conditional observed** bounds, *not* full-corpus true oracles. Track judged coverage; unlabeled is **UNKNOWN**, not a negative. A genuinely complete corpus-wide oracle is not available under ordinary BEIR sparse judgments.

**Signal types** to compare explicitly: original retriever score; fixed-backbone embeddings/ranks/gaps; teacher **scalar** score (the primary case); teacher **ordinal rank only** converted by a prespecified monotone mapping in a separate ablation; independent qrel grade **only for evaluation**.

## 3. One fair common benchmark, not incomparable published tables

**Primary text-retrieval pilot (chosen because the same task family occurs in both ReFIT and TTT-Embed):**
- **FiQA-2018**: in TTT-Embed's task manifest: **648 test queries, 57,638 corpus entries**. Use the exact task revision/qrel split, hash and save identifiers. ReFIT reports FiQA on standard BEIR; same task name does **not** guarantee matching task revision, candidate pools, encoders or evaluation.
- **SCIDOCS**: TTT-Embed manifest: **1,000 queries, 25,657 corpus entries**; ReFIT also evaluates Scidocs. Different relevance densities test whether policy robustness transfers. The intended first pilot is FiQA; SCIDOCS is a confirmation, not data to tune against after seeing results.
- **Not our primary common dataset:** Q-REACT is ViDoRe V3 **visual pages**, not comparable directly to FiQA text without running all baseline methods on that task.
- **ReFIT original checkpoint is not held constant with TTT**; numerical comparisons across published papers are forbidden. We rerun available methods with the *same new checkpoint and candidates*.

**Fixed-model primary setup (preregister as a plan, not an already executed run):**
- Frozen base: **Qwen3-Embedding-0.6B**, chosen to overlap the published TTT-Embed models; freeze precise upstream revision/hash and encode corpus **once**.
- Frozen teacher: **Qwen3-Reranker-4B** (matching TTT-Embed's default), take \`P(Yes)\` computed from frozen Yes/No logits if using its documented input template; snapshot model/prompt and score extraction. If not available, **do not silently substitute a MiniLM and call the result a TTT-Embed reproduction**. A MiniLM teacher is a separately tagged cheap-signal sensitivity study.
- L2-normalize vectors; primary *common* ANN: **FAISS HNSWFlat / inner product**, fixed index file, insertion order, M and efConstruction, **same efSearch** for every method. First use an exact **IndexFlatIP** shadow baseline for candidate overlap and leakage checks. Calibrate HNSW only on development queries; freeze its settings before evaluating held-out queries. IndexFlatIP is exact; do **not** call it an ANN approximation.
- One shared initial depth \`B=100\`, shortlist/output K=10, teacher-score eligible candidate universe \`C_B\` for every feedback policy, plus exactly the same immutable document universe. All methods make a fresh corpus search against the **same fixed index** when their method supports a modified query vector.
- Fix query-ID splits for **held-out transfer**: a disjoint feedback acquisition/development set and an untouched evaluation set. For the *separate online-per-query* evaluation, test queries may be teacher-scored within their charged budgets but their qrels remain hidden from the method. Reserve a third task for stronger cross-task transfer if extended.
- Record all per-query vectors, raw similarity, initial top-100 IDs/ranks, teacher-score cache keys, final full-index top-100 IDs/ranks, qrels, and runtime. Reconstruct no missing observation as 0.

**Two test-query regimes (DO NOT MIX):** The *online/per-query* regime permits a charged teacher score for the very query being evaluated (but never its qrels); that is the native ReFIT/TOuR setting, and a global/task state may be fitted on the same rewarded queries with results reported as **transductive adaptation**. The *held-out transfer* regime collects teacher scores only on a disjoint feedback-acquisition subset, then evaluates later queries without teacher scores; ReFIT/TOuR query-private states have no transfer mechanism in that regime and must be shown as **unadapted / not applicable**, not given free teacher calls. Record both regimes separately, with the same pair eligibility and accounting within each regime. No test qrels may inform either regime.

**Two separate and fair experiments (DO NOT MIX):**

### E1. Fixed-feedback update-rule test
All variants **must receive the exact same scored query–document pairs** and same reranker numeric scores, and run over the exact same frozen query/document vectors and index:
1. Raw frozen retriever (zero score calls; report its lower cost separately).
2. Direct reranking only of those scored IDs, preserving unchanged unscored positions (zero full-corpus discovery).
3. Teacher-weighted Rocchio/centroid query correction (simple, **gradient-free** baseline; with and without teacher weights).
4. ReFIT query-specific adaptation, with frozen weights and same scored IDs.
5. TOuR query-specific hard/soft adaptation, same scored IDs (where model architecture permits).
6. TTT-Embed query, task, global variants, with the exact same teacher pairs and intended original hyperparameters. When test queries have no feedback, query-specific alternatives are not magically given extra rewards. Compare overall score and conditional score on rewarded vs unscored queries separately.
7. **Optional** Q-REACT-style low-rank external state on text, explicitly marked a *port*, with equal update compute and documented context-feature controls. If not implemented, report "NOT RUN," not 0 performance.

E1 isolates **how rewards are converted into a query-correction state**. Control optimization steps (and report total FLOPs/latency), not just reward calls.

### E2. Fixed-budget acquisition test
Hold **the adaptation rule fixed**, preferably TTT-Embed task-wise first, and choose a total feedback-pair budget \`M\`. Policies may request **different** pairs (otherwise active selection has no meaning) but must share the same candidate universe, teacher, score cache, max unique pairs, query splits and compute budget.

Compare **task-balanced water filling from TTT-Embed**; seeded random query selection; highest base-score-gap/uncertainty query selection; cheap lexical–dense disagreement; and a prespecified candidate-diversity or novelty policy. Include simple score-gap+random and PRF controls. For every budget show unique pairs, distinct queries, candidates/query, number of new second-search relevant IDs, delta nDCG@10 and 95% paired query bootstrap CI. Choose budget points in normalized \`b=M/N\` e.g. \`0, 0.1, 1, 10\` with the exact integer M specified and the same cache accounting. **Never use gold labels or unpurchased teacher scores to choose a pair**; the large reward-cache used for offline diagnostic oracles is masked from each policy.

The proposal *might* have value if E2 reallocates sparse teacher calls more effectively than water filling, under a metric/latency Pareto frontier and with E1 demonstrating the improvement is not purely from a stronger optimizer. It is not enough to outperform direct reranking, which TTT-Embed has already shown in its source setting.

## 4. Dataset columns and audit constraints

Use one row per stable \`(dataset, query_id, doc_id, run_id)\`, plus a separate per-query record and full-model/index manifest. Existing [26-column CSV header](datasets/candidate_repair_schema_v1.csv) is a **schema only**, with **zero** measured rows at this date.

**Must contain**: \`retriever_score_raw\`, \`retriever_rank\`, \`reranker_score_raw\` (nullable), \`reranker_was_called\`, \`desired_relevance_grade\` from qrels (nullable), \`qrel_judgment_status\`, \`oracle_candidate_pool_id\`, \`desired_oracle_rank_in_pool\`, \`initial_top_k\`, \`initial_top_b\`, \`expanded_candidate\`, \`candidate_source\`, \`final_rank\`, and distinct feedback cost. Add these fields when materializing a real run: \`feedback_set_id\`, \`reranker_input_kind\` (scalar/ordinal), \`rank_to_score_rule\`, \`base_index_sha256\`, \`index_kind\`, \`model_revision\`, \`score_normalization\`, \`document_ever_rewarded\`, \`document_seen_in_initial_B\`, \`query_rewarded_in_feedback_split\`, \`judged_query_coverage\`, \`oracle_coverage_warning\`, and \`latency_p95\` from per-query telemetry.

**Do not confuse**: teacher preference/oracle (stronger *prediction*), independently judged evaluation relevance, and mathematically computed ideal ordering. The reranker does not know true qrels; qrels are not a training signal in E1/E2.

## 5. Decision tree, first three falsification gates

- **G0 — Headroom audit without teacher calls:** Does exact/frozen HNSW base retrieval show material judged relevance in ranks 11–100? If **no**, an admission-only method cannot help; proceed to discovery probes or stop.
- **G1 — Query feedback value:** Under same scored pair set, does TTT-Embed beat (a) no-feedback centroid, (b) simple score-weighted Rocchio, and (c) ReFIT? If **not**, score-probe complexity or sharing scope is unjustified.
- **G2 — Sparse budget allocation:** At equal unique teacher pairs and frozen adaptation rule, does the cheap-signal selector beat TTT-Embed's task-balanced water filling with query-paired uncertainty intervals, **on held-out queries**, without higher retrieval/search latency? If **not**, no active-selection benefit demonstrated.
- **G3 — Out-of-feedback generalization:** Evaluate **(a)** originally rewarded docs, **(b)** unscored docs previously in top-100, **(c)** docs absent from all original rewarded slates, and **(d)** separate unscored test queries. Improvements to (a) alone are insufficient to claim full-corpus transfer; note that TTT-Embed and Q-REACT already enable nonlocal retrieval.

**Negative-result rule:** A score improvement obtained from reordering a teacher-scored set is a *reranking* result, **not proof of retriever discovery**; a result obtained by deeper retrieval or more teacher calls without budget accounting is **not a fair win**. At least the frozen base, matched teacher rerank, score-weighted Rocchio, ReFIT, TOuR and TTT-Embed must be audited before describing novelty.

## 6. First executable tasks and explicit current status

1. [DOCS DONE] Verify TTT-Embed name/Google DeepMind collaboration, read its algorithm, main baselines and 15-task manifest; read ReFIT, TOuR, FIRST, SIGIR 2025 and Q-REACT primary descriptions. Source links are recorded above.
2. [SPEC DONE] Separate E1 fixed feedback from E2 active feedback acquisition and choose common FiQA/SCIDOCS + HNSW/exact-shadow setup.
3. [NOT RUN] Pin dataset snapshot hashes, model weights/revisions, reranker prompt, FAISS build, qrel protocol, seeds, and query IDs in a fresh experiment manifest; do not reuse the older unverified Ceous MS MARCO/ANCE pilot as source of ground truth.
4. [NOT RUN] Execute a teacher-free Stage-0 analysis; output actual candidate lists, qrel coverage, and conditional oracle headroom.
5. [NOT RUN] Cache bounded teacher-scored pairs; benchmark direct rerank, teacher-free/teacher-weighted Rocchio, ReFIT, TOuR, TTT-Embed on **the same pairs**.
6. [NOT RUN] Run a separate feedback-selection E2 test with balanced budgets and paired confidence intervals; independent scientific review.

**Scientific caution:** The literature matrix is grounded in checked primary sources. It is **not** a completed systematic review of every 2026 paper. Claims in the third column are specific *falsifiable comparison requests*, not assertions of proven prior-art absence. This text records no new retrieval, teacher calls, experiments, observed improvements, GPU usage or model modifications.

## 7. Primary-source anchors

- [Google DeepMind collaboration: TTT-Embed, Chen & Wu, 2026](https://arxiv.org/abs/2608.12569) — Sections 2–5 and App. 12; actual 15-task list in additional experiment details; baseline comparisons in §3.4; confirms acquisition allocation is external.
- [TOuR, ACL Findings 2023](https://aclanthology.org/2023.findings-acl.354/) — per-query hard/soft pseudo-label guidance and frozen encoders.
- [ReFIT, 2023](https://arxiv.org/abs/2305.11744) — updated query vector, second retrieval, BEIR and failures.
- [FIRST, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.491/) — trained reranker, feedback demonstration.
- [SIGIR 2025 large-scale study](https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference) — multi-domain/modality/latency follow-up.
- [GQR, preprint 2025](https://arxiv.org/abs/2510.05038) — complementary-retriever query refinement in visual retrieval.
- [Test-time LLM guidance, preprint 2026](https://arxiv.org/abs/2605.12487) — task-adaptive generative-judge guidance.
- [Frozen-embedding test-time compute, preprint 2026](https://arxiv.org/abs/2605.11374) — zero-teacher comparator.
- [DART, 2026](https://arxiv.org/abs/2606.01070) — self-pseudo-label test-time scoring updates.
- [Q-REACT, 2026](https://arxiv.org/abs/2609.27688) — visual full-index low-rank cross-query feedback; explicit baseline ablations required for fair interpretation.
