# Ceous | Training-Free Candidate Repair with Budgeted Relevance Feedback

**Version:** 0.1 — 16 November 2026  
**Status:** Research specification and literature map; **hypotheses only, no new experimental result or novelty claim**.  
**Project:** [Ceous](https://github.com/nilansa/ire-project-2026) · **Task ID:** ire-ceous-training-free-rerank-20261116-0524  
**Research boundary:** Freeze retriever and reranker weights; optimize candidate discovery, query representations, score fusion, and selective scoring *without model fine-tuning*. Inference-time query-vector adjustment is a distinct, explicitly labeled category; the strictest variant disallows even gradient-based query optimization.

## 1. Why this is the right pivot

The starting [Microsoft Research challenge](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/) seeks efficient **post-training of retrieval models from high-quality reward feedback**, asking which query-document pairs to score, how to formulate losses, and how to reduce repeated expensive post-training iterations. Ceous first investigated ANN candidate generation and teacher-guided retriever training. Earlier Ceous pilot notes did **not** establish that the retrained student improves over the frozen starting retriever; do not present those negative/partial results as evidence for this new idea.

**New first-stage question:**

> With a frozen retriever and a frozen stronger reranker, can a budgeted, training-free policy improve independently judged top-K relevance by correcting ranking errors **and** recovering useful documents omitted from the initial top-K, using cheap retriever-side signals and only selectively acquired expensive reranker feedback?

The distinction is non-negotiable:

- **Ordering error:** A relevant document is present in the initial top-K, but ranked too low. A reranker can fix order.
- **Admission error:** A relevant document is at positions K+1 to B of an initially retrieved larger pool. A policy may promote it into the **K candidates selected for expensive reranking**.
- **Discovery error:** A relevant document is outside the entire initial B-candidate pool. No policy confined to that pool can recover it; a second query, a different retriever, or graph/index expansion is needed.

The goal is **not** simply to invent "retrieval without training": that is established. The difficult, falsifiable proposal is **budgeted allocation of inexpensive candidate correction and expensive supervision across these error types**. Training a new retriever is a possible *later* extension, not part of the principal claim.

## 2. Precise task and boundaries

For a query q, corpus D, frozen first-stage score s_R(q,d), frozen expensive teacher/reranker score s_T(q,d), and independent relevance judgment y(q,d):

- Let C_B(q) be the first B documents surfaced by the initial retriever, with B > K.
- Let S_K(q) be the K documents admitted for the final expensive reranking stage.
- Allow an **optional** expanded set C_plus(q) from extra retrieval probes or hybrid/graph expansion; record exactly where its documents came from.
- The inference policy may inspect raw retriever scores, score gaps, rank, lexical signals, embedding geometry, and any teacher scores **actually purchased within budget**.
- It must not read offline qrels, hidden teacher scores, or oracle rankings during inference. No model-parameter updates.
- The policy outputs candidates for a fixed reranker and/or final top-k documents; report both intermediate selection quality and final independent relevance quality.

Conceptual objective: maximize E_q[utility_y(final_K(q))] subject to teacher calls <= M, total retrieval work <= R, and p95 latency <= L. Report a Pareto frontier in quality vs. teacher calls, retrieval probes, latency, and monetary compute cost instead of silently collapsing these into one unsupported number. All budgets and K/B values must be fixed *before* test evaluation.

**Three tiers of "without training":**

| Tier | What is allowed? | Example | Important qualification |
|---|---|---|---|
| A. Strictly training-free and gradient-free | Fixed models; deterministic score fusion, PRF, vector arithmetic, query expansion, query-adaptive rules, extra search | BM25/RM3, RRF, centroid shifts, ColBERT-PRF | Strongest interpretation of user's request |
| B. Frozen-weight test-time optimization | Fixed model weights; per-query vector updates, possibly gradient steps | TOuR and ReFIT | No retriever fine-tuning, but **does** optimize a query at inference |
| C. Training or distillation | Updates student/retriever/reranker parameters across examples | RocketQAv2, DynamicRAG | **Out of scope** for primary experiment; optional later work |

## 3. Canonical prior-work comparison (first-pass; expand and independently verify)

**Read the "remaining question" column as a distinction to investigate, not proof that prior work failed or that Ceous succeeds.** Some methods use pre-existing trained models, which is compatible with freezing them in the evaluation.

| Work (year) | What was already demonstrated | Model weights trained for this method? | New candidate discovery? | Expensive teacher at query time? | Remaining comparison / exact overlap with Ceous |
|---|---|---|---|---|---|
| Rocchio / RM3 (1971/2001 onward) | Pseudo-relevance feedback adjusts a lexical/vector query from retrieved documents | No new neural training required | Yes, if retrieval reruns | No | Cheap, indispensable baseline; PRF topic drift and false-positive reinforcement must be checked |
| [Reciprocal Rank Fusion (RRF, 2009)](https://doi.org/10.1145/1571941.1572114) | Combines several ranked lists without fitting a ranker | No | Yes, via union of source lists | No | Strong training-free fusion baseline; cost includes extra retriever runs |
| [ColBERT-PRF (2023)](https://doi.org/10.1145/3572405) | Embedding-based PRF can rerank and reretrieve by expanding query representation | No **additional** ColBERT training | Yes | No new teacher needed | Direct baseline for frozen dense/late-interaction retrieval and expanded-pool gain |
| [TOuR (Findings ACL 2023)](https://aclanthology.org/2023.findings-acl.354/) | Iteratively optimizes a test query vector using cross-encoder pseudo-labels | Query-time vector gradient steps, not weight training | Yes | Yes | **Direct novelty blocker** for claiming "reranker feedback changes frozen retriever and retrieves missed documents" |
| [ReFIT (2023)](https://arxiv.org/abs/2305.11744) | Distills a reranker's scores into a test query embedding, then retrieves again | Query-vector update, not base-weight training | Yes | Yes | **Closest direct competitor** for reranker-to-retriever inference feedback; must match teacher calls, second-search cost, latency |
| [HyDE (ACL 2023)](https://aclanthology.org/2023.acl-long.99/) | Generates a hypothetical document and embeds it for zero-shot dense retrieval | No retriever fine-tune in method | Yes | Generation LM needed | Extra-generation cost; not simply a cheap signal, but a useful fixed-model retrieval baseline |
| [Query2doc (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.585/) | LLM-created pseudo-document expands a sparse/dense query without fine-tuning | No | Yes | Generative LM query cost | Test whether a cheap local alternative can beat or match it under same total compute |
| [UPR (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.249/) | Unsupervised pretrained-LM passage reranking | No task-specific fine-tune | No (fixed input list) | LM likelihood on passages | Illustrates pure reranking bound; not candidate discovery |
| [LADR (2023)](https://github.com/Georgetown-IR-Lab/ladr) | Lexically seeded graph exploration accelerates dense retrieval | Does not require retriever retraining | Yes | No | Relevant training-free source for extra candidate admission and latency comparison; not inherently teacher-aligned |
| [FIRST (EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.491/) | Faster listwise reranking; importantly tests LLM-reranker feedback for a retriever during inference | FIRST itself trains a listwise reranker; inference feedback uses frozen weights | Yes in its relevance-feedback experiments | Yes | **Direct novelty blocker**: also studies richer feedback for query-level retrieval adaptation; compare cost/quality, do not claim unexplored feedback |
| [DISCo (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ded182245f8e4db3963d2027174133d-Abstract-Conference.html) | Builds a dense subset by iterative query-token *collective coverage* and residual query edits | Index construction required; not standard teacher distillation | Yes | No required teacher | Its objective is multi-item coverage, NOT merely individual document relevance; evaluate on multi-evidence tasks separately |
| [NashANN / Welfarist Diverse Similarity Search (2026)](https://arxiv.org/abs/2602.08742) | Welfare-based adaptive balance of relevance and diversity with ANN subroutines | No new language-model fine-tuning implied | Yes | No required teacher | Diversity alone is not a demonstrated teacher-alignment or downstream relevance gain |
| [Bi-metric Framework (ICML 2026)](https://proceedings.mlr.press/v306/xu26au.html) | Nearest-neighbor search with cheap proxy distance and limited expensive ground-truth distance evaluations; reports better accuracy-efficiency tradeoffs than retrieve-then-rerank on many BEIR datasets | Data-structure construction; no retriever fine-tuning required | Yes | Yes, in expensive metric evaluations | **Closest systems/theory challenge** to "cheap scores + sparse expensive scores"; check metric assumptions and reproductions |
| [DynamicRAG (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f69d2246a85e28c3e63dee3daf8d4af8-Abstract-Conference.html) | Learns to adapt reranking order and document count using generation feedback | Yes, RL optimization | Works over selected retrieved documents | Generation feedback | Shows adaptive decisions are established, but violates strict no-training objective |

**Important historical distinction:** The supplied 2024 Big ANN Challenge paper, associated presentation, and workshop summary concern ANN index/search efficiency and nearest-neighbor accuracy; they do **not** provide ground-truth semantic relevance labels or a measured retriever–reranker alignment dataset for this task. Do not substitute ANN Recall@K for independently judged retrieval relevance.

### What would actually count as a new result?

Only after a systematic review and experiments, a possible narrow contribution is: a **fully frozen, optionally gradient-free, query-adaptive candidate-repair policy** that identifies admission versus discovery failures using inexpensive observable signals, chooses whether to expand retrieval or purchase teacher scores, and improves judged top-K quality at **matched teacher, retrieval and latency budgets** relative to RRF/PRF/ColBERT-PRF and ReFIT/TOuR where applicable.

This is **a hypothesis, not a novelty claim**. If ReFIT, TOuR, the bi-metric method, or a simple deeper rerank dominates at comparable cost, pivot or report the negative result. A DISCo + NashANN + uncertainty combination by itself is not evidence of novelty; prior Claude Science review of a different such hybrid recommended PIVOT.

## 4. Oracle ladder: isolate the source of possible improvement

Define evaluation **only on held-out, independently judged relevance**, and state which document universe is being used:

1. **Baseline ranking:** initial retriever top-K and frozen reranker on that same top-K.
2. **Oracle-order@K:** best ordering of *exactly the original top-K IDs*. Identifies pure ordering headroom; cannot change set-based recall.
3. **Oracle-select@B-to-K:** best K IDs from the initial broad pool C_B. Identifies potential admission gain available *without new retrieval*.
4. **Oracle-select@plus-to-K:** best K IDs from C_B union all explicitly evaluated expanded candidates. Upper bound for the chosen expansion process.
5. **Full-corpus oracle:** only definable if sufficient relevance judgments exist for the full corpus, which generally they do not. Do not claim a computable full-corpus oracle on sparse qrels.

**Teacher oracle vs. relevance oracle:** Ranking C_B by a reranker's score (teacher-ranked upper-bound experiment) is **not** the same as ranking by true human qrel relevance. Preserve both as different columns and report teacher–qrel disagreement. Unjudged is UNKNOWN, not negative.

Example diagnostic for one query with K=3, B=6: a relevant document at retriever rank 5 can be promoted by an admission policy; one that is absent from all six requires candidate expansion. An offline oracle can reveal possible headroom but may not be used as a live selector.

## 5. Dataset contract — record scores AND counterfactual target, without leakage

**Primary unit:** one row per (dataset_id, query_id, document_id, candidate_generation_run), deduplicated by stable IDs when multiple retrievers surface the same document. Keep per-source ranks/scores in an auxiliary source-score table; never destroy provenance.

**Minimal essential columns requested:**

| Column | Type / allowed null | Meaning |
|---|---|---|
| query_id, doc_id, dataset_id, split | string, required | Stable join keys; held-out test split |
| retriever_score_raw | float or null if no corresponding score | Original frozen retriever score **before** policy, with model/version |
| retriever_rank | integer or null | Original rank; null for truly not retrieved |
| reranker_score_raw | float or null | Expensive frozen teacher score **only if actually evaluated**; never fill unscored with zero |
| reranker_was_called | bool, required | Separates missing feedback from an actual poor score |
| desired_relevance_grade | integer or null | Independent qrel/target utility; null for UNJUDGED, **not inferred from teacher** |
| desired_oracle_rank_in_pool | integer or null | Counterfactual rank among a specified *judged candidate pool*; report tie handling and pool |
| initial_top_k, initial_top_b, expanded_candidate | bool each | Distinguish ordering, admission and outside-pool discovery |
| candidate_source | string | Dense, BM25, PRF, hybrid, graph, or additional probe |
| selection_action | string | keep / promote / extra-retrieve / probe-teacher / reject |
| final_rank, final_selected | integer/null, bool | Actual policy output and evaluation decision |

**Additional provenance/measurement columns (required before reporting results):**

- query_text_hash, corpus_snapshot_id, run_id, query_split, scoring_model_id, scoring_model_sha, embedding_index_id, score_direction, normalization_method
- reranker_rank_among_scored, reranker_score_observation_stage, teacher_call_id, teacher_query_doc_budget, teacher_tokens, teacher_latency_ms
- qrel_source, qrel_label_scheme, qrel_judgment_status = JUDGED / UNJUDGED, qrel_gain_for_metric; qrel threshold (TREC DL passage-grade 1 is *related*, not answer-relevant for binary relevance)
- baseline_rank, baseline_score, corrected_rank, was_new_to_C_B, initial_candidate_depth_B, final_shortlist_K, expansion_probe_count, per_query_p50/p95 latency bookkeeping, error_category
- oracle_definition, oracle_candidate_pool_id, oracle_completeness_flag, evaluation_cutoff, score_cache_key, teacher_total_cost, test_visibility_policy

**Storage recommendation:** immutable JSONL/Parquet candidate rows and a separate query-level run manifest; export CSV for inspection. There should be explicit nulls for unobserved scores. All rows must have a **query/document score provenance**, not a synthetic "teacher score" invented from retriever rank.

**Strict separation:** Offline, full-budget teacher scoring of a *small audit subset* may establish a **teacher preference ceiling** and measure score agreement, but those withheld scores cannot flow into a budget-limited test-time policy. Relevance qrels never flow into runtime policy; training/tuning and test queries must be disjoint.

### Suggested CSV header (schema only; zero real rows as of v0.1)

~~~csv
dataset_id,query_id,doc_id,split,candidate_generation_run,candidate_source,retriever_score_raw,retriever_rank,reranker_score_raw,reranker_was_called,desired_relevance_grade,qrel_judgment_status,desired_oracle_rank_in_pool,oracle_candidate_pool_id,initial_top_k,initial_top_b,expanded_candidate,selection_action,final_rank,final_selected,teacher_calls_cumulative,retrieval_probe_count,teacher_model_id,scoring_model_sha,qrel_source,query_latency_ms
~~~

An example dataset should contain three *numerical* axes when available: **retriever score**, **reranker score**, and **independent target relevance/desired ordering**. They are deliberately different; the reranker is a predictor, not ground truth. Documents not initially surfaced may still receive *offline* qrels and expanded-retrieval ranks but cannot have a fabricated original retriever rank.

## 6. Experiments — sequence of falsification gates

**Stage 0 — no expensive inference, no new training.** Select a public passage-retrieval benchmark with accessible candidate runs and qrels; start with TREC DL 2019/2020 passage ranking or a suitable BEIR dataset. Check licensing and qrel semantics. Compute rank gaps, candidate overlap, qrel judged fraction, Oracle-order@K, Oracle-select@B-to-K, and candidate-set coverage with K in {5,10,20} and B in {50,100,200}, only where available. Pre-register relevance thresholds. If Oracle-select@B-to-K has negligible headroom, the first proposed task fails for that setting; test discovery not admission.

**Stage 1 — training-free and teacher-free comparators.** Starting retriever, BM25+dense RRF, classical PRF/RM3, naive larger candidate depth, and any feasible ColBERT-PRF / LADR-style fixed-model expansion. For each, record distinct candidate IDs, retrieval work and judged relevance. Avoid assuming available pretrained dense-index exports are interchangeable.

**Stage 2 — expensive-feedback baselines (controlled).** Freeze a disclosed cross-encoder; compute fixed-budget rerank-only top-K, randomly selected teacher probes, score-gap/uncertainty probes, and teacher scoring the B pool as an expensive reference ceiling. Reproduce ReFIT and TOuR if infrastructure and budget permit; otherwise mark "not run" explicitly. Run the proposed policy with *the same teacher-call quotas*, extra-retrieval accounting, and comparable wall-clock settings.

**Stage 3 — selective candidate repair hypothesis.** Starting with cheap top-B metadata, estimate whether each query likely has (a) only ordering headroom, (b) hidden high-value B-to-K candidates, or (c) evidence of missing facets requiring a new retrieval probe. Choose: leave unchanged; promote via cheap score fusion; purchase a small teacher batch; or expand retrieval once. Keep the initial design fully heuristic/deterministic; no fitted classifier and no query gradients. Measure separate outcomes for each trigger.

**Stage 4 — only after Stage 2 is positive.** Explore query-vector inference optimization and perhaps stronger LLM feedback, not simultaneously with the strict no-training claims. Later work may revisit Microsoft-style post-training on collected high-value examples as an explicit new question.

### Required baselines and ablations

- Initial retriever top-K; full rerank of the **same** top-K; deeper B rerank; fixed number of candidate-expansion probes.
- RRF / RM3 / ColBERT-PRF where feasible; score-only, rank-only, uncertainty-only, lexical-only and cheap fusion.
- ReFIT / TOuR (frozen weights, per-query adaptation) and an honest comparison with the 2026 bi-metric framework if an implementation is feasible.
- Teacher selection: random versus highest initial rank versus score-gap/uncertainty versus the proposed selective policy, all with equal *unique* query-document teacher score budgets (respect cached score reuse).
- Index effects: exact versus ANN when relevant, using the *same* queries, corpus, checkpoint and evaluation; do not treat ANN Recall as semantic relevance recall.

### Outputs and decision criteria

**Effectiveness:** independently judged nDCG@10 (primary if graded qrels), MRR@10 and Recall@K where valid; oracle gaps; incremental *relevant* candidate coverage; unjudged rate; per-query gains and losses.  
**Efficiency:** unique teacher pairs, calls/tokens, extra embedding/retrieval/search work, p50/p95 wall-clock latency, CPU/GPU and memory, amortized index build time.  
**Reliability:** paired bootstrap confidence intervals (query as unit), at least two K/B settings, multiple datasets when feasible, qrel protocol, score calibration and seed controls.

A result qualifies as progress only if the new method improves held-out judged ranking quality at the same teacher-call and total-work budget, or demonstrates a better documented quality–cost Pareto frontier. It is **not sufficient** to correlate with teacher scores, improve teacher-preference coverage alone, or perform better by purchasing more computation. Report losses and failure cases.

## 7. Ceous evidence and collaboration boundary

- Existing project discussion: [current provisional problem framing](https://github.com/nilansa/ire-project-2026), earlier ANN selection controls, and separate local final-report draft. These are **not** independent evidence that this new policy improves retrieval.
- On 15 Nov 2026, an existing Claude Science P1 review marked a **different DISCo + NashANN + uncertainty hybrid** as PIVOT / incremental at best, not experimentally validated. Do not turn that into an endorsement of this proposal.
- The original 50k MS MARCO/ANCE historical pilot has documented score, evaluator and cohort provenance gaps in earlier logs. **Do not reconstruct or silently reuse unverified historical teacher scores**. A new public benchmark must have its own exact manifest and qrels.
- **Requested for Claude Science:** independently stress-test the matrix against TOuR/ReFIT/FIRST/bi-metric; identify missing closer papers; review dataset nullability and oracle definitions; pre-register cost-matched Stage 0–2 comparisons; run only bounded, authorized and auditable work. Do not modify the retriever or reranker weights. Label claims as SOURCE / PROJECT HISTORY / HYPOTHESIS / MEASURED and leave unsupported gains blank.
- Record further decisions and actual run links as dated additions rather than changing this v0.1 retrospectively; preserve failures.

## 8. Evidence anchors and source provenance

**From supplied project files:** Microsoft Research proposal (post-training, data selection, loss, repeated reward calls); 2024 BigANN practical-vector-search paper + presentation/summary (ANN/index effectiveness-efficiency, not an independently judged reranking dataset); original IRE project brief requests literature, gap and executable plan.

**Verified external primary sources to start the canonical survey:**

1. [TOuR — Sung et al., Findings ACL 2023](https://aclanthology.org/2023.findings-acl.354/) — gradient-based test-query adaptation from cross-encoder feedback.
2. [ReFIT — Reddy et al., arXiv 2023](https://arxiv.org/abs/2305.11744) — fixed-weight inference feedback into query embedding plus new retrieval.
3. [FIRST — Reddy et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.491/) — efficient listwise feedback and retrieval application; training of FIRST itself must not be mislabeled training-free.
4. [A Bi-metric Framework — Xu, Silwal & Indyk, ICML 2026](https://proceedings.mlr.press/v306/xu26au.html) — efficient proxy/expensive-metric search, closest systems/theory baseline.
5. [ColBERT-PRF — Wang et al., ACM TWEB 2023](https://doi.org/10.1145/3572405) — training-free embedding feedback and reretrieval.
6. [RRF — Cormack et al., SIGIR 2009](https://doi.org/10.1145/1571941.1572114); [HyDE — ACL 2023](https://aclanthology.org/2023.acl-long.99/); [Query2doc — EMNLP 2023](https://aclanthology.org/2023.emnlp-main.585/).
7. [DISCo — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ded182245f8e4db3963d2027174133d-Abstract-Conference.html) and [NashANN — 2026](https://arxiv.org/abs/2602.08742) — fundamentally set-coverage/diversity objectives, not proven teacher-rank alignment.
8. [TREC DL 2020 official relevance guidelines](https://trec.nist.gov/data/deep2020.html) — passage grade 1 ("related") is **not** binary relevant; threshold at least 2 in that protocol.

**Source-check status:** The claims in this version follow the cited abstracts/available primary descriptions, **not a completed full-paper systematic review or reproduction**. Recheck all quantitative statements at full text before citing a scientific report. No Ceous improvement numbers are asserted here.
