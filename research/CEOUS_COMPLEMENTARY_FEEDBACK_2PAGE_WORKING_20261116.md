# Ceous: Can Complementary Retrieval Improve TTT-Embed?
**A feedback-selection hypothesis grounded in our replication results**  
Nilanjan Sarkar · Raman Rayaprolu · Team Ceous · IIIT Hyderabad · IRE M26CS4.406 · 16 November 2026  
**Status:** two-page working research proposal; the complementary selector is not yet experimentally validated.

## Abstract
A fast dense retriever supplies candidate documents, but an expensive reranker can judge only a limited number of query–document pairs. The Microsoft Research post-training challenge asks how to use such scarce feedback effectively. We investigate an inference-time alternative: leave embedding-model weights and the document index fixed and learn a reusable query-side residual, following TTT-Embed. Our SCIDOCS ReFIT reproduction and FiQA2018 task-wise TTT-Embed cross-fit do not show reliable held-out gains. We therefore propose a controlled test of whether complementary candidate retrieval—rather than conventional dense top-*k* selection—supplies more informative rewarded pairs and produces a residual that transfers better. This selector has not yet been validated.

## 1. Research motivation and trajectory
Our original plan was to post-train a retriever using expensive teacher feedback and to study the role of ANN candidate selection. An MS MARCO 50,000-passage diagnostic found different training pairs from HNSW and IVF-Flat but no improvement over untouched ANCE: MRR@10 0.8908 (raw), 0.8667 (HNSW-teacher), 0.8559 (IVF-teacher); one seed, positive-enriched subset. This prompted a narrower question: can strong reranker signals improve search without retraining encoders or rebuilding the corpus index? TOuR and ReFIT already perform per-query updates. TTT-Embed (Chen & Wu, 2026) further reuses a small learned output-space vector across queries, tasks, or globally. Its controlled protocol fixes the rewarded pairs, leaving their acquisition policy as a separate question.

## 2. TTT-Embed baseline and precise gap
For query \(q\) and document \(d\), frozen encoders return normalized vectors \(x_q\) and \(z_d\). The fast retriever ranks by \(x_q^\top z_d\). A reranker (default in TTT-Embed: **Qwen3-Reranker-4B**) scores a chosen pair \((q,d)\), yielding scalar \(r(q,d)\); one scored pair costs one reward. Teacher and student probabilities on the same candidate slate are \(p_T=\operatorname{softmax}(r/\tau_T)\) and \(p_S(v)=\operatorname{softmax}(\operatorname{unit}(x_q+v)^\top z_d/\tau_S)\). For sharing group \(g\), TTT-Embed learns

\[
v_g^* = \underset{v}{\arg\min}\; \sum_{q\in\mathcal F_g} w_q\,\mathrm{KL}(p_T(q)\Vert p_S(q;v)) + \lambda\lVert v\rVert^2, \qquad
x'_q=\operatorname{unit}\!\left(x_q+\frac{n_g}{n_g+1}v_g^*\right).
\]

Global-wise uses one vector across tasks; task-wise uses one per task; query-wise uses one for each rewarded query. TTT-Embed reports best average scope at low/medium/high normalized reward budgets \(b=B/N\): global \(b=0.01\) (**+1.23 nDCG@10 points**), task \(b=10\) (**+6.36**), query \(b=100\) (**+8.36**), across five embedding models and 15 MTEB tasks. These are the authors' results, not ours; query-wise is strongest at high budget.

## 3. Proposed complementary-feedback mechanism
Let \(C_0(q)\) be top-\(B\) dense candidates, \(C_c(q)\) a complementary pool from BM25, ColBERT token-level retrieval, or a DISCo-inspired coverage search, and \(U(q)=C_0(q)\cup C_c(q)\). Choose exactly \(k\) distinct documents \(S_\pi(q)\) from \(U(q)\) using **only pre-teacher signals**, e.g., query-token coverage, retriever disagreement and redundancy; then purchase precisely \(k\) teacher judgments. Feed these scores to the original, unchanged TTT-Embed objective:

\[
S_\pi(q)=\underset{S\subseteq U(q),|S|=k}{\arg\max}\;
\left[\lambda_c\mathrm{Coverage}(q,S)+\lambda_d\mathrm{Disagreement}(q,S)-\lambda_r\mathrm{Redundancy}(S)\right].
\]

**Hypothesis:** at equal total teacher-scoring budget and accounted retrieval work, complementary \(S_\pi\) improves **held-out** nDCG@10 over dense-top-\(k\) feedback because the learned vector better captures relevance directions absent from redundant initial slates. This is a falsifiable selection hypothesis, not a claim that diversity automatically predicts relevance or that combining named papers is novel.

## 4. Completed Ceous measurements (not directly comparable)
| Experiment | Dataset and feedback | Raw → adapted | Interpretation |
|---|---|---|---|
| **SCIDOCS ReFIT**, frozen Contriever-MSMARCO + MiniLM-L6 cross-encoder | 1,000 queries, 25,657 docs; 125 scored pairs/query, 100 used for query updates | **Recall@100 37.8067% → 37.8717%**; Δ +0.065 pp; 95% CI [−0.180,+0.325] pp | Small, statistically inconclusive. Published ReFIT 40.1% not reproduced. **Not query-wise TTT-Embed.** |
| **FiQA task-wise TTT-Embed**, frozen Qwen3-Embedding-0.6B + Qwen3-Reranker-4B | 648 queries, 57,638 docs; 5 folds, 80% feedback/20% unscored evaluation; 10 rewarded pairs per training query | **nDCG@10 0.46722 → 0.46313**; Δ −0.00408; 95% CI [−0.01072,+0.00262] | Every fold improved in-sample; no reliable held-out transfer. One exposed task, not paper's 15-task replication. |

SCIDOCS is a per-query ReFIT reproduction scored with Recall@100, whereas FiQA is task-wise TTT-Embed with nDCG@10. They use different encoders, teachers and costs. **Neither tests complementary pair selection.**

## 5. Datasets, models, controls and falsification
TTT-Embed's paper evaluates 15 English MTEB retrieval tasks (12,263 test queries, 1,661,208 documents across task-specific corpora) and five frozen embedding models: **Qwen3-Embedding-0.6B**, **Qwen3-Embedding-4B**, **EmbeddingGemma-300M**, **Gemini Embedding 1** and **Gemini Embedding 2**. Default teacher: **Qwen3-Reranker-4B**, using \(P(\text{Yes}\mid q,d)\). Our first controlled study should be FiQA2018 (648 queries/57,638 documents), followed by an independently held-out multi-evidence task such as HotpotQA-HN (1,000 queries/225,621 documents). Pin dataset revisions before claiming fidelity.

Match rewarded query IDs and \(k\) selected pairs/query when comparing candidate selection, holding embedding vectors, reranker, prompt, temperatures, optimizer, residual scope and full-corpus search fixed. Baselines: dense-top-\(k\), random/rank-stratified, dense MMR, BM25/RRF, ColBERT candidate pools, and faithful DISCo coverage if available (else label an approximation). Count extra candidate-retrieval probes, indexing, duplicates, latency and teacher costs separately. **ColBERT** uses independent token-vector encodings with MaxSim and is *not* a joint cross-encoder; **DISCo** maximizes *collective* token coverage, which is *not* the same as relevance. Do not add ColBERT vectors directly to the single-vector Qwen residual.

For held-out transfer, learn only on an acquisition split and evaluate **unscored** queries without teacher calls; use independent qrels only for measurement. Report nDCG@10, Recall@K, judged relevant documents gained outside the original candidate pool, teacher pairs acquired, unique candidates, search probes, p50/p95 latency and paired confidence intervals. Compare cost-matched direct reranking and deeper dense retrieval. Pre-register a meaningful DEV threshold (e.g., +0.01 absolute nDCG@10 over strongest cost-matched baseline) and preserve negative outcomes.

## 6. Claim boundary
TTT-Embed already establishes reusable output-space state in its tested settings; TOuR/ReFIT already optimize query embeddings from teacher scores; GQR uses complementary-retriever feedback in visual retrieval, and Q-REACT studies query-dependent shared residuals. Thus **“DISCo + TTT-Embed” alone is not demonstrated novelty**. Our open empirical question is whether **better-chosen purchased pairs** improve a fixed shared correction at equal supervision and complete search cost. Current Ceous results motivate testing this question; they do not answer it.

## Selected references
1. Microsoft Research, [Efficient and accurate post-training of retrieval models](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/).
2. Chen and Wu, [TTT-Embed](https://arxiv.org/abs/2608.12569), arXiv:2608.12569 (2026), Sections 2–4 and Appendix 12.
3. Reddy et al., [ReFIT](https://arxiv.org/abs/2305.11744), arXiv:2305.11744 (2023).
4. Khattab and Zaharia, [ColBERT](https://arxiv.org/abs/2004.12832), SIGIR 2020.
5. Nair et al., [DISCo](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ded182245f8e4db3963d2027174133d-Abstract-Conference.html), ICLR 2026.
6. [Ceous public research repository](https://github.com/nilansa/ire-project-2026) and original private FiQA cross-fit run ledger (not published publicly).
