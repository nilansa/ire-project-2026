---
id: ire-paper-disco-iclr2026
type: research-paper-analysis
title: "A Dense Subset Index for Collective Query Coverage"
short_name: DISCo
authors: [Kartik Nair, Pritish Chakraborty, Atharva Tambat, Indradyumna Roy, Soumen Chakrabarti, Anirban Dasgupta, Abir De]
venue: ICLR 2026
year: 2026
arxiv_id: null
paper_url: "https://openreview.net/forum?id=cUdODCFjUM"
published_pdf: "https://proceedings.iclr.cc/paper_files/paper/2026/file/6ded182245f8e4db3963d2027174133d-Paper-Conference.pdf"
code_url: "https://github.com/structlearning/DISCo"
research_axis: collective-multivector-subset-retrieval
keywords: [dense-retrieval, submodular-coverage, multi-vector-index, set-selection, query-facets, random-hyperplanes, ANN, RAG, candidate-generation]
related_papers: [ire-paper-ance-2007-00808, ire-paper-nashann-2602-08742]
source_status: primary-paper-reviewed
review_date: 2026-11-15
project: IRE Ceous / Microsoft Research efficient post-training of retrieval models
---

# DISCo: indexed collective coverage instead of independent top-k

> **Central question.** If a question requires *several complementary pieces of evidence*, how can a retriever efficiently select a set of documents that jointly covers its information needs rather than selecting highly similar, redundant documents?

**Source convention.** **[Paper]** indicates a claim or measurement in the ICLR 2026 DISCo paper. **[Analysis]** indicates our own explanatory example, limitation or possible use in Microsoft Research's retrieval post-training problem. Distinguish the paper's approximation theorem for its **ideal randomized greedy method** from empirical behavior of its complete ANN index and pruning pipeline.

## 1. The one-paragraph contribution

Traditional top-K retrieval scores each candidate separately and returns documents with the largest individual similarities. For a multi-faceted query, multiple top hits may cover the **same** aspect while another crucial aspect is absent. DISCo proposes a **collective set coverage** score over the *union of token/atom embeddings* in selected documents. The natural exact greedy optimization is computationally infeasible on large corpora. DISCo contributes a lifted-vector / random-hyperplane surrogate for each *marginal gain*, allowing a multi-vector ANN system to retrieve promising **complementary** documents iteratively. A staged index performs coarse-to-fine pruning and finally checks selected candidates with the true marginal coverage score.

**It is a test-time set retrieval/indexing method, not a retriever post-training recipe.** Its natural research use for IRE is **candidate construction and teacher-data selection**—but that extension must be evaluated.

## 2. Why this was built: the redundancy failure

### 2.1 Independent MaxSim is not a set objective

**[Paper, Eq. 1–2]** A multi-vector retriever such as ColBERT represents a query as token/atom vectors \(Q=\{q_1,\ldots,q_m\}\) and document \(c\) as embeddings \(X_c=\{x_{c,1},\ldots\}\). The usual *independent* document score is
\[
\operatorname{MaxSim}(Q,X_c)=
\sum_{q\in Q}\max_{x\in X_c}q^\top x.
\]
Top-K search independently maximizes this quantity for each document. Because it gives each selected document fresh credit for covering the same query term, a collection of near-duplicates may dominate.

**Toy multi-hop example [Analysis].** Query: “Where did the chemist who invented process X study?” Evidence A identifies process X's inventor. Evidence B names where that inventor studied. Several passages repeating A can rank higher individually than B. Top-K might return A, A', A'' but not B. A collective objective should prefer A+B because B supplies new information.

### 2.2 Why plain diversity heuristics are insufficient

“Different documents” is a weak proxy: two pages might look dissimilar at surface level but both cover only fact A; two textually similar pages can include the only complementary fact. DISCo focuses on **newly covered query components**, not metadata category counts or arbitrary embedding distance between documents.

### 2.3 Why straightforward submodular optimization is impractical

A monotone submodular coverage objective invites greedy selection, which has a classic \(1-1/e\) approximation guarantee (under standard assumptions). But greedy must evaluate the gain of almost every corpus item every round.

**[Paper, Section 3.1]** MS MARCO has roughly **8.8 million passages**. Choosing K=10 items naïvely requires about **88 million** gain computations for a single query. The paper uses the thought experiment that even 1 ms per gain would imply over 24 hours per query. The challenge is therefore not merely *which set objective*, but **how to make marginal-gain search indexable**.

## 3. Exact objective and marginal gain

### 3.1 Union coverage

Let \(S\) be a set of retrieved document IDs. Define:
\[
F(S,Q)=\sum_{q\in Q}\max_{x\in\bigcup_{c\in S}X_c}q^\top x,
\qquad |S|\le K.
\]
The inner \(\max\) over the **union** means that once a query atom has been well matched by an existing selected item, adding another identical item contributes little or nothing for that atom.

**[Paper]** This is a soft facility-location style set coverage function, monotone submodular with the stated similarity/initialization conventions. Greedy iteratively chooses
\[
c_k=\arg\max_{c\notin S_{k-1}}
\Delta(c\mid S_{k-1},Q),\quad
\Delta(c\mid S,Q)=F(S\cup\{c\},Q)-F(S,Q).
\]
For exact greedy, the classical \(1-1/e\approx0.632\) lower bound relates its value to the optimal K-item set.

### 3.2 Uncovered “residual” query information

**[Paper, Proposition 1 / Eq. 3]**
\[
\Delta(c\mid S,Q)=
\sum_{q\in Q}\max_{x\in X_c}
\left[q^\top x-F(S,\{q\})\right]_+,
\]
where \([z]_+=\max(0,z)\), and \(F(S,\{q\})\) is the best score that query atom q has received so far. (The empty-set baseline needs consistent treatment in an implementation.)

A candidate gets credit only when it **beats the current best support** for an atom. That is the precise mechanism that prevents redundant selection.

**Three-atom toy calculation [Analysis].** A query asks for atoms A,B,C. First document covers [0.9,0.1,0.1]. Second document covers [0.85,0.8,0.2]. Third covers [0.8,0.1,0.95]. Independent sums are 1.1,1.85,1.85; a conventional top-2 might return 2+3, which also covers all three well. But after selecting document 2, doc 3 contributes 0.75 primarily for C, whereas another near-duplicate of doc 2 could independently score 1.85 but contribute almost 0. The **marginal gain** is what determines complementarity, not the standalone sum.

## 4. The key trick: make coverage searchable by ANN

### 4.1 Why Eq. 3 is not directly ANN-friendly

The hinge function \([q^\top x - F(S,\{q\})]_+\) is nonlinear, and its threshold depends on the already selected set \(S\), which changes every greedy round. Standard ANN indexes expect a fixed precomputed corpus representation and a query-time similarity. Re-encoding the whole corpus after each selected document would be untenable.

### 4.2 Lift the state dependency into the query

**[Paper, Eq. 4–5]** Augment query and corpus atoms from d to d+1 dimensions:
\[
\hat q_S=[q;F(S,\{q\})],\qquad \hat x=[x;-1].
\]
Then
\[
\hat q_S^\top\hat x = q^\top x-F(S,\{q\}),
\]
and
\[
\Delta(c\mid S,Q)=
\sum_{q\in Q}\max_{x\in X_c}[\hat q_S^\top\hat x]_+.
\]
This is the conceptual turning point. **The corpus representation remains indexable and query-agnostic**, while the changing “already-covered” state is encoded in \(\hat q_S\). At each round we update the query representation rather than rebuilding corpus embeddings.

### 4.3 Random-hyperplane feature map approximates the hinge

**[Paper, Eq. 6–8 / Theorems 2–3]** For a Gaussian random hyperplane \(w\sim\mathcal{N}(0,I)\), define
\[
\Phi_w(u)=\frac{1}{\sqrt2}[u;\operatorname{sign}(w^\top u)\,u].
\]
Now compute a **linear dot product** between transformed query/corpus vectors. Depending on whether the random-hyperplane signs match, the inner product becomes either the original \(u^\top v\) or zero. The paper shows this realizes \([u^\top v]_+\) with probability at least 1/2 under its stated setup.

Sample **R independent hyperplanes** and max-pool their transformed dot products. Increasing R reduces the probability that the hinge is represented incorrectly; this creates an ANN-compatible estimate of the marginal gain.

**Subtlety:** this is not the conventional Johnson–Lindenstrauss dimensionality reduction story. It is an intentionally constructed **feature lift** using random sign agreement to approximate a nonlinear *positive-part inner product*. With \(\hat q,\hat x\in\mathbb R^{d+1}\), mapped features lie in \(\mathbb R^{2(d+1)}\).

### 4.4 Mathematical guarantee and what it does not guarantee

**[Paper, Theorem 4]** For the idealized greedy algorithm using the randomly approximated marginal gains, if \(R\ge \log(|Q|/\delta)\), its *expected* coverage can achieve
\[
\mathbb E[F(S_K,Q)]\ge(1-1/e-\delta)\,F(S^\star,Q).
\]
Read this as a theorem about the **specified approximation and selection algorithm**. The deployed DISCo pipeline also uses limited ANN probes, quantized residuals and candidate truncation. Such engineering may introduce additional approximation errors and cannot automatically inherit an identical end-to-end guarantee without separate assumptions. Evaluate it empirically.

## 5. DISCo's actual index design

**[Paper, Section 3.4 / Algorithm 3]** DISCo is built on ColBERT-family multi-vector search practices, adapted to the new objective:
1. Start with corpus token embeddings \(x\) and add coordinate -1 to get \(\hat x\).
2. For each projection replica \(r=1,\ldots,R\), sample \(w_r\) and transform \(\Phi_{w_r}(\hat x)\).
3. Cluster transformed vectors into B centroids (IVF).
4. Encode each token vector as a centroid ID plus compressed residual \(\Delta\).
5. Maintain **inverted lists** from centroid to documents and **forward lists** from document to its token/centroid/residual codes.
6. Repeat the index structure per projection replica.

The multi-vector representation is important: a single global embedding of a paragraph can lose which individual query atoms it supports.

### 5.1 Query processing, round by round

**[Paper, Section 3.4 / Algorithm 4]** There are K rounds. For each round, with current selected set S:
1. **Update query atom states** \(\hat q_S\), subtracting what existing S already covers.
2. **Per-replica coarse IVF probing:** nearest centroids select candidate postings for transformed query atoms.
3. **Per-replica centroid pruning:** use cheap centroid-level MaxSim-like scores; retain top n.
4. **Pool replicas early:** union the candidate sets across projections to avoid missing candidates valuable in only some representations.
5. **Fine-grained centroid scoring:** pool contributions across replicas; retain roughly n/4.
6. **Residual scoring:** reconstruct approximate projected token vectors from centroid + residual, score survivors and retain top n'.
7. **Full-precision true coverage rerank:** for the small final set, compute original \(F(c\mid S,Q)\) and choose the largest-gain item.
8. **Update** \(S\leftarrow S\cup\{c\}\), recompute coverage state, repeat.

The “reranking” in step 7 is an **exact marginal-coverage check** among surviving candidates, *not necessarily an LLM cross-encoder relevance reranker*. If integrating into a modern RAG architecture, an external cross-encoder can be applied after or during candidate selection, but that is a further design.

### 5.2 Simplified pseudocode [Analysis]

~~~python
index = build_multivector_ivf_replicas(corpus_tokens, random_planes=R)
S = []
coverage = {query_atom: empty_baseline for query_atom in Q}
for round_id in range(K):
    query_state = lift(query_atoms=Q, coverage=coverage)
    candidates = union_replica_ivf_probes(index, query_state)
    candidates = centroid_prune(candidates, query_state, top_n=n)
    candidates = cross_replica_fine_prune(candidates, top_n=n // 4)
    candidates = residual_rank(candidates, top_n=n_prime)
    choice = max(candidates, key=lambda c:
                 exact_marginal_coverage_gain(c, S, Q))
    S.append(choice)
    coverage = update_covered_scores(coverage, choice)
return S
~~~

This is pedagogical: production code has several more details and must exclude selected IDs on later rounds.

## 6. Empirical evidence and benchmark design

### 6.1 Datasets, baselines, metrics

**[Paper, Section 4]** Seven retrieval datasets: MS MARCO, HotpotQA, FEVER; LoTTE Pooled, Technology, Writing and Science. Paper evaluates multi-vector embeddings obtained with **bert-base-uncased** in the described comparison.

Methods:
- **Exact Greedy**, **Lazy Greedy**, **Stochastic Greedy**, **Lazier-than-lazy Greedy** (submodular selection baselines).
- **PLAID**, **MUVERA**, **WARP** (efficient search methods optimizing independent MaxSim retrieval, not this collective set objective).
- **DISCo** (indexed approximated-marginal-gain selection).

Primary quality: mean **coverage \(F(S_K,Q)\)** over queries and how quickly it improves with small K. Efficiency definition is a ratio against **Exact Greedy at K=10**: \(t_{\mathrm{exact\ greedy},10}/t_{\mathrm{method},K}\). Do not compare plotted “efficiency” directly to absolute QPS reported in NashANN.

The main coverage experiments use K=10; **the HotpotQA gold-label comparison uses K=2** because there are two gold supporting items. This difference matters when reproducing Table 3.

### 6.2 Gold evidence retrieval: concrete numbers

**[Paper, Table 3, HotpotQA with two gold supporting items]**

| Method | Error(F) ↓ | MAP ↑ |
|---|---:|---:|
| Exact Greedy | 0.69 | 0.83 |
| Lazy Greedy | 0.69 | 0.83 |
| Stochastic Greedy | 2.77 | 0.19 |
| Lazier-than-lazy Greedy | 3.09 | 0.14 |
| PLAID | 1.30 | 0.81 |
| MUVERA | 3.55 | 0.49 |
| WARP | 1.51 | 0.77 |
| **DISCo** | **0.68** | **0.84** |

Here Error(F) is deviation from coverage achieved by the **gold supporting set**; lower is better. MAP measures retrieval alignment with gold items; higher is better. The authors report DISCo retrieves all required items within roughly top 2–3 ranks on many HotpotQA queries, comparable with the costly deterministic greedy methods (Figure 4).

**Caution:** the small numeric improvement over exact greedy (0.68 vs 0.69) is an empirical difference on this dataset, not evidence of a mathematical violation of exact-greedy approximation theory (which compares to the *optimization optimum*, not to gold-label MAP or an error-to-gold metric).

### 6.3 Runtime: what is actually faster?

**[Paper, Figure 2 / Section 4.1]** On MS MARCO DISCo is **at least 100× faster than the *evaluated greedy variants*** in the paper's measured scenario. It reaches coverage close to deterministic greedy with much less per-query compute by avoiding global per-round scans.

**Do not overclaim:** PLAID, MUVERA, WARP are **also indexed and efficient**. The paper's major benefit relative to them is **higher collective coverage at the measured operating points**, not necessarily that DISCo always outruns them in wallclock latency. A useful comparison must show a **Pareto curve of coverage vs p50/p99 latency** including pre-processing and index memory.

### 6.4 Ablations and mechanism checks

**[Paper, Figure 5]** Increasing the number of random projection replicas R improves approximate-marginal-gain quality toward exact greedy; in shown experiments even R≈5 offers close coverage trajectories.

**[Paper, Figure 6]** Early pooling (merging replica candidates before later refinement) outperforms late pooling in both coverage and efficiency in reported settings; the reason is that the approximate **pooled** marginal objective is better reflected before pruning.

**[Paper, Algorithm 4]** Hyperparameters include R (replicas), B (IVF centroids), n (per-replica candidates), n' (full-rescore pool), τ (threshold), and K (documents). All can change quality, index footprint and latency. The results are not free of implementation tuning.

## 7. End-to-end cost ledger, beyond a quoted speedup

A serious deployment measurement should include:
- Offline token embedding computation: \(O(\sum_c |X_c|)\) encoder inference, tokenizer and text truncation.
- R projection replicas and their **extra index storage**, quantization codes, centroids, postings, forward index.
- Index-build latency vs corpus size and index-maintenance cost on added/changed documents.
- Per-query K sequential rounds; index probes and centroid/residual scoring each round.
- Extra *full-precision* scoring on n' items per round.
- Multi-vector memory traffic and random-access I/O.
- Final cross-encoder relevance reranking, if integrated, measured separately.
- Downstream answer-generation token cost and hallucination/faithfulness accuracy.

**[Analysis]** K dependent latency and R-index amplification are the main potential operational weaknesses; a low-K setting such as K=2–5 might be an attractive first experiment if supporting facts are few. DISCo is *not* a single top-k ANN lookup; its K rounds are sequential and stateful.

## 8. What the method does not automatically solve

1. **Coverage ≠ factual entailment.** Max token similarity is a learned proxy; it can reward superficial token matches, false evidence or misleading paraphrases.
2. **No explicit metadata fairness.** A set may cover many query atoms using only one seller/topic; NashANN targets a different notion of diversity.
3. **No encoder post-training.** DISCo chooses items with a fixed embedding model; teacher-based updates require an added learning step.
4. **Rare evidence can be pruned.** Missing the right item in early ANN candidate generation makes exact final selection ineffective.
5. **Index expansion and online updates** cost memory/maintenance; paper flags adaptive selection/weighted coverage/dynamic updates as future directions.
6. **Objective depends on query atomization.** Learned token embeddings may not correspond cleanly to distinct real-world facts.
7. **The theorem addresses a surrogate method.** Actual quantized ANN index/truncation must be audited empirically.
8. **Gold-label metrics matter.** High F-score alone is insufficient if all gold supporting documents are not jointly retrieved.

## 9. Direct bridge to Microsoft Research post-training retrieval

Microsoft asks how to select queries/documents for **expensive LLM reward feedback**, choose a loss, and iterate efficiently. DISCo's contribution is a **candidate generation objective that rewards marginal new information**. It can be used at two distinct stages:

### Proposal A: teacher-data acquisition, not just test-time retrieval

At student checkpoint \(\theta_t\):
1. ANN-mine candidate documents from full corpus (ANCE-like).
2. For each training query, select a diverse evidence *set* via DISCo using residual query-atom coverage.
3. Have LLM cross-encoder score **only selected** candidates (pointwise/pairwise/listwise).
4. Distill into retriever; refresh corpus index when student changes.
5. Compare with the same teacher budget allocated to top-similarity and random samples.

**[Analysis]** Hypothesis: relative to scoring 20 near-duplicate high-similarity passages, spending 20 scores across complementary facts may give more training signal for multi-hop RAG. But the student may also need hard near-duplicate negatives to improve fine discrimination. The acquisition policy should **mix** redundancy/challengingness with coverage; too much coverage-only selection could produce negatives that are individually easy.

### Proposal B: efficient output candidate generation before cross-encoder reranking

First-stage DISCo returns a small collectively covering set; reranker judges relevance and/or answer support. Compare:
- independent top-K then cross-encoder,
- DISCo K-set then same cross-encoder,
- union top-K + DISCo candidates then cross-encoder,
- learned retriever post-trained on DISCo-constructed feedback.

Important: if the reranker itself chooses items independently and truncates the pool, DISCo's complementarity may be destroyed. Preserve set-wise coverage during final selection or test this explicitly.

### Proposed loss integration [Analysis]

A set-level downstream objective might combine teacher scores with coverage:
\[
\mathcal{J}(S)=
\lambda\sum_{d\in S}r_{\mathrm{teacher}}(q,d)
+(1-\lambda)F(S,Q),
\]
with calibration and scale normalization. This is our suggestion, **not** DISCo's published training loss. It can produce redundant gradients if an LLM teacher already rewards complementary evidence; ablate components.

## 10. Falsifiable research questions and proposed measurements

**H1 (coverage):** at equal first-stage latency and retriever architecture, DISCo raises *complete supporting-evidence recall* on HotpotQA and 2WikiMultiHopQA compared with independent top-K.

**H2 (teacher efficiency):** at equal number of teacher-scored pairs, coverage-aware acquisition improves held-out multi-hop answer F1/EM compared with top-ANN mining.

**H3 (compute):** multi-round and R-replica cost may outweigh coverage gains at large K or small corpora. Plot Pareto curves rather than assume index efficiency from theorem.

**H4 (hybrid):** NashANN (metadata diversity) and DISCo (query-facet coverage) improve **different slices**; the hybrid improves macro-average topics without reducing full-evidence support.

**H5 (false positives):** high coverage F but low true-support recall indicates semantic/token-level mismatch; audit human/LLM evidence entailment.

Suggested metrics and controls:
| Target | Metric | Required caveat |
|---|---|---|
| First-stage recall | Recall@K of each gold document | Independent relevance |
| Multi-hop retrieval | **All-support Recall@K** | Requires every supporting document |
| Coverage objective | Mean F(S,Q) | Proxy, not answer correctness |
| Reranking | nDCG@10 / MAP | Include same reranker |
| RAG | EM/F1, answer support, faithfulness | Identical generator/prompts |
| Efficiency | p50/p95/p99 query time, QPS | Separate index build and GPU footprint |
| Teacher budget | judged pairs, token cost, $ | Equal budget across selectors |
| Selection robustness | R,n,n',K, ANN recall, seeds | Report missing-gold-at-candidate-stage |

## 11. Reproduction plan / danger checks

- Obtain official code and reproduce a **small** HotpotQA gold-set MAP+Error(F) before large MS MARCO.
- Verify token embedding implementation and normalization; preserve the **same** atom vectors across all baselines.
- Build an exact greedy oracle on a corpus subset (e.g., NFCorpus), compare it to surrogate greedy before introducing ANN index truncation.
- Separate errors from random-hyperplane approximation, centroid quantization, IVF pruning and final scoring; this matters more than one aggregate speed number.
- Test varying numbers of replicas R and shortlist sizes; log index size bytes and index-build wall time.
- Log each round's selected document, best support per query atom, marginal gain and whether gold documents remained in each prune pool.
- Run with/without cross-encoder reranking; compare final **set-wise** selection to independent rerank.
- Store reproducible seeds, model checkpoint, dataset revision, exact corpus IDs and query IDs.
- Avoid labeling every high-coverage item as a relevant positive; teacher/human relevance judgment must remain separate.

## 12. Notebook / Redis schema

Document key: **paper:disco:iclr2026**. Suggested chunks:
- failure_of_independent_topk
- coverage_objective
- marginal_gain
- lifted_query
- random_hyperplane_features
- theorem_scope
- index_architecture
- online_pruning
- benchmarks_and_metrics
- runtime_and_tradeoffs
- post_training_hypotheses

Claim records: {paper_id, section_id, claim_type, text, equation_id, source_pdf, table_anchor, metric, baseline, dataset, value, unit, extractor_version}. Distinguish **exact greedy, surrogate greedy, indexed DISCo** as separate method variants. Preserve K=2 HotpotQA vs K=10 primary coverage experiments.

## 13. Primary references / reproducible anchors

- ICLR 2026 published PDF: https://proceedings.iclr.cc/paper_files/paper/2026/file/6ded182245f8e4db3963d2027174133d-Paper-Conference.pdf
- OpenReview record: https://openreview.net/forum?id=cUdODCFjUM
- Repository: https://github.com/structlearning/DISCo
- **Section 3.1 / Eq. 2 / Algorithm 1**: collective coverage, submodularity, cost of exhaustive greedy.
- **Section 3.3 / Proposition 1 / Eq. 3–8 / Theorems 2–4**: marginal gain, lifted vectors, random hyperplanes, expected bound.
- **Section 3.4 / Algorithms 3–4 / Figure 1**: multi-vector IVF construction and staged query processing.
- **Section 4 / Figure 2**: coverage vs efficiency, including MS MARCO greedy-baseline speedups.
- **Section 4 / Table 3 / Figure 4**: HotpotQA all-support evidence alignment.
- **Section 4.1 / Figures 5–6**: projection count and early-vs-late pooling ablations.
- **Section 5**: future work on alternative similarity, adaptive retrieval and dynamic updates.
- Contextual Microsoft Research challenge: https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/

**Bottom line:** DISCo makes **jointly useful retrieval sets** computationally practical by expressing “what is still uncovered?” as a changing query state compatible with an ANN index. Its IRE opportunity is to enrich cheap candidate pools before expensive teacher reranking/post-training, while measuring *all required evidence retrieved* at a fixed teacher-call and end-to-end latency budget.
