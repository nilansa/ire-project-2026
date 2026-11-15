---
id: ire-paper-nashann-2602-08742
type: research-paper-analysis
title: "Welfarist Formulations for Diverse Similarity Search"
short_name: NashANN
authors: [Siddharth Barman, Nirjhar Das, Shivam Gupta, Kirankumar Shiragur]
venue: ICLR 2026
year: 2026
arxiv_id: "2602.08742"
paper_url: "https://arxiv.org/abs/2602.08742"
published_pdf: "https://openreview.net/pdf/cfafb6e42fa52a198856f231e89c74b0846d8560.pdf"
code_url: "https://github.com/shivi98g/NashANN"
research_axis: diverse-nearest-neighbor-selection
keywords: [diversity, nash-social-welfare, p-mean, approximate-nearest-neighbor, set-selection, multi-attribute, query-latency, candidate-generation]
related_papers: [ire-paper-ance-2007-00808, ire-paper-disco-iclr2026]
source_status: primary-paper-reviewed
review_date: 2026-11-15
project: IRE Ceous / Microsoft Research efficient post-training of retrieval models
---

# NashANN: turning diversity into an economic welfare objective

> **Central question.** How can approximate-nearest-neighbor retrieval return a *set* of relevant results that also fairly covers different categories or stakeholders, without hard-coding the same quota for every query?

**Reading convention.** **[Paper]** is a statement from the authors' ICLR 2026 text or tables. **[Analysis]** is our synthesis or research proposal. Source anchors are listed at the end; arithmetic examples illustrate the mechanism and are not reported empirical results.

## 1. Quick summary

Standard nearest-neighbor search (NNS/ANN) optimizes the similarity of individual retrieved vectors to the query. If one type of result dominates top scores, nearly all k slots might go to it. Hard diversity quotas can prevent dominance, but force irrelevant groups into results even when the query genuinely favors one group.

**[Paper]** *Welfarist Formulations for Diverse Similarity Search* proposes viewing each attribute class as an economic agent, assigning each agent utility from selected items' relevance, and maximizing a **social welfare** function. Nash social welfare balances large total utility with protection against an ignored class. The paper develops single-attribute **Nash-ANN**, multi-attribute extensions and a family of p-mean retrieval objectives.

Its key insight is not “always show one of every category”; it is **let relevance and diversity adapt to each query through the objective**.

## 2. Why this work was built

### 2.1 The structural failure of top-k similarity

**[Paper, Introduction]** ANN indexes excel at quickly returning high-similarity vectors, but a top-k list can be homogenous with respect to metadata (e.g., sellers, product colors, publication categories). That can reduce user choice, representation or information diversity. Relevance-only ranking has no mathematical reason to allocate slots across classes.

**Thought experiment [Analysis].** Query “formal shoes”: ten nearly identical black products from seller A score 0.96–0.92; three slightly lower-scored products from seller B score 0.91–0.88. Standard top-k is overwhelmingly A, even if users value seller variety. Adding a rigid minimum of three B products to **every** query may also be unreasonable when B's products are irrelevant to a different query.

### 2.2 Why fixed quotas are not the solution

A hard-constraint approach demands a fixed count per category, irrespective of category relevance to the current query. It can improve diversity counts while sacrificing similarity. Worse, if category distribution shifts or queries are narrow, quotas must be retuned by hand.

**[Paper]** The economic approach instead uses a **single objective** in which a class that has received little utility has a high marginal incentive, but only if its candidate is still similar enough to justify a slot. This balances two objectives without a universal minimum representation rule.

### 2.3 Why economic fairness is a novel perspective

Nash social welfare is classically the product/geometric mean of agents' utilities; maximizing it discourages extreme inequality. In retrieval, an “agent” is a **metadata attribute** (not necessarily a human being). Its welfare represents relevance allocated to that class. The model thereby translates “diversity” from a handcrafted quota into a principled optimization target.

Crucial distinction: **the agent is an attribute class, not a document or a retriever network**. The welfare objective is applied to the retrieved **set of items**.

## 3. Formal setup and interpretation

Let a corpus \(P\) contain \(n\) vectors \(v\in\mathbb{R}^d\), query \(q\), similarity \(\sigma(q,v)\), selected subset \(S\subseteq P\) with \(|S|=k\), and attribute classes \(D_1,\ldots,D_c\). For a multi-attribute record, an item may belong to several \(D_\ell\).

**[Paper, Definition 1]** Utility of class \(\ell\):
\[
u_\ell(S)=\sum_{v\in S\cap D_\ell}\sigma(q,v).
\]
Nash social welfare with smoothing \(\eta>0\):
\[
\operatorname{NSW}(S)=
\left[\prod_{\ell=1}^{c}(u_\ell(S)+\eta)\right]^{1/c}.
\]
Equivalent optimization (log preserves the ordering):
\[
S^\star\in\arg\max_{|S|=k}\log\operatorname{NSW}(S)
=\arg\max_{|S|=k}\frac{1}{c}\sum_{\ell=1}^c\log(u_\ell(S)+\eta).
\]
The smoothing constant keeps the objective finite if a class has zero utility. Similarity weights and nonnegativity assumptions matter for formal guarantees; do not transplant the theorem unchanged to signed cosine similarities without checking normalization.

### 3.1 Interpret the marginal score, not just the formula

In a single-attribute partition, candidate \(v\in D_\ell\) with similarity \(s\) gives a welfare increment:
\[
\Delta_\ell(v\mid S)=
\log(u_\ell(S)+\eta+s)-\log(u_\ell(S)+\eta).
\]
This is **larger when the same relevance contribution goes to an under-served class**, because log has diminishing returns. Yet a candidate with very small \(s\) may offer less gain than a highly relevant item in a well-served class. That is the adaptive balance.

**Numerical example [Analysis].** Suppose \(\eta=1\), class A currently has utility 2 and class B has 0; two next candidates have similarity 0.9 in A and 0.7 in B. Increment A: \(\log(3.9)-\log(3)\approx 0.262\). Increment B: \(\log(1.7)-\log(1)\approx0.531\). B wins despite slightly lower similarity, **not because of a quota**, but because its marginal utility gain is greater. If B had similarity 0.05, its increment \(\log(1.05)\approx0.049\), so A wins.

### 3.2 p-mean generalization

**[Paper, Section 2]** The authors also consider generalized p-means:
\[
M_p(u_1,\ldots,u_c)=
\left(\frac1c\sum_{\ell=1}^c u_\ell^p\right)^{1/p}
\quad (p\neq 0),
\]
with continuous limit \(M_0\) equal to the geometric mean. Interpretation:
- \(p=1\): arithmetic/utilitarian welfare; relevance-dominant nearest neighbor selection.
- \(p\to0\): Nash geometric mean; compromise.
- \(p\to-\infty\): max–min or egalitarian welfare; emphasize the lowest-utility category.
- Lower p generally raises priority for underrepresented classes; the exact curve depends on data and smoothing.

These are **choices of objective**, not trained model layers.

## 4. Algorithms: why this can run as nearest-neighbor search

### 4.1 Single attribute: per-class ANN + welfare-aware merge

Assume each vector has **exactly one class**, so \(D_1,\ldots,D_c\) partition the corpus. Since within one class utility rises with similarity, obtain the k highest-similarity candidates from each class using a nearest-neighbor oracle. Then iteratively choose the class whose next highest-similarity candidate increases log welfare the most.

**[Paper, Algorithm 1 / Theorem 1]** If each per-class neighbor oracle is *exact*, this procedure yields an **exact Nash-optimal set** with runtime
\[
O(kc)+\sum_{\ell=1}^{c}\operatorname{ENN}(D_\ell,q).
\]
This is *not* a trivial claim for arbitrary metadata overlap; the partition is essential. The paper also proves an \(\alpha\)-approximate Nash result when the neighbor oracle returns suitably \(\alpha\)-approximate per-rank neighbors (Corollary 2). Read the paper's exact definition of \(\alpha\); it is not simply empirical ANN recall@k.

### 4.2 Why the merge is efficient in single-attribute case

Once within-class candidate lists are sorted, any lower-ranked item in one class is dominated by the higher-ranked unused item from the same class for the utility increment. Thus the algorithm only needs to compare the next available item from each of c lists and update one class's utility each round. It is an **adaptive merge of c nearest-neighbor lists**, not a fresh full-corpus search every greedy iteration.

**Pseudocode [Analysis]**
~~~python
lists = {cat: top_k_nearest(query, index_for_cat[cat], k)
         for cat in categories}
utility = {cat: 0.0 for cat in categories}
pointer = {cat: 0 for cat in categories}
selected = []
for step in range(k):
    best = None
    for cat in categories:
        candidate = lists[cat][pointer[cat]]
        score = log(utility[cat] + eta + sim(query, candidate)) \
              - log(utility[cat] + eta)
        if best is None or score > best.score:
            best = (cat, candidate, score)
    selected.append(best.candidate)
    utility[best.cat] += sim(query, best.candidate)
    pointer[best.cat] += 1
~~~
The true implementation must handle empty class lists and ties. This sketch is **not verbatim** from authors' repository.

### 4.3 Approximate oracle variant

Per-class HNSW/DiskANN/another ANN engine can replace exact search. Then theoretical quality follows **only under the paper's strong definition of oracle approximation**; realistic approximate indexes should be benchmarked against exhaustive results to see whether assumptions hold. The separability into class-specific searches creates a potential latency cost proportional to the number of attribute classes.

### 4.4 Multiple attributes: a qualitatively harder problem

If one item belongs to several attributes (e.g., product color *and* seller), selecting it updates utilities for multiple agents. Class lists overlap and the clean optimal merge no longer applies.

**[Paper, Theorems 3–4]** For multi-attribute NaNNS (including \(\eta=1\)), optimization is NP-hard. The paper gives a polynomial-time \((1-1/e)\)-approximation guarantee for the **log-social-welfare objective** via greedy submodular maximization under its stated conditions.

**Critical nuance:** an approximation ratio on **log welfare** is not automatically the same multiplicative ratio on **raw Nash social welfare**. The displayed theorem uses \(\log\operatorname{NSW}\); interpret this carefully, particularly when logs may be negative. Likewise, guarantees in single-attribute exact search should never be asserted for the multi-attribute heuristic.

In practice, multi-attribute search operates over a **bounded candidate pool** before welfare selection. That selection can be high quality but is only as diverse as the pool; **missing category candidates cannot be recovered during the merge**.

## 5. Runtime and engineering contribution

The user-facing reason this paper matters is that it makes relevance-diversity optimization *retrieval-compatible*: it is not necessary to evaluate every possible size-k subset.

However **“ANN-based” does not mean zero overhead**:
- Single-attribute Nash-ANN obtains per-group candidate lists; the cost includes c approximate/exact searches plus \(O(kc)\) selection.
- Multi-attribute instances may need a sizable candidate pool and iterative gain calculations.
- More metadata classes, frequent metadata changes, skewed class sizes or high k may increase costs.
- \(\eta\), p and number of candidate slots define the relevance-diversity frontier.

### 5.1 Fast approximate heuristic: p-FetchUnion-ANN

**[Paper, Appendix F.5, Tables 8–9]** A separate **p-FetchUnion-ANN** heuristic first retrieves a larger shared ANN pool from the union of all classes and then selects with the welfare objective, rather than performing a separate neighbor search per class. It trades stronger per-class coverage guarantees for much higher throughput.

Reported comparisons for \(k=50\), \(p=0\):
| Benchmark | Per-class p-Mean-ANN | p-FetchUnion-ANN | Scope |
|---|---:|---:|---|
| Sift1m-(Clus), QPS | 135.98 | 1442.03 | Approx. 10.6× throughput |
| Sift1m-(Clus), mean latency | 230318 µs | 22170 µs | Approx. 10.4× lower |
| Amazon, QPS | 179.03 | 608.76 | Approx. 3.4× throughput |
| Amazon, mean latency | 178555 µs | 52522 µs | Approx. 3.4× lower |

**Important comparison:** These ratios are **against the paper's own per-class p-Mean-ANN** and specific benchmark setups, *not* “NashANN 10× faster than ordinary ANN” and not directly comparable with DISCo/ANCE timings. FetchUnion can lose diversity if a global top-L pool excludes minority attributes (paper explicitly discusses this tradeoff).

### 5.2 Multi-objective evaluation requires multiple axes

**[Paper, Experiments]** Benchmarks include Amazon, ArXiv, Sift1m and Deep1b-derived vector sets, with real or synthetic attributes. Key measurements include:
1. **Approximation ratio/relevance** relative to standard top-k similarity.
2. **Entropy of selected attributes** as a diversity proxy.
3. Related diversity indices (e.g., distinct attribute coverage).
4. **Query throughput and tail latency**.
5. Sensitivity to p, k, attribute distribution and candidate pool.

Example **[Paper, Appendix Table 2, Amazon k=50]**:
- p-Mean-ANN p=0 relevance approximation ratio ~0.938 ± 0.023 and entropy ~5.058 ± 0.178.
- Unconstrained ANN has ratio 1.000 and entropy ~2.782 ± 0.684.
- A fixed-quota Div-ANN k'=1 baseline has relevance ratio ~0.813 ± 0.053 and entropy ~5.594 ± 0.049.

Interpretation: Nash gains category diversity at a **measurable but sometimes moderate relevance cost**, while hard quotas can gain slightly more diversity with greater relevance cost. Exact tradeoffs are dataset-dependent, and entropy values must be interpreted relative to the number/frequency of classes.

**Do not infer downstream RAG accuracy from entropy alone.** Entropy counts how classes are represented, not whether their selected passages contain answer-bearing facts.

## 6. What this paper establishes and leaves unresolved

**Strongest findings [Paper]:**
- A principled welfare-based notion of diversity can replace per-query fixed quotas.
- Single-attribute problem admits an efficient optimal algorithm using exact nearest-neighbor oracles.
- A formal approximate-oracle guarantee applies in the single-attribute setting.
- Multi-attribute selection is hard; a greedy objective approximation is available.
- ANN-compatible implementations show empirical relevance-diversity benefits and tunable throughput.

**Limitations / unanswered issues [Analysis]:**
- **Ontology dependency:** data must have meaningful categories/attributes; poor labels yield poor “fairness.”
- **Fairness semantics:** equalizing topic/category utility is not the same as fairness for people or sellers, and may not reflect user preference.
- **Rare-class omission:** finite global candidate pools can erase rare categories; per-class indexes may be needed to preserve opportunity.
- **Similarity calibration:** cosine/dot scores across groups must be comparable; systematic class bias can distort social welfare.
- **Smoothing parameter:** \(\eta\) changes incentives and should be tuned without contaminating test results.
- **Overlapping classes:** multi-attribute utility double-counting effects may benefit some documents; category selection can become sensitive to taxonomy granularity.
- **Production scalability:** per-class index storage, metadata update, imbalanced partitions and read amplification need operational evaluation.
- **Post-training distinction:** the method modifies *which candidates are returned*, not directly the encoder's weights.

## 7. Bridge to ANCE and DISCo: three different “diversities”

- **ANCE**: hard-negative diversity in **training** (which errors update the model?). It improves learned representations.
- **NashANN**: group/attribute diversity in **retrieved output** (which classes get utility?).
- **DISCo**: information/facet diversity in **retrieved output** (which *query atoms* remain uncovered?). It reduces redundancy across selected documents.
- **Possible synthesis [Analysis]**: use NashANN to ensure underrepresented *query-document categories* receive expensive reward-model labels, and DISCo to ensure multi-fact evidence is represented; train the student on these samples with ANCE-like adaptive index refresh.

“Diverse” means something different in each objective; do not merge metrics without explicitly defining the target distribution.

## 8. Direct relevance to Microsoft's retrieval post-training questions

### Data selection

**[Analysis]** If cross-encoder scoring is expensive, top student ANN negatives are often clustered. Allocate a reward-scoring budget across topic/domain/category agents using Nash welfare, with candidate relevance as a weight:
\[
\max_{S:|S|\le B}\sum_\ell
\log\Big(\eta+\sum_{(q,d)\in S:\,\ell\in a(q,d)}
w(q,d)\Big).
\]
Here \(B\) is a **teacher-scoring budget** and \(w\) could be uncertainty, teacher-disagreement proxy, or student similarity. This equation is our **proposed acquisition adaptation**, not an objective evaluated in NashANN.

### Loss formulation

Given selected query-doc judgments, pairwise or listwise distillation can be balanced across groups via group-normalized weights. This is a *new learning problem*; NashANN itself does not identify an optimal post-training loss.

### Computational efficiency

Two-stage selection: (1) cheap global ANN fetch, (2) welfare-aware allocation, (3) expensive teacher scores only on shortlisted pairs, (4) cache. Tune pool size L and number of categories c against **teacher cost**, not just retrieval milliseconds.

## 9. Proposed experiments and falsifiable hypotheses

**Hypothesis H1:** Given the same number of teacher calls, Nash-balanced selection improves macro-average retriever quality across rare groups compared with raw student-top-k acquisition.

**H2:** Nash selection may hurt micro-average retrieval if diversity is over-weighted (small p, overly egalitarian objective). Plot Pareto frontier macro-recall vs nDCG/teacher-call.

**H3:** Per-class and FetchUnion implementations differ most in highly skewed datasets; global FetchUnion collapses rare-class coverage at small L, whereas per-class remains more robust but slower.

**H4:** Metadata diversity only helps downstream RAG if metadata aligns with genuine evidence distinctions. Compare category entropy with multi-hop **complete-support recall**; require both.

Minimum experiment matrix:
| Arm | Candidate source | Selector | Teacher budget |
|---|---|---|---|
| Base | student ANN top-L | top score | B fixed |
| Quota | student ANN / per-class | fixed quota | B fixed |
| Nash | per-class ANN | Nash marginal utility | B fixed |
| Fast Nash | union ANN top-L | Nash marginal utility | B fixed |
| Nash-p | same pool | p=-1,0,0.5,1 | B fixed |
| Hybrid | ANN+BM25 | uncertainty-weighted Nash | B fixed |

Controls: same query set, student checkpoint, relevant-doc annotation, class distribution, teacher version, relevance threshold, index recall and wallclock. Evaluate nDCG@10, Recall@K, category entropy, per-class macro performance, RAG support coverage, query p50/p99 latency, **cost per relevant teacher judgment**.

## 10. Redis/Notebook metadata and chunking

Proposed key **paper:nashann:2602.08742**. Sections for chunks:
- question / motivating gap
- social welfare vs hard quotas
- utility/NSW equations
- p-means
- single-attribute exact algorithm and guarantees
- multi-attribute hardness/approximation
- FetchUnion and runtime tables
- tradeoffs/limitations
- teacher-budget acquisition adaptation

Per-claim fields: claim_type={paper,analysis}, paper_section, equation, venue, arxiv_id, dataset, metric, unit, baseline, table_anchor, source_url, extraction_date. Retain **fixed vs multi-attribute** flags on guarantee claims; this prevents an indexing agent from misapplying Theorem 1 to overlapping groups.

## 11. Primary references

- arXiv preprint: https://arxiv.org/abs/2602.08742
- ICLR 2026 published PDF: https://openreview.net/pdf/cfafb6e42fa52a198856f231e89c74b0846d8560.pdf
- Implementation: https://github.com/shivi98g/NashANN
- **Introduction and Section 2 / Definition 1**: purpose, agent utilities and Nash objective.
- **Algorithm 1 / Theorem 1 / Corollary 2**: exact and approximate single-attribute search.
- **Theorems 3 and 4**: multi-attribute NP-hardness and approximation on log welfare.
- **Appendix F.5; Tables 2, 8, 9**: p-mean comparison, FetchUnion speed vs diversity.
- Microsoft Research (research context): https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/

**Bottom line:** NashANN is an *economically motivated, adaptive result-set allocation mechanism*. Its value for post-training is a **testable proposal** to spend scarce reward-model evaluations across competing query/document categories without rigid quotas; it does not itself optimize retriever weights or solve query-atom collective evidence coverage.
