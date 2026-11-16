# Ceous — ten-minute presentation narration

**Target:** approximately 10:00 · **Slides:** 13 · **Status:** evidence-led research presentation, not a validated new-method result.  
**Presented by:** Nilanjan Sarkar, Team Ceous, IIIT Hyderabad, IRE M26CS4.406 · 16 November 2026.

**Recording deck:** https://nilansa.github.io/explainer/explainers/ceous-tttembed-10-minute-presentation-2026-11-16.html  
**Teaching explainer:** https://nilansa.github.io/explainer/explainers/ceous-tttembed-start-here-2026-11-16.html

Record in a calm teaching pace. Timings allocate time for reading equations and pointing at the screen, so actual spoken time may vary. Use **N** in the browser to show/hide your slide script; use **→** for next slide.

## 00:00–00:35 — Slide 1: The retrieval problem

*On screen: Why a strong reranker cannot rescue what it never sees*

Our project began from a Microsoft Research challenge: how can a retrieval system benefit from expensive high-quality relevance feedback? In a standard two-stage search system, a fast retriever returns a shortlist from a very large corpus. A more accurate but expensive reranker scores only that shortlist. Here is the limitation: if a relevant document was never retrieved, the reranker cannot bring it back. Our investigation therefore focuses on improving what the first stage can retrieve, not merely changing the order of an unchanged list.

## 00:35–01:17 — Slide 2: Who does which job?

*On screen: Dense retriever, lexical retriever, ColBERT and cross-encoder*

Let me separate four names that sound similar. BM25 retrieves documents through matching words and term statistics. A dense bi-encoder independently maps a query and a document to vectors; their dot product supports very fast search through a precomputed index. ColBERT is another retriever. Instead of one vector per document, it keeps contextual token vectors and computes a late-interaction MaxSim score. A cross-encoder is different: it reads the query and one candidate document together and predicts a relevance score. This more powerful model can be too expensive to run on every document. We use its output as teacher feedback.

## 01:17–02:03 — Slide 3: One score is one query–document pair

*On screen: The budget is measured in purchased teacher judgments*

What does a query-document pair mean? Suppose the query asks for a particular scientific explanation. Pair one consists of that query with abstract A, and pair two consists of the same query with abstract B. The reranker evaluates the two pairs separately, so that costs two relevance judgments. TTT-Embed measures supervision through a total pair-scoring budget B, or an average budget b equal to B divided by the number of workload queries. If one thousand queries share ten thousand purchased scores, b equals ten. A shared method may spend these scores on a subset of queries and reuse what it learns for the rest. Evaluation labels remain separate.

## 02:03–02:47 — Slide 4: Scores become a probability distribution

*On screen: Softmax allows us to compare relative preferences*

A black-box reranker can return scalar relevance scores, and a retriever can return similarity scores. Their raw numeric scales are different. Rather than demand identical raw values, both methods apply softmax on the same candidate list. Softmax exponentiates and normalizes scores so they become positive numbers summing to one. The temperature controls how sharp the preferences become. Here is an invented example: the teacher puts eighty percent of its mass on document one, while the retriever assigns only thirty percent. The goal is to make the retriever prefer the document that the teacher prefers. These are relative preferences over three candidates, not real ground-truth probabilities.

## 02:47–03:43 — Slide 5: Why KL divergence?

*On screen: A differentiable way to align teacher and student ranking preferences*

KL divergence, short for Kullback-Leibler divergence, measures how much one probability distribution differs from another. In our example, the teacher probabilities are point eight, point fifteen, and point zero five. The student probabilities are point three, point four, and point three. We multiply each teacher probability by the logarithm of teacher over student, then add the three terms. That gives approximately zero point five four eight nats. If the distributions were identical, KL would be zero. Why optimize this? Because the teacher provides relative relevance preferences, and KL is a smooth, differentiable objective that asks the student to imitate those preferences. This is knowledge distillation at inference time. It is not automatically maximizing the real retrieval metric: the teacher can be wrong, and the three scored documents might omit the actual relevant evidence.

## 03:43–04:40 — Slide 6: Where do the gradients flow?

*On screen: The reranker is a black box; the vector arithmetic is not*

This is the most important technical slide. A black-box reranker returns numbers for query-document pairs. We store those numbers as fixed teacher targets. We do not ask the reranker for its hidden states or gradients. The differentiable part is our own retriever scoring formula: a query vector dotted with each fixed document vector, followed by softmax and KL. If a student probability is larger than the teacher target, the derivative with respect to that logit is positive; if it is smaller, the derivative is negative. Backpropagation through this simple arithmetic tells us how to move the query vector. Gradient descent or Adam updates that vector, or TTT-Embed's residual, not the neural encoder. One hundred optimizer steps can reuse the same scored slate. It does not mean one hundred fresh calls to the expensive reranker.

## 04:40–05:30 — Slide 7: Three related approaches, different lifetimes

*On screen: TOuR, ReFIT and TTT-Embed are not interchangeable*

The methods have a common pattern, but we must separate their optimization lifetimes. TOuR uses cross-encoder pseudo-labels to optimize a test query and can repeat retrieval and feedback rounds. Its soft variant uses a KL-style distribution loss, while its hard variant uses a different pseudo-positive objective. ReFIT also optimizes the current query using KL, but typically takes one feedback round and many cheap vector-update steps before searching again. TTT-Embed moves the optimization into an external residual vector that can be kept and reused. So we are not updating model weights in any of these examples, but there is still actual test-time optimization. This is why the exact phrase is frozen-model query or residual adaptation, not zero-training magic.

## 05:30–06:19 — Slide 8: TTT-Embed: global, task and query

*On screen: How sharing changes the best use of a small reward budget*

Now the main TTT-Embed insight: we choose how widely to reuse the learned vector. Global means one correction shared across tasks; task-wise means one per task; and query-wise means one for one scored query. The paper defines a normalized budget b as rewarded document pairs divided by workload queries. At extremely small budgets, shared global state can spread scarce supervision further. At a medium budget, task-wise specialization works better in the reported average. At high budgets, query-wise adaptation is strongest. The paper's representative gains are plus one point two three, plus six point three six, and plus eight point three six nDCG points respectively, but at very different budgets. These are the paper's results, not ours.

## 06:19–06:57 — Slide 9: What data and models?

*On screen: The paper's benchmark is larger than our reproduction*

What do they actually evaluate? TTT-Embed has fifteen MTEB text retrieval tasks, spanning financial questions, scientific-paper retrieval, argument retrieval, fact verification and multi-hop questions. Across those task-specific corpora, the paper reports more than twelve thousand queries and about one point six six million documents. It uses five embedding models: two Qwen3 embedding sizes, EmbeddingGemma, and two Gemini embedding models. Its main reward teacher is Qwen3-Reranker-4B. Importantly, these are single-vector retrievers. ColBERT is an alternative token-level retriever that could provide complementary candidates, not one of those five main TTT-Embed backbones.

## 06:57–07:45 — Slide 10: What we reproduced: ReFIT on SCIDOCS

*On screen: Full query-wise run, but not query-wise TTT-Embed*

Our first completed evaluation is a ReFIT reproduction attempt, which applies query-wise feedback, but is not query-wise TTT-Embed. On the original SCIDOCS test set, we had one thousand queries and more than twenty-five thousand document abstracts. Contriever-MSMARCO was the frozen retriever and MiniLM-L6 was the frozen cross-encoder teacher. We scored initial candidates, took one hundred query-vector KL steps, and searched the full corpus again. Raw Recall at one hundred was thirty-seven point eight zero six seven percent. Our ReFIT result was thirty-seven point eight seven one seven, a gain of only zero point zero six five percentage points. Its confidence interval includes zero, and we did not reproduce the paper's forty point one percent.

## 07:45–08:35 — Slide 11: What we reproduced: task-wise TTT-Embed on FiQA

*On screen: The learned vector fits training feedback but did not transfer reliably*

Our second completed experiment is closer to TTT-Embed itself. We tested task-wise sharing on FiQA2018 with six hundred forty-eight queries and fifty-seven thousand six hundred thirty-eight documents. We froze Qwen3-Embedding-0.6B and used cached Qwen3-Reranker-4B probabilities for the ten highest-ranked pairs per rewarded query. Five cross-fitting folds held out each query exactly once. The original nDCG at ten was zero point four six seven two two; after task-wise adaptation it was zero point four six three one three. The average difference was slightly negative, and the confidence interval included zero. Interestingly, every vector improved the queries it was fitted on, but the gain did not transfer reliably. That motivates investigation; it is not proof that TTT-Embed fails in general.

## 08:35–09:32 — Slide 12: Our hypothesis: buy better feedback pairs

*On screen: Complementary retrieval can change what the shared vector learns*

This leads to our actual research proposal. In the baseline, TTT-Embed learns from whatever documents the ordinary dense retriever selects for teacher scoring. But what if the top-ranked candidates are redundant? A complementary system, such as BM25 for rare words, ColBERT for token-level matches, or a DISCo-inspired coverage mechanism, may reveal evidence missing from those candidates. We would take the union of the initial dense pool and the complementary pool, select exactly k distinct documents without seeing the teacher's unpurchased scores, then buy exactly k relevance judgments. We keep the teacher, learning rule, number of scored pairs, residual sharing scope and evaluation fixed. If unseen-query retrieval improves at matched total search and teacher cost, then candidate selection—not a new KL loss—may be the source of improvement. That is our hypothesis, not a completed result.

## 09:32–10:00 — Slide 13: What the report can honestly claim

*On screen: A precise falsifiable research direction, not a success story*

To conclude, earlier research already established score-guided test-time query optimization and reusable residuals. Our local reproduction attempts showed how difficult the gains are to recover reliably. The next falsifiable question is whether complementary candidate selection improves the value of a limited number of purchased teacher scores. Until that controlled experiment is completed, we describe it as a research hypothesis, not as a demonstrated new method.

---

## Scientific integrity / oral defense

- **Do not say:** we train the reranker or backpropagate through a black-box teacher. We only receive scalar scores and differentiate the retriever-side similarity/softmax objective with respect to an accessible query vector or external residual.
- **Do not say:** 100 query-vector optimization steps mean 100 paid reranker calls. ReFIT can take multiple cheap steps over one fixed feedback slate; TOuR may involve further feedback rounds.
- **Do not say:** minimizing KL directly maximizes nDCG or Recall. It is a differentiable distillation surrogate for teacher-preference matching within a selected slate.
- **Do not conflate:** query-wise ReFIT SCIDOCS Recall@100 with task-wise TTT-Embed FiQA nDCG@10, or paper-wide TTT-Embed numbers with ours.
- **Do not call:** a candidate-selection proposal a validated algorithm. No controlled Ceous result yet supports the complementary-selector hypothesis.
- **Explain access assumptions:** a black-box teacher score alone is sufficient only when we can also read retriever-side embeddings and run vector searches with adjusted queries. A text-only search API cannot necessarily support this mechanism.
- **Sources:** [ReFIT](https://arxiv.org/html/2305.11744), [TOuR](https://aclanthology.org/2023.findings-acl.354/), [TTT-Embed](https://arxiv.org/abs/2608.12569), [Ceous result ledger](https://github.com/nilansa/ire-project-2026/blob/main/experiments/inference-time-ceous-20261115/RESULTS.md), and the FiQA private five-fold cross-fit record.
