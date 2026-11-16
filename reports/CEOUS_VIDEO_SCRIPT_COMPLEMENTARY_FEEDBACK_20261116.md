# Ceous — spoken video script (approximately 3 minutes)
**16 November 2026 · Draft based on completed work, not a new-method success claim**

## 0:00–0:25 — Problem (show the two-stage pipeline)
Our project began from a Microsoft Research question: when strong relevance judgments are expensive, how can we use them to improve retrieval? A dense retriever searches a large collection quickly, but a stronger reranker is usually allowed to inspect only a small shortlist. If a relevant document is missing from that shortlist, reranking alone cannot recover it.

## 0:25–0:55 — What is TTT-Embed? (show the vector equation)
Recent work called TTT-Embed gives us a different route from retraining an entire encoder. The retriever and all document embeddings stay frozen. We take a small number of query–document pairs, ask a cross-encoder teacher to score them, and fit a small residual vector so that the retriever's similarity distribution better matches the teacher's preferences. We add that vector to future query embeddings and search the same full index again.

## 0:55–1:20 — Scopes and budget (show 0.01 / 10 / 100 table)
The paper has three sharing scopes, not three separate methods trained with different labels: global-wise pools across tasks, task-wise pools inside a task, and query-wise adapts one query. The budget b is the total number of expensive scored query–document pairs divided by the number of queries. In the published averages, global sharing is strongest at very low budgets, task-wise at medium budgets, and query-wise at high budgets. Those numbers belong to the paper, not our experiments.

## 1:20–2:05 — What we reproduced (show the two measurements separately)
We completed two related tests. On SCIDOCS we reproduced the per-query ReFIT procedure with frozen Contriever and a MiniLM cross-encoder. Recall@100 moved from 37.8067 to 37.8717 percent—a very small, statistically inconclusive gain. The published 40.1 percent was not reproduced. Separately, we tested task-wise TTT-Embed on FiQA, using Qwen3-Embedding-0.6B, Qwen3-Reranker-4B scores and five cross-fitting folds. The held-out nDCG@10 went from 0.46722 to 0.46313. All folds improved on their own training queries, but there was no reliable generalization to unscored queries. These are different experiments and metrics, so we do not combine their improvements.

## 2:05–2:40 — Our precise hypothesis (show dense vs complementary feedback)
That brings us to the new question. In TTT-Embed, the expensive feedback pairs are selected externally. Dense top-k candidates may contain similar or repeated information. Could a complementary retriever—BM25 for exact terms, ColBERT for token-level matching, or DISCo-style collective coverage—find more informative candidates? We propose to keep the teacher, number of scored pairs, optimizer, residual scope and evaluation fixed, and change only which candidate documents are bought for feedback. Mathematically, this changes the selected set S, not TTT-Embed's KL optimization rule.

## 2:40–3:05 — What it would take to claim success
This is a hypothesis, not a demonstrated improvement. We must measure full-corpus retrieval on unseen queries, track whether genuinely new judged-relevant documents appear, and count additional retrieval cost as well as reranker calls. We must also compare simple random, rank-stratified, MMR and deeper-retrieval controls. Our goal is to learn *when* complementary feedback makes a reusable residual more useful—not to claim that combining existing methods is automatically novel.

**Recording reminder:** Say “no reliable held-out improvement” and “proposed complementary feedback selector.” Do not say we reproduced the paper's large global gains or built an officially faithful DISCo index.
