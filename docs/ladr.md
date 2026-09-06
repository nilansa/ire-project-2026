# Lexically-Accelerated Dense Retrieval (LADR)

## Paper and code

- Paper: [Lexically-Accelerated Dense Retrieval](https://arxiv.org/abs/2307.16779) (SIGIR 2023; [DOI](https://doi.org/10.1145/3539618.3591715)).
- Original reproduction repository: [Georgetown-IR-Lab/ladr](https://github.com/Georgetown-IR-Lab/ladr).
- Executable PyTerrier implementation: [`pyterrier_dr/flex/ladr.py`](https://github.com/terrierteam/pyterrier_dr/blob/f790d83b420ab70abafe8c608c8d3a086c430f3b/pyterrier_dr/flex/ladr.py).
- Related project context: [Microsoft retrieval post-training challenge](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/).

## Overview

LADR is an approximate dense-retrieval strategy. It uses a fast lexical retriever to produce seed documents, then explores a document-proximity graph and scores the resulting candidates with the dense query/document model. The goal is to approach exhaustive dense retrieval quality at lower query latency, not to train a new encoder.

## Exact inference pipeline

The implementation in `pyterrier_dr` performs the following operations:

1. Encode the query as a dense vector and obtain lexical seed documents, normally BM25/PISA results.
2. Convert external `docno` values to the FlexIndex's internal integer document IDs.
3. **Proactive LADR:** collect the seed IDs and the neighbors of every seed for the configured number of hops. Deduplicate the candidate IDs and optionally cap the scoring budget.
4. **Adaptive LADR:** score the current IDs, select the highest-scoring `depth` IDs, add their unseen graph neighbors, and repeat until the budget or hop limit is reached.
5. Score all collected candidate document vectors with the dense scorer, keep the top `num_results`, sort by descending dense score, and map internal IDs back to `docno` values.

The default graph builder computes document-document cosine neighbors in batches and stores directed top-`k` edges. The paper's principal setting used `k=128`; it also studied approximate and BM25-derived proximity graphs. LADR therefore explores a restricted candidate set but does not change the dense scoring function.

## Metrics and datasets

The paper evaluates MS MARCO Passage Dev (small), 6,980 queries, with RR/MRR@10 and Recall@1000, and uses TREC Deep Learning 2019/2020 for graded nDCG@10 and Recall@1000. Its TAS-B exhaustive reference is MRR@10 0.347 and Recall@1000 0.978; at the reported approximately 8 ms/query point, Proactive LADR is 0.345/0.932 and Adaptive LADR is 0.347/0.960. These numbers are dense TAS-B experiments on the paper's hardware and are not Big-ANN sparse-track results.

## What LADR is—and is not

LADR is not a cross-encoder reranker and does not apply a second text model to a fixed top-`k` list. It is a candidate-generation/search procedure: lexical retrieval seeds the search, a graph expands candidates, and one dense model scores those candidates. The paper reports a separate “Re-Ranking” baseline (BM25 followed by dense scoring); that baseline should not be conflated with LADR.

This distinction explains why Recall@1000 can be high while MRR@10 or nDCG@10 is lower: a relevant passage may be present somewhere in the 1,000 candidates but receive a low dense score and therefore appear outside the top ten. Candidate coverage and ordering are different properties.

## Relation to Microsoft's post-training data selection

Microsoft's post-training challenge asks which query-document examples should receive expensive reward-model or LLM cross-encoder scoring during model improvement. LADR addresses a different stage: serving-time retrieval efficiency for an already-trained dense model. LADR does not select training data, define a reward, update model weights, or perform post-training. It could be used as an engineering component to generate or prioritize candidate examples before reward scoring, but that would be a new experiment rather than a claim made by the LADR paper.
