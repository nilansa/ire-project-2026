# Retrieval Work: Manik Varma and Kirankumar Shiragur

## Manik Varma

**Overall focus:** extreme classification as large-scale retrieval and recommendation: map a query to a small relevant set among millions or billions of labels, items, or documents.

Representative directions:

- Scalable label partitioning and classification: [Parabel](https://www.microsoft.com/en-us/research/publication/parabel-partitioned-label-trees-for-extreme-classification-with-application-to-dynamic-search-advertising/) and [Slice](https://www.microsoft.com/en-us/research/publication/slice-scalable-linear-extreme-classifiers-trained-on-100-million-labels-for-related-searches/).
- Deep extreme retrieval and negative mining: [DeepXML](https://arxiv.org/abs/2111.06685) and [NGAME](https://arxiv.org/abs/2207.04452).
- ANN-based hard-negative training and stale-index mitigation: [ASTRA](https://arxiv.org/abs/2409.20156).
- Efficient non-autoregressive generative retrieval: [PIXAR](https://arxiv.org/abs/2406.06739).

Background: [Microsoft profile](https://www.microsoft.com/en-us/research/people/manik/), [retrieval keynote](https://www.microsoft.com/en-us/research/video/keynote-extreme-classification-for-dense-retrieval-and-personalized-recommendation/), and [XC repository](https://manikvarma.org/downloads/XC/XMLRepository.html).

## Kirankumar Shiragur

**Overall focus:** efficient vector search, quantization, single-vector and multi-vector embedding models, diversity-aware search, and agentic retrieval.

Representative directions:

- Graph construction and pruning guarantees for the DiskANN family: [Sort Before You Prune](https://openreview.net/forum?id=JnXbUKtLzz).
- ANN with diversity constraints: [Graph-Based Algorithms for Diverse Similarity Search](https://proceedings.mlr.press/v267/anand25a.html) and [Welfarist Formulations for Diverse Similarity Search](https://openreview.net/forum?id=UWhOUrsgkA).
- Multi-vector nearest-neighbour indexing: [α-Reachable Graphs](https://openreview.net/forum?id=v8jSxLHEE9).
- Learned token weighting for late-interaction retrieval: [Incorporating Token Importance in Multi-Vector Retrieval](https://arxiv.org/abs/2511.16106).

Background: [personal research page](https://sites.google.com/view/kiran-shiragur).

## Relevance to this project

Varma's work connects candidate generation and negative mining to learning at extreme scale. Shiragur's work studies the vector-search structures and retrieval objectives that generate those candidates. Together, they motivate evaluating ANN by downstream learning utility, not only recall and latency.
