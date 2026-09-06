# Welfarist Formulations for Diverse Similarity Search

## Paper

- **Welfarist Formulations for Diverse Similarity Search** — Siddharth Barman, Nirjhar Das, Shivam Gupta, Kirankumar Shiragur (ICLR 2026)
- [arXiv](https://arxiv.org/abs/2602.08742)
- [ICLR paper page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/abaab8d9908c3df048fbfc0802dc778e-Abstract-Conference.html)
- [OpenReview PDF](https://openreview.net/pdf?id=UWhOUrsgkA)

## Brief overview

This paper studies nearest-neighbor retrieval when the returned set should be not only relevant to the query, but also **diverse across attributes**. Instead of treating diversity as a hard constraint and then maximizing relevance subject to that constraint, the authors formulate retrieval using welfare functions from mathematical economics. In particular, they focus on **Nash social welfare**, which provides a single objective that can adaptively trade off relevance and diversity depending on the query.

The formulation also exposes a parameter for controlling the relevance–diversity trade-off. The paper then develops efficient nearest-neighbor algorithms with provable approximation guarantees for these welfare-based objectives. Importantly, the method is designed to sit on top of a standard ANN system: an existing ANN method can be used as a subroutine while the welfare-based layer chooses a set of neighbors that approximately maximizes the desired relevance-and-diversity objective.
