# DISCo: Collective Query Coverage

## Paper

- **A Dense Subset Index for Collective Query Coverage** — Kartik Nair, Pritish Chakraborty, Atharva Tambat, Indradyumna Roy, Soumen Chakrabarti, Anirban Dasgupta, Abir De (ICLR 2026)
- [ICLR paper page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6ded182245f8e4db3963d2027174133d-Abstract-Conference.html)
- [OpenReview PDF](https://openreview.net/pdf?id=cUdODCFjUM)

## Brief overview

DISCo studies retrieval settings where one document is not enough to cover a complex query, such as multi-hop question answering or text-to-SQL. Instead of independently ranking documents by how well each one matches the complete query, it asks for a **small set of corpus items whose contextual token vectors collectively cover the query's contextual token vectors**. The resulting coverage objective is submodular, which lets the method build the retrieved set iteratively.

At each step, DISCo probes a dense index for an item with high remaining marginal coverage, selects it, and then **edits the remaining query representation** so that later probes focus on parts of the query that are still uncovered. Its index is built using random projections in a lifted dense vector space, allowing this successive retrieval process to remain sublinear in corpus size. The "decomposition" here is therefore not explicit natural-language subquestion generation; it is an iterative decomposition of the query representation through residual query edits and repeated retrieval.
