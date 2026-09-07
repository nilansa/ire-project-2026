# Research Plan (Draft)

**Team:** Ceous

## Brief of the idea

This project studies **data selection for post-training dense retrieval models**: under a fixed budget for expensive relevance/reward feedback, which query-document pairs should be selected for scoring so that the resulting supervision improves the final retriever most effectively?

## 1. What is the idea?

The project was motivated by the Microsoft Research Fellowship challenge [**Efficient and accurate post-training of retrieval models**](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/). The initial exploration had started from the **Big ANN Benchmarks/competition**, which studies the effectiveness-efficiency trade-off of approximate nearest-neighbor search at large scale. The Microsoft challenge suggested a more direct training-time question: how should retrieval systems decide **which query-document pairs are worth obtaining expensive feedback for?**

Retrieval typically has sparse supervision: relevance signals are unavailable for most possible query-document pairs. The Microsoft challenge proposes using high-quality feedback from reward/relevance models, such as LLM-based cross-encoders or human feedback, during post-training. Such feedback is substantially more expensive than first-stage retrieval, so it cannot generally be obtained for every possible query-document pair.

Two immediate questions in the challenge are:

1. **Data selection:** which queries and corresponding documents should be selected for scoring by reward models during post-training so as to maximize downstream retrieval and/or RAG performance?
2. **Loss formulation:** given the retrieval architecture and the available pointwise, pairwise, or listwise feedback, what post-training loss should be optimized?

The **initial scope of this project is the data-selection question**. Under a fixed feedback budget, the selection mechanism determines the empirical distribution of scored/labeled query-document pairs. Since the post-training loss is computed from these pairs, this determines which examples contribute supervision and gradients to the retriever. Loss formulation is therefore an important coupled question, but it will initially be **held fixed** so that the effect of data/candidate selection can be isolated.

A first concrete question is whether the **search procedure used to construct the candidate pool** is itself consequential. Before an expensive reward model can score documents, some retrieval/search mechanism must decide which documents are exposed as candidates. We therefore ask:

> **Holding the retriever, reward model, feedback budget, post-training loss, and evaluation fixed, does changing the ANN/search procedure used to construct candidates change the final post-trained retriever?**

### a. Why is it important? If it were solved what would improve?

The central practical constraint is that high-quality feedback is expensive. If only a fixed number of query-document pairs can be scored, improving **where this budget is spent** can improve the quality of supervision without requiring a larger teacher or more reward-model calls.

Candidate generation is upstream of this decision. A document that is never surfaced cannot be scored and therefore cannot contribute to the post-training loss. Consequently, the search layer may affect not only retrieval latency but also the **training distribution** seen during post-training.

This concern is especially relevant because ANN systems expose a strong effectiveness-efficiency trade-off. The [Big ANN competition results](https://papers.neurips.cc/paper_files/paper/2025/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets_and_Benchmarks_Track.pdf) compare systems using recall and queries-per-second (QPS), and show that substantially different throughput can be obtained at comparable target recall. If ANN search is used inside candidate generation for post-training, the important unanswered question is whether moving along this recall/throughput frontier is **training-neutral**, or whether the changed candidate set also changes the final learned retriever.

If this effect can be characterized, candidate search can be chosen jointly with the feedback budget rather than being treated as an independent systems component. More broadly, it would make it possible to distinguish gains due to **better supervision selection** from gains due to the teacher model or the loss itself.

### b. What is the relevant literature? Does the literature acknowledge this gap? Were there attempts at it.

**[For review: ANCE — Xiong et al., ICLR 2021]**  
[Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://openreview.net/forum?id=zeFrfgyZln)

ANCE identifies negative sampling as a training bottleneck for dense retrieval and uses an ANN index over the corpus to retrieve global hard negatives. Its results establish that the distribution of documents exposed to the loss can materially change retriever optimization and final retrieval quality. This is strong motivation for our premise that candidate selection matters. However, ANCE does **not** isolate the ANN algorithm or ANN operating point itself as the intervention while holding the rest of post-training fixed.

**[For review: RocketQA — Qu et al., NAACL 2021]**  
[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/)

RocketQA retrieves candidate hard negatives and then uses a stronger cross-encoder to denoise them before training the dual encoder. This gives a close precedent for the pipeline considered here: **candidate retrieval → expensive cross-encoder feedback → retriever training**. It again shows that the candidate pool and teacher signal are coupled, but does not ask whether different ANN/search procedures create different downstream post-training outcomes.

**[For review: LADR — Kulkarni et al., SIGIR 2023]**  
[Lexically-Accelerated Dense Retrieval](https://arxiv.org/abs/2307.16779)

LADR is an inference-time retrieval method rather than a post-training method, but it is relevant to the **allocation of retrieval compute**. It begins with lexical seed documents and then explores a document-proximity graph using dense scores. Adaptive LADR selectively expands the neighborhoods of the most promising documents, rather than spending dense-scoring compute uniformly. This demonstrates that deciding **where to search next** can itself be a structured budget-allocation problem. Our project asks whether an analogous choice, when made upstream of expensive reward scoring during post-training, changes the supervision and ultimately the trained retriever.

**[For review: Big ANN Benchmarks]**  
[Results of the Big ANN: NeurIPS'23 competition](https://papers.neurips.cc/paper_files/paper/2025/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets_and_Benchmarks_Track.pdf)

Big ANN makes explicit that ANN methods and operating points should be compared on a recall-throughput frontier rather than by recall alone. It establishes the systems-level motivation for varying search procedures. It does **not** answer the training-time question of whether two search procedures with similar retrieval recall but different candidate identities lead to the same post-trained model.

**[For review: Preliminary sanity baseline]**

A small controlled MS MARCO pilot was run with the same dense-retriever embeddings while changing only candidate search between **exact search, HNSW, and IVF-Flat**. HNSW and IVF-Flat were calibrated to similar high Recall@200 (approximately **95.6%** and **96.2%**), yet their candidate sets were not identical: exact/HNSW and exact/IVF candidate Jaccard were approximately **0.922** and **0.939**. After selecting 20 deterministic candidate positions per query, only about **44.9%** and **48.1%** of the selected examples were shared with exact search. This is only a sanity check, not evidence of a downstream training effect, but it shows that **similar ANN recall can still expose substantially different training examples**, making the proposed controlled post-training experiment worth pursuing.

Taken together, the literature supports three pieces of the motivation: **(i)** which negatives/candidates are used for training matters, **(ii)** expensive cross-encoder feedback is naturally applied only after candidate retrieval, and **(iii)** ANN/search procedures differ in how they allocate retrieval compute and which candidates they recover. The gap we want to test directly is narrower: **when all other post-training choices are fixed, does the search mechanism that generates the candidate pool itself change the final retriever?**

## 2. What is the plan?

### a. What are the various threads you want to pursue to solve this (more literature survey can be one of them but you should tell which specific papers, problems, etc)? Does it justify the specified team size? Do you have the resources (skills and hardware) to execute it?

The broader goal is to view retrieval post-training as a sequence of **modules**—candidate generation, budgeted selection of query-document pairs for feedback, reward/relevance scoring, loss construction, and model update—and ask which of these decisions should remain fixed heuristics and which can be made **learnable**.

The first thread will focus on **learnable data/candidate selection**. Instead of using a fixed rule to choose which candidates receive expensive reward-model scores, a selection policy can potentially learn which query-document pairs are most useful to label under a fixed budget. The initial ANN study provides the controlled foundation for this direction: before learning a selector, we first need to establish whether changing only the candidate-generation mechanism changes the supervision distribution and the final retriever.

[For review: a complementary direction is to study training-free/ad-hoc selection rules under the same budget. This will be specified after the initial learnable-selection formulation is fixed.]
