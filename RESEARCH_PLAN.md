# Research Plan (Draft)

**Team:** Ceous

## Brief of the idea

- The project is based on the Microsoft Research proposal [**Efficient and accurate post-training of retrieval models**](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/).
- A retriever first produces candidate documents for a query.
- High-quality relevance/reward feedback is then obtained for only some query-document pairs using an expensive scorer, for example:
  - an LLM-based cross-encoder, or
  - human feedback.
- Retrieval is often constrained by **sparse training data**: relevance signals are unavailable for most query-document pairs.
- Under a fixed feedback budget, the main question is: **which query-document pairs should be selected for reward scoring?**
- We will study this first from the **ANN/search side**: does the search method used to generate candidates affect which examples receive feedback and, finally, the post-trained retriever?

## 1. What is the idea?

- From the [Microsoft Research proposal](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/), two directions are especially relevant to this project:
  - **Data Selection:** which queries and corresponding documents should be selected for scoring by reward models during post-training to maximize downstream retrieval and/or RAG performance?
  - **Loss Formulation:** what post-training loss should be used for a given retrieval architecture and feedback design?
- Our **initial research will focus on Data Selection**.
- The basic pipeline is:
  - query → retriever/search → candidate documents → expensive reward/relevance scoring → post-training loss → updated retriever.
- Under a fixed scoring budget:
  - the search stage decides which documents are available as candidates;
  - only selected query-document pairs receive expensive feedback;
  - only those scored/labeled pairs can contribute to the post-training loss.
- Our first question is:

> **Holding the retriever, reward model, feedback budget, post-training loss, and evaluation fixed, does changing the ANN/search procedure used to generate candidates change the final post-trained retriever?**

- This direction grew out of an initial study of the **Big ANN Benchmarks**, where ANN methods are compared using retrieval effectiveness (for example, recall) and efficiency (for example, queries per second).

### a. Why is it important? If it were solved what would improve?

- High-quality reward/relevance feedback is expensive.
- If only a fixed number of query-document pairs can be scored, the feedback budget should be spent on useful pairs.
- Candidate generation matters because:
  - a document that is not retrieved cannot be scored;
  - a document that is not scored cannot contribute feedback to the post-training loss.
- ANN methods have an **effectiveness-efficiency trade-off**:
  - higher recall can require more search work;
  - faster search can return a different candidate set.
- The training-time question is therefore:
  - if two ANN/search methods have similar retrieval effectiveness but return different candidates, do they lead to the same final retriever after post-training?
- If this is understood, candidate search can be chosen not only for speed, but also for the usefulness of the candidates it exposes for post-training.

### b. What is the relevant literature? Does the literature acknowledge this gap? Were there attempts at it.

**[For review: ANCE — Xiong et al., ICLR 2021]**  
[Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://openreview.net/forum?id=zeFrfgyZln)

- ANCE uses ANN retrieval to find global hard negatives for dense-retriever training.
- It shows that **which negatives are used during training matters** for the final retriever.
- This supports the idea that the candidate set exposed to training can matter.
- ANCE does not directly compare different ANN methods while keeping the rest of training fixed.

**[For review: RocketQA — Qu et al., NAACL 2021]**  
[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/)

- RocketQA first retrieves candidate hard negatives.
- A stronger cross-encoder then filters/denoises them.
- The selected examples are used to train the retriever.
- This is close to the structure we study:
  - candidate retrieval → expensive cross-encoder feedback → retriever training.

**[For review: LADR — Kulkarni et al., SIGIR 2023]**  
[Lexically-Accelerated Dense Retrieval](https://arxiv.org/abs/2307.16779)

- LADR is an inference-time retrieval method, not a post-training method.
- It starts from lexical seed documents and then spends dense-retrieval compute around promising document neighborhoods.
- It is relevant because it shows that **where retrieval compute is spent** can be chosen in a structured way.
- In our setting, the related question is where to spend the expensive reward-scoring budget after candidate retrieval.

**[For review: Big ANN Benchmarks]**  
[Results of the Big ANN: NeurIPS'23 competition](https://papers.neurips.cc/paper_files/paper/2025/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets_and_Benchmarks_Track.pdf)

- Big ANN compares ANN methods using retrieval effectiveness and efficiency, including recall and QPS.
- It shows that different ANN methods/settings can operate at different points on this effectiveness-efficiency trade-off.
- It does not study whether these ANN choices later change a trained retriever.

**[For review: Preliminary sanity test]**

- **Dataset:** a 50,000-document MS MARCO passage subset.
- **Queries:** 1,000 train, 200 calibration, and 500 evaluation queries.
- **Encoder:** a fixed **ANCE FirstP** checkpoint.
- **Candidate depth:** top-200 documents per query.
- **Search methods:** exact search, HNSW, and IVF-Flat.
- HNSW and IVF-Flat were calibrated to almost the same Recall@200:
  - HNSW: **0.956**
  - IVF-Flat: **0.962**
- Even at this similar recall, they did not return the same top-200 candidates.
- When 20 candidate positions were selected per query, the selected pairs shared with exact search were only:
  - HNSW: **44.9%**
  - IVF-Flat: **48.1%**
- This is only a sanity test; it does **not** yet show that final retriever quality changes.
- But it gives a useful reason to run the controlled post-training experiment:
  - ANCE shows that the negatives used for training matter;
  - our sanity test shows that different ANN methods can produce different negatives even at similar retrieval recall.

- The specific gap we want to test is:
  - **when all other post-training choices are fixed, does the ANN/search method used for candidate generation change the final retriever?**

## 2. What is the plan?

### a. What are the various threads you want to pursue to solve this (more literature survey can be one of them but you should tell which specific papers, problems, etc)? Does it justify the specified team size? Do you have the resources (skills and hardware) to execute it?

- We will treat post-training retrieval as a small set of parts:
  - candidate generation;
  - selection of query-document pairs for reward scoring;
  - reward/relevance scoring;
  - post-training loss;
  - retriever update.
- These parts can be studied independently first, and later combined.

- **Direction 1 — ANN/search and data selection**
  - Keep the retriever, reward model, feedback budget, loss, and evaluation fixed.
  - Change only the ANN/search procedure used to generate candidates.
  - Measure whether the changed candidate set changes the final post-trained retriever.

- **Direction 2 — Diversity-aware selection (training-free)**
  - [Barman et al., *Welfarist Formulations for Diverse Similarity Search*, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/abaab8d9908c3df048fbfc0802dc778e-Abstract-Conference.html) studies nearest-neighbor retrieval that balances relevance and diversity.
  - Their method can use a standard ANN method as a subroutine and then choose a more diverse set of neighbors without training a new retriever.
  - In our setting:
    - ANN produces candidate documents;
    - a training-free diversity step chooses a less redundant subset;
    - the same fixed reward-model budget is spent on these query-document pairs.
  - The question is whether scoring **more diverse documents** gives better post-training supervision than scoring a more redundant top-k set.

- **[For review: later learnable direction]**
  - Instead of using fixed rules for selecting query-document pairs, ask whether the selection policy itself can be learned from which feedback is most useful for improving the retriever.
