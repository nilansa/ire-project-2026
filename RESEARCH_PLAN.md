# Research Plan (Draft)

**Team:** Ceous

## Brief of the idea

- The project is based on the Microsoft Research proposal [**Efficient and accurate post-training of retrieval models**](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/).
- A retriever first produces candidate documents for a query.
- High-quality relevance/reward feedback is then obtained for some query-document pairs using an expensive scorer, for example:
  - an LLM-based cross-encoder, or
  - human feedback.
- The main problem is that relevance signals are missing for most possible query-document pairs, while obtaining high-quality feedback for all pairs is too expensive.
- So, under a fixed feedback budget, the key question is: **which query-document pairs should be selected for reward scoring?**
- We will study this question from the **ANN/search side**: does the search method used to generate candidates affect which examples receive feedback and, finally, the post-trained retriever?

## 1. What is the idea?

- The Microsoft Research proposal studies post-training retrieval models using high-quality feedback from reward models.
- The proposal notes that retrieval is often constrained by **sparse training data**, meaning that relevance signals are unavailable for most query-document pairs.
- It asks, among other questions:
  - **Data Selection:** which queries and corresponding documents should be selected for scoring by reward models during post-training to maximize downstream retrieval and/or RAG performance?
  - **Loss Formulation:** what post-training loss should be used for a given retrieval architecture and feedback design?
- This project will first focus on the **Data Selection** question.
- The basic pipeline is:
  - query → retriever/search → candidate documents → expensive reward/relevance scoring → post-training loss → updated retriever.
- Under a fixed scoring budget:
  - the search stage decides which documents are available as candidates;
  - only the selected query-document pairs receive expensive feedback;
  - the post-training loss is computed using those scored/labeled pairs.
- Therefore, changing the candidate-generation search may change what the model learns even when the reward model and loss are fixed.
- The first concrete question is:

> **Holding the retriever, reward model, feedback budget, post-training loss, and evaluation fixed, does changing the ANN/search procedure used to generate candidates change the final post-trained retriever?**

- This direction grew out of an initial study of the **Big ANN Benchmarks**, where different ANN methods are compared on both retrieval effectiveness (for example, recall) and efficiency (for example, queries per second).

### a. Why is it important? If it were solved what would improve?

- High-quality reward/relevance feedback is expensive.
- If only a fixed number of query-document pairs can be scored, we should spend that budget on the most useful pairs.
- Candidate generation matters because:
  - a document that is not retrieved cannot be scored;
  - a document that is not scored cannot contribute feedback to the post-training loss.
- ANN methods have an **effectiveness-efficiency trade-off**:
  - higher recall can require more search work;
  - faster search can return a somewhat different candidate set.
- Big ANN shows this trade-off directly through recall and QPS comparisons.
- The training-time question is therefore:
  - if two ANN/search settings have similar retrieval effectiveness but return different candidates, do they lead to the same final retriever after post-training?
- If this is understood, we can choose candidate search not only for speed, but also for how useful its candidates are for post-training.

### b. What is the relevant literature? Does the literature acknowledge this gap? Were there attempts at it.

**[For review: ANCE — Xiong et al., ICLR 2021]**  
[Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://openreview.net/forum?id=zeFrfgyZln)

- ANCE uses ANN retrieval to find global hard negatives for dense-retriever training.
- It shows that **which negatives are used during training matters** for the final retriever.
- This supports our motivation that the candidate set can affect learning.
- ANCE does not directly compare different ANN methods or ANN search settings while keeping the rest of training fixed.

**[For review: RocketQA — Qu et al., NAACL 2021]**  
[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/)

- RocketQA first retrieves candidate hard negatives.
- A stronger cross-encoder is then used to filter/denoise them.
- The selected examples are then used to train the retriever.
- This is close to the structure we study:
  - candidate retrieval → expensive cross-encoder feedback → retriever training.

**[For review: LADR — Kulkarni et al., SIGIR 2023]**  
[Lexically-Accelerated Dense Retrieval](https://arxiv.org/abs/2307.16779)

- LADR is an inference-time retrieval method, not a post-training method.
- It first gets seed documents using lexical retrieval and then spends dense-retrieval compute around promising document neighborhoods.
- It is relevant because it shows that **where retrieval compute is spent** can be chosen in a structured way instead of searching every region equally.
- In our setting, the related question is where to spend the expensive reward-scoring budget after candidate retrieval.

**[For review: Big ANN Benchmarks]**  
[Results of the Big ANN: NeurIPS'23 competition](https://papers.neurips.cc/paper_files/paper/2025/file/63092d79154adebd7305dfd498cbff70-Paper-Datasets_and_Benchmarks_Track.pdf)

- Big ANN compares ANN methods using retrieval effectiveness and efficiency, including recall and QPS.
- It shows that different ANN methods/settings can operate at different points on this effectiveness-efficiency trade-off.
- It does not study whether these ANN choices later change a trained retriever.

**[For review: Preliminary sanity baseline]**

- We compared **exact search, HNSW, and IVF-Flat** using the same dense-retriever embeddings on a small MS MARCO setup.
- HNSW and IVF-Flat had similar Recall@200:
  - HNSW: **0.956**
  - IVF-Flat: **0.962**
- But the candidate sets were not identical:
  - exact/HNSW Jaccard: **0.922**
  - exact/IVF Jaccard: **0.939**
- After selecting 20 candidate positions per query, the fraction shared with exact search was only:
  - exact/HNSW: **0.449**
  - exact/IVF: **0.481**
- This does **not** yet show a training effect.
- It only shows that similar ANN recall can still expose noticeably different query-document pairs to the next scoring stage.

- The literature therefore supports that:
  - candidate/negative selection matters for retriever training;
  - expensive cross-encoder feedback can be applied after candidate retrieval;
  - ANN methods can return different candidates under different effectiveness-efficiency settings.
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
- The first thread will study **candidate generation and data selection**.
- The first controlled experiment will keep the retriever, reward model, feedback budget, loss, and evaluation fixed, and change only the ANN/search procedure.
- A later direction is to ask whether some of these selection decisions can be made **learnable** rather than fixed by hand.

[For review: a complementary direction is to study simple training-free selection rules under the same feedback budget.]
