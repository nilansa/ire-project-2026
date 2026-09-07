# Research Plan (Draft)

**Team:** **Koios** (Κοῖος; commonly anglicized as *Coeus*), the Greek Titan.

## 1. Problem / Idea

**Source problem:** Microsoft Research — [Efficient and accurate post-training of retrieval models](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/)

Microsoft frames retrieval post-training around using high-quality feedback from reward models such as LLM-based cross-encoders or human feedback. One of its immediate questions is **data selection**: which query-document pairs should be selected for reward scoring so that post-training improves downstream retrieval and/or RAG performance?

### Our focus

Before a reward model can score query-document pairs, a retrieval/search stage must produce candidate documents. We will study how this **candidate-generation search** affects the final retriever.

> **Primary research question:** Does the ANN/search method and its operating hyperparameters used to generate candidates for reward scoring affect the final post-trained retriever, when the retriever, reward model, feedback budget, training objective, and evaluation are held fixed?

Examples of interventions include the ANN family/search procedure and operating parameters such as search depth, `efSearch`, or `nprobe`.

## 2. Why This Matters

- Microsoft explicitly identifies **which query-document pairs receive expensive reward-model feedback** as a core post-training problem, together with the computational cost of large reward models.
- Under a fixed feedback budget, every method can spend the **same full number of reward-model calls**. The question is which documents those calls are spent on.
- A candidate generator decides which documents are even available to be labeled/scored. Documents not surfaced by candidate search cannot contribute reward supervision in that iteration.
- [ANCE (ICLR 2021)](https://www.microsoft.com/en-us/research/publication/approximate-nearest-neighbor-negative-contrastive-learning-for-dense-text-retrieval/) showed that changing the training-negative distribution by retrieving global hard negatives from an ANN index materially changes dense-retriever learning and performance.
- [RocketQA (NAACL 2021)](https://aclanthology.org/2021.naacl-main.466/) retrieves candidate hard negatives and then uses a stronger cross-encoder to denoise them, demonstrating the importance of both the candidate pool and the teacher signal.
- [LADR (SIGIR 2023)](https://doi.org/10.1145/3539618.3591715) shows that candidate generation itself can be a structured search procedure: lexical retrieval seeds a dense document-proximity graph, and the search policy controls which documents are explored.
- A recent [survey of dense text retrieval (TOIS 2024)](https://doi.org/10.1145/3637870) summarizes that dense retrievers are sensitive to negative quality, especially for hard negatives and false negatives.

This motivates treating the ANN/search stage not only as an inference-speed component, but as a component that can change the **training data seen by the post-training pipeline**.

## 3. Relevant Literature and Research Framing

### ANCE — Xiong et al., ICLR 2021
[Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://www.microsoft.com/en-us/research/publication/approximate-nearest-neighbor-negative-contrastive-learning-for-dense-text-retrieval/)

- Uses an ANN index over the corpus to retrieve global hard negatives during training.
- Shows that realistic, globally retrieved negatives improve learning compared with conventional negative sampling.
- Directly motivates our controlled question: if ANN-retrieved candidates shape the training signal, how sensitive is the final retriever to the **search method and operating point** that generated those candidates?

### RocketQA — Qu et al., NAACL 2021
[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/)

- Retrieves top-ranked passages as candidate hard negatives.
- Uses a cross-encoder to filter/denoise these candidates before training the dual encoder.
- Closely matches the retrieve → expensive scorer → retriever-training structure we want to study.

### LADR — Kulkarni et al., SIGIR 2023
[Lexically-Accelerated Dense Retrieval](https://doi.org/10.1145/3539618.3591715)

- Starts from lexical retrieval results and explores a document-proximity graph using dense scores.
- Its proactive and adaptive variants explicitly change **which documents enter the explored candidate set under a search budget**.
- LADR itself is an inference-time retrieval method, not a reward-feedback/post-training method. For our project it is useful evidence that candidate construction is a meaningful algorithmic choice, which we then study inside the Microsoft feedback-selection setting.

### Dense Retrieval Survey — Zhao et al., TOIS 2024
[Dense Text Retrieval Based on Pretrained Language Models: A Survey](https://doi.org/10.1145/3637870)

- Summarizes evidence that dense-retriever training is sensitive to negative-sample quality.
- Highlights the trade-off between informative hard negatives and false negatives, motivating careful candidate selection.

### ADAM — Tao et al., ACL Findings 2024
[ADAM: Dense Retrieval Distillation with Adaptive Dark Examples](https://aclanthology.org/2024.findings-acl.692/)

- Studies cross-encoder-to-dual-encoder distillation and shows that the examples presented to the teacher/student matter for how much useful supervision is transferred.

## 4. Research Question

**Main question**

> Does the ANN/search method used to generate candidates affect the final post-trained retriever?

**Sub-questions**

1. How do ANN/search choices and hyperparameters change candidate overlap, hardness, and teacher-score distributions?
2. Do these changes translate into measurable differences in the final retriever?
3. Under the **same fully spent reward-scoring budget**, can a cheaper search procedure produce equally useful supervision and essentially the same final retriever quality as a more expensive/high-recall search?

## 5. Experimental Plan

### Keep fixed

- corpus and training/evaluation queries;
- starting dense-retriever checkpoint;
- document/query embeddings used for the candidate-generation comparison;
- **reward-scoring budget: the same number of query-document pairs are scored in every condition**;
- reward model / cross-encoder;
- post-training loss, optimizer, steps, batch size, and training data budget;
- downstream evaluation protocol.

### Change only candidate generation

Initial comparison:

- exact nearest-neighbor search as a reference;
- HNSW at multiple `efSearch` operating points;
- IVF-style search at multiple `nprobe` operating points.

A later extension can include structured/hybrid candidate-generation procedures such as LADR-style lexical seeding plus graph exploration.

Where possible, compare both **matched-recall** and **matched-compute/latency** operating points.

### Measure

**Before training**
- ANN recall against exact search;
- overlap/Jaccard of candidate sets;
- candidate rank changes;
- cross-encoder/reward-score distribution;
- positive / useful-negative yield among scored candidates.

**After identical post-training**
- MRR / nDCG / Recall@k on the retrieval benchmark;
- difference from the exact-search candidate baseline;
- relationship between candidate-set differences and final retriever performance.

## 6. Research Threads

1. **Candidate-generation sensitivity:** isolate ANN/search method and search hyperparameters as the intervention.
2. **Feedback-budget trade-off:** spend the same complete reward-model budget in every condition and test whether different candidate generators use that budget more or less effectively.
3. **Mechanism analysis:** connect ANN recall/overlap/hardness to the resulting teacher labels and gradient/training signal.
4. **Iterative post-training:** after establishing the one-iteration effect, study whether the effect compounds when the retriever/index is refreshed over multiple rounds.

### Additional training-free candidate-selection directions

| Direction | How it fits the Microsoft feedback-selection problem |
|---|---|
| **Diversity-aware candidate selection** | If the ANN top results are redundant, use a training-free diversification step before reward scoring so the fixed teacher budget covers different attributes/facets. [Barman et al., *Welfarist Formulations for Diverse Similarity Search*, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/abaab8d9908c3df048fbfc0802dc778e-Abstract-Conference.html) gives a relevance-diversity objective that can be applied on top of a standard ANN method. |
| **Query decomposition** | For complex/multi-hop queries, decompose the original query into subqueries, retrieve candidates for each, merge/deduplicate them, and spend the **same total reward-model budget** on the merged pool. This can make the feedback budget cover different evidence needs that one query embedding may miss. A concrete training-free example is [Ammann et al., *Question Decomposition for Retrieval-Augmented Generation*, ACL SRW 2025](https://aclanthology.org/2025.acl-srw.32/). |

For the query-decomposition direction, if we want to isolate **data selection only**, the selected documents should still be attached to the **original query** for post-training; training directly on the generated subqueries would be a separate query-augmentation intervention.

## 7. Resources and Feasibility

- **Data:** standard dense-retrieval benchmarks such as MS MARCO passage retrieval for the initial controlled study.
- **Search tooling:** FAISS / HNSW implementations; exact search for the reference condition.
- **Models:** one fixed dense retriever and one fixed cross-encoder/reward model.
- **Compute:** small candidate-generation studies can run locally; retriever post-training and larger sweeps can use available GPU compute.
- The first milestone is deliberately small: establish whether changing only the ANN candidate-generation stage produces a reproducible downstream signal before scaling the study.

---

## Current one-line formulation

> **Holding the retriever, teacher, fully spent feedback budget, and post-training procedure fixed, how does the ANN/search procedure used to choose candidate query-document pairs for expensive feedback affect the final retriever?**
