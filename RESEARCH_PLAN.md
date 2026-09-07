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
- A candidate generator decides which documents are even available to be labeled/scored. Documents not surfaced by candidate search cannot contribute reward supervision in that iteration.
- [ANCE (ICLR 2021)](https://www.microsoft.com/en-us/research/publication/approximate-nearest-neighbor-negative-contrastive-learning-for-dense-text-retrieval/) showed that changing the training-negative distribution by retrieving global hard negatives from an ANN index materially changes dense-retriever learning and performance.
- [RocketQA (NAACL 2021)](https://aclanthology.org/2021.naacl-main.466/) retrieves candidate hard negatives and then uses a stronger cross-encoder to denoise them, demonstrating the importance of both the candidate pool and the teacher signal.
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

1. How do ANN choices/hyperparameters change candidate overlap, hardness, and teacher-score distributions?
2. Do these changes translate into measurable differences in the final retriever?
3. Under a fixed reward-scoring budget, can a cheaper ANN operating point produce essentially the same final retriever quality as a more expensive/high-recall search?

## 5. Experimental Plan

### Keep fixed

- corpus and training/evaluation queries;
- starting dense-retriever checkpoint;
- document/query embeddings used for the candidate-generation comparison;
- number of candidate pairs sent to the teacher;
- reward model / cross-encoder;
- post-training loss, optimizer, steps, batch size, and training data budget;
- downstream evaluation protocol.

### Change only candidate generation

Initial comparison:

- exact nearest-neighbor search as a reference;
- HNSW at multiple `efSearch` operating points;
- IVF-style search at multiple `nprobe` operating points.

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

1. **Candidate-generation sensitivity:** isolate ANN method and search hyperparameters as the intervention.
2. **Feedback-budget trade-off:** test whether cheaper search produces equally useful candidates under a fixed number of expensive teacher scores.
3. **Mechanism analysis:** connect ANN recall/overlap/hardness to the resulting teacher labels and gradient/training signal.
4. **Iterative post-training:** after establishing the one-iteration effect, study whether the effect compounds when the retriever/index is refreshed over multiple rounds.

## 7. Resources and Feasibility

- **Data:** standard dense-retrieval benchmarks such as MS MARCO passage retrieval for the initial controlled study.
- **Search tooling:** FAISS / HNSW implementations; exact search for the reference condition.
- **Models:** one fixed dense retriever and one fixed cross-encoder/reward model.
- **Compute:** small candidate-generation studies can run locally; retriever post-training and larger sweeps can use available GPU compute.
- The first milestone is deliberately small: establish whether changing only the ANN candidate-generation stage produces a reproducible downstream signal before scaling the study.

---

## Current one-line formulation

> **Holding the retriever, teacher, feedback budget, and post-training procedure fixed, how does the ANN/search procedure used to choose candidate query-document pairs for expensive feedback affect the final retriever?**
