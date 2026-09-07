# Research Plan (Draft)

**Team:** Ceous

## Brief of the idea

## 1. What is the idea?

**Source problem:** Microsoft Research — [Efficient and accurate post-training of retrieval models](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/)

Microsoft Research Fellowship has posed the research challenge **“Efficient and accurate post-training of retrieval models.”** The challenge considers post-training retrieval models using high-quality relevance/reward feedback, for example from LLM-based cross-encoders or human feedback. Obtaining such feedback can be expensive, especially when the reward model is large. One of the immediate questions is therefore **data selection**: which query-document pairs should be selected for reward scoring so that post-training improves downstream retrieval and/or RAG performance?

Under a fixed feedback budget, the data-selection policy determines the empirical distribution of query-document pairs for which supervision is obtained. The post-training loss is then computed from these scored/labeled pairs, so the selection policy determines which examples contribute training signal to the retriever.

**Our focus.** Before reward scoring, a candidate-generation/search stage determines which documents are available for selection. We want to study whether the ANN/search procedure used to construct this candidate pool changes the supervision available to post-training enough to affect the final retriever.

> **Primary research question:** Holding the retriever, reward model, feedback budget, training objective, and evaluation fixed, how does the ANN/search procedure used to generate candidates for reward scoring affect the final post-trained retriever?

### a. Why is it important? If it were solved what would improve?

- Reward-model or human feedback is expensive, so under a fixed budget it matters which query-document pairs receive that feedback.
- Candidate generation constrains the set of pairs that can be scored. A document that is never surfaced by the search stage cannot provide supervision in that post-training iteration.
- Dense-retriever training is known to be sensitive to the negatives/candidates used for training. If candidate-generation choices systematically change this supervision distribution, then the search layer is part of the post-training design rather than only an inference-time engineering choice.
- Understanding this effect would make the data-selection stage more principled: we could separate improvements caused by the reward model or loss from improvements caused by which candidates were exposed to them.

### b. What is the relevant literature? Does the literature acknowledge this gap? Were there attempts at it.

**[For review: ANCE — Xiong et al., ICLR 2021]**  
[Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://openreview.net/forum?id=zeFrfgyZln)

- ANCE uses approximate nearest-neighbor retrieval over the corpus to obtain global hard negatives for dense-retriever training.
- It shows that the negative-sampling distribution materially affects optimization and retrieval performance, and that ANN-mined global negatives better approximate an informative sampling procedure than conventional local/in-batch negatives.
- This supports the premise that **which documents are selected for the loss matters**. It does not, by itself, establish that changing the ANN algorithm or ANN operating point changes the final trained retriever; that is the controlled question we want to test.

**[For review: RocketQA — Qu et al., NAACL 2021]**  
[RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/)

- RocketQA retrieves candidate hard negatives and uses a stronger cross-encoder to denoise/filter them before training the dual encoder.
- This is close to the structure of the Microsoft challenge: a retrieval stage proposes candidate query-document pairs, a more expensive model provides higher-quality feedback, and the resulting supervision is used to improve the retriever.

**[For review: LADR — Kulkarni et al., SIGIR 2023]**  
[Lexically-Accelerated Dense Retrieval](https://arxiv.org/abs/2307.16779)

- LADR is an inference-time retrieval method, not a reward-feedback or post-training method.
- It first obtains seed documents using a lexical retriever (BM25 in the paper's experiments), then explores a precomputed document-proximity graph and applies dense scoring to the explored documents.
- Its adaptive variant selectively expands the neighborhoods of the most promising documents, explicitly allocating a limited dense-scoring budget to promising **document neighborhoods**. This is a more precise description than treating them as generic “clusters” or “regions.”
- For our problem, LADR is useful as evidence that candidate construction/search can itself be a structured budget-allocation decision before an expensive scoring stage.

**[For review: Dense Retrieval Survey — Zhao et al., TOIS 2024]**  
[Dense Text Retrieval Based on Pretrained Language Models: A Survey](https://doi.org/10.1145/3637870)

- The survey summarizes evidence that sampled negatives have a significant effect on dense-retriever performance, and that hard negatives are informative but can also contain false negatives.
- This supports the weaker premise that **candidate/negative selection affects the training signal**. It would be too strong to say that the survey validates our hypothesis that the ANN itself determines what the loss sees; it does not directly compare ANN families or ANN hyperparameters as the causal intervention.

**[For review: ADAM — Tao et al., Findings of ACL 2024]**  
[ADAM: Dense Retrieval Distillation with Adaptive Dark Examples](https://aclanthology.org/2024.findings-acl.692/)

- ADAM studies cross-encoder-to-dual-encoder distillation and adaptive example selection, further motivating the view that the examples on which expensive teacher supervision is obtained can affect what is transferred to the retriever.

The literature therefore motivates the importance of candidate/negative selection and expensive teacher feedback. Our specific hypothesis is narrower: **when the feedback budget and post-training procedure are fixed, does changing the search/ANN mechanism that constructs the candidate pool change the final post-trained retriever?**

## 2. What is the plan?

### a. What are the various threads you want to pursue to solve this (more literature survey can be one of them but you should tell which specific papers, problems, etc)? Does it justify the specified team size? Do you have the resources (skills and hardware) to execute it?

