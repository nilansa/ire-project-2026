# Efficient and accurate post-training of retrieval models

**Source:** [Microsoft Research Fellowship — Research challenges](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/)

## Source screenshot

![Microsoft Research challenge: Efficient and accurate post-training of retrieval models](../assets/microsoft-retrieval-post-training.png)

## Verbatim text

**Principal Investigator(s):** Gaurav Sinha (Microsoft Research India), Kiran Shiragur (Microsoft Research India), and Shivam Garg (Microsoft Research AI Frontiers)

**Additional Microsoft collaborators:** Arun Iyer (Microsoft Research India), Sonu Mehta (Microsoft Research India)

**Summary:**

This challenge aims to develop novel mechanisms for post-training retrieval models using high quality feedback from reward models (such as LLM based cross encoders) to optimize downstream retrieval and/or generation (RAG) performance.

**Description:**

Post-training has emerged as a powerful technique for steering language models toward maximizing desired rewards. This approach presents a significant opportunity for retrieval; a domain often constrained by sparse training data (i.e., lacking relevance signals for most query-document pairs). The availability of high-quality reward models, such as LLM-based cross-encoders or human feedback, makes this avenue particularly promising.

We intend to develop and adapt post-training techniques specifically for retrieval models, targeting applications like Search, Advertising, and Retrieval-Augmented Generation (RAG).

Some of our immediate research questions include:

(1) Data Selection: Which queries and corresponding documents should be selected for scoring by reward models during the post-training stage to maximize downstream retrieval and RAG performance?

(2) Loss Formulation: What is the optimal post-training loss function given a specific retrieval architecture, reward feedback design (e.g., pointwise, pairwise, listwise), and application scenario?

(3) Computational Efficiency: How can we efficiently execute multiple post-training iterations, especially when the reward models are large and computationally expensive?

**Ideal collaborator:**

Ideal collaborators for this project include PhD students (and faculty) working in machine learning with a strong focus on reinforcement learning or information retrieval with an exposure to both theoretical and empirical research in these areas. Prior hands-on experience in working with large language models specifically in developing RL based post training algorithms will be extremely valuable.

**Eligible candidates:** PhD students and faculty
