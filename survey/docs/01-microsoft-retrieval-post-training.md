# Microsoft Research: Efficient and Accurate Post-Training of Retrieval Models

**Primary source:** [Microsoft Research Fellowship challenge](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/)

**Principal investigators:** Gaurav Sinha, Kiran Shiragur, and Shivam Garg.  
**Additional collaborators:** Arun Iyer and Sonu Mehta.

## Objective

Post-train retrieval models using high-quality feedback from reward models, such as LLM cross-encoders or human feedback, to improve Search, Advertising, and RAG.

## Immediate research questions

- **Data selection:** Which query-document pairs should receive expensive reward scores?
- **Loss design:** How should pointwise, pairwise, or listwise feedback train a given retriever?
- **Efficiency:** How can repeated post-training iterations remain computationally affordable?

## Connection to this project

ANN can generate candidates before reward scoring. We study whether ANN approximation, coverage, distribution shift, and index staleness change the training signal and final retrieval quality under a fixed reward-model budget.
