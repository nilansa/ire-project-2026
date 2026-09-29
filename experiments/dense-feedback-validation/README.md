# Dense feedback-selection validation

This experiment is the decisive follow-up to the 100-direction lightweight screen.

It evaluates the general question:

> Under a fixed expensive-feedback budget, which queries and candidate documents should be scored so that dense-retriever post-training improves most reliably?

## Controlled factors

- Datasets: SciFact, NFCorpus and ArguAna, up to 300 judged queries each.
- Student: `sentence-transformers/all-MiniLM-L6-v2`.
- Update: top two query-encoder layers; document embeddings remain frozen.
- Teacher: `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Candidate generators: exact dense, HNSW, IVF-Flat and lexical+dense hybrid.
- Budgets: 4, 12 and 32 selected candidates/query on average.
- Splits: three deterministic 60/20/20 train/validation/evaluation splits.
- Evaluation: full-corpus exact NDCG@10, MRR@10 and Recall@100.

## Policies

1. Standard top-candidate feedback.
2. Random feedback control.
3. Difficulty-adaptive feature-space leverage.
4. Uniform uncertainty-diversity MMR.
5. Disagreement-adaptive hard-negative selection.
6. Disagreement-adaptive relevance-diversity MMR.
7. Disagreement-adaptive uniform-rank coverage.

The full seven-policy comparison uses exact candidates and budget 12. The preliminary winner and top-candidate control are additionally compared under every budget/backend combination.

## Reproduction

The GitHub Actions workflow `.github/workflows/dense-feedback-validation.yml` runs three datasets in parallel and commits the complete aggregated results to `final-results/` after all jobs succeed.

This is a transformer query-side post-training experiment. It does not claim that updating both query and document encoders would produce identical results.
