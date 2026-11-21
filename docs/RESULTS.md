# Experiment results referenced by the final report

The result values below are recorded project measurements. **Different datasets, scopes and feedback budgets are not directly comparable.** See [the full evidence archive](https://github.com/nilansa/ire-project-2026/tree/archive/pre-final-cleanup-20261121) for original scripts, source-linked logs and metric audits.

| Experiment | Raw or reference | Feedback result | Main interpretation |
|---|---:|---:|---|
| FiQA-200 query-wise, nDCG@10 | Dense10 TTT 0.5060 | Coverage TTT 0.4807 | Coverage trails dense; paired Δ = −0.0253 (95% exploratory interval [−0.0433, −0.0075]) |
| FiQA-200 MMR, nDCG@10 | Dense10 TTT 0.5060 | MMR TTT 0.5046 | Similar descriptive point estimate; difference interval spans zero |
| FiQA-200 hybrid, nDCG@10 | Dense10 TTT 0.5060 | Hybrid TTT 0.5002 | Includes BM25 out-of-dense-pool candidates |
| HotpotQA-HN DEV400, nDCG@10 | Dense10 TTT 0.7195 | Coverage TTT 0.7298 | Paired Δ = +0.0103, interval [−0.0041, +0.0251]; preregistered result PARK |
| HotpotQA-HN POST600, nDCG@10 | Dense10 TTT 0.7447 | Coverage TTT 0.7628 | Post-decision descriptive analysis only |
| FiQA task-wise 5-fold, held-out nDCG@10 | Raw 0.4672 | Adapted 0.4631 | Different experiment and feedback scope |
| SCIDOCS ReFIT, Recall@100 (%) | Raw 37.8067 | ReFIT 37.8717 | Did not reproduce published 40.1 value |
| SciFact, nDCG@10 on DEV405 | Rerank20 0.7443 | TOuR-style 0.7476 | Extra full-corpus scan; exposed queries |

When explaining the mechanism, keep *teacher-selected documents*, *unscored in-pool promotion* and *outside-pool discovery* distinct. The new main-branch source files are a curated evaluation handoff, not an independent rerun.
