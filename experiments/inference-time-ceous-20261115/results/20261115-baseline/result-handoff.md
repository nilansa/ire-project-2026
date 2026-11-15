# Private result handoff — exploratory SciFact cache-only study

- task_id: ire-final-submission-20261115-1943
- message_id: ceous-reviewed-baseline-execution-20261115-v1-result
- origin_surface: local Codex
- actual_sender: native Codex execution owner
- requested_action: Independent metrics auditor to inspect raw manifests, frozen predictions, summaries, source mappings, paired comparisons and costs before any public result claim.
- approval_status: Single run executed under the independent Sol review relayed in the execution handoff.
- reservation_id: CEOUS-BASELINE-H1H2-001
- classification: EXPLORATORY; both 405 DEV and 100 INNER were previously exposed. INNER never selected settings. Bootstrap intervals are descriptive and not multiplicity- or duplicate-group-adjusted.
- run_prefix: `/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/baseline-work/ceous-approved-study-20261115T175011Z-c8f4e559-9a3a-4d49-9652-f700eec75c19`
- run outcome: completed; return code 0; wall 1377.94 seconds; shared cap 1800 seconds.
- cost scope: frozen vectors and teacher caches; zero fresh teacher calls. Reported CPU timing is warm exact-search/update work and excludes encoder, teacher inference and full deployment latency. Logical per-arm pair/scan/slot budgets and actual harness work are retained separately in the raw output.
- code commit: https://github.com/nilansa/ire-project-2026/commit/b62f385f0b8b0fa7b01d056e24ec7fc9a9394eb1
- immutable file manifest: `/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-final-submission-20261115-1943/shared/baseline-work/ceous-approved-study-20261115T175011Z-c8f4e559-9a3a-4d49-9652-f700eec75c19-file-manifest.json`

The table shows the exact DEV-selected MedCPT arms on both partitions; BGE and all other teacher/grid arms remain in the raw summaries. Outside gains and losses are judged-positive totals at rank 20: gains are outside the original 50, while losses are relevant items from the original top 20 missing in the final top 20.

## DEV

| Arm | nDCG@10 | MRR@10 | Recall@20 | Outside-50 gains@20 | Initial-top20 losses@20 | Teacher pairs | Scans | Slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense (`dense`) | 0.6504 | 0.6154 | 0.8440 | 0 | 0 | 0.0000 | 1.0000 | 50.0000 |
| deep100 (`deep100`) | 0.6504 | 0.6154 | 0.8440 | 0 | 0 | 0.0000 | 1.0000 | 100.0000 |
| MedCPT rerank20 (`rerank20:medcpt`) | 0.7443 | 0.7206 | 0.8440 | 0 | 0 | 20.0000 | 1.0000 | 50.0000 |
| MedCPT rerank50 diagnostic (`rerank50:medcpt`) | 0.7538 | 0.7290 | 0.8743 | 0 | 2 | 50.0000 | 1.0000 | 50.0000 |
| DEV selected PRF (`rocchio:0.05`) | 0.6494 | 0.6133 | 0.8416 | 0 | 1 | 0.0000 | 2.0000 | 100.0000 |
| DEV selected weighted (`weighted:medcpt:1:t4`) | 0.7415 | 0.7197 | 0.8597 | 3 | 2 | 20.0000 | 2.0000 | 100.0000 |
| DEV selected TOUR (`tour:medcpt:0.8:stop0:t4`) | 0.7476 | 0.7238 | 0.8531 | 3 | 5 | 20.0000 | 2.0000 | 100.0000 |
| DEV selected H2 (`h2:medcpt:0.8:10:t4`) | 0.7451 | 0.7244 | 0.8605 | 5 | 3 | 20.0000 | 2.0000 | 100.0000 |
| H2 matched TOUR (`tour:medcpt:0.8:stop0:t4`) | 0.7476 | 0.7238 | 0.8531 | 3 | 5 | 20.0000 | 2.0000 | 100.0000 |

## INNER

| Arm | nDCG@10 | MRR@10 | Recall@20 | Outside-50 gains@20 | Initial-top20 losses@20 | Teacher pairs | Scans | Slots |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense (`dense`) | 0.6630 | 0.6255 | 0.8300 | 0 | 0 | 0.0000 | 1.0000 | 50.0000 |
| deep100 (`deep100`) | 0.6630 | 0.6255 | 0.8300 | 0 | 0 | 0.0000 | 1.0000 | 100.0000 |
| MedCPT rerank20 (`rerank20:medcpt`) | 0.7538 | 0.7355 | 0.8300 | 0 | 0 | 20.0000 | 1.0000 | 50.0000 |
| MedCPT rerank50 diagnostic (`rerank50:medcpt`) | 0.7915 | 0.7752 | 0.8875 | 0 | 1 | 50.0000 | 1.0000 | 50.0000 |
| DEV selected PRF (`rocchio:0.05`) | 0.6634 | 0.6263 | 0.8300 | 0 | 0 | 0.0000 | 2.0000 | 100.0000 |
| DEV selected weighted (`weighted:medcpt:1:t4`) | 0.7653 | 0.7428 | 0.8600 | 0 | 0 | 20.0000 | 2.0000 | 100.0000 |
| DEV selected TOUR (`tour:medcpt:0.8:stop0:t4`) | 0.7603 | 0.7372 | 0.8700 | 0 | 0 | 20.0000 | 2.0000 | 100.0000 |
| DEV selected H2 (`h2:medcpt:0.8:10:t4`) | 0.7579 | 0.7400 | 0.8700 | 0 | 0 | 20.0000 | 2.0000 | 100.0000 |
| H2 matched TOUR (`tour:medcpt:0.8:stop0:t4`) | 0.7603 | 0.7372 | 0.8700 | 0 | 0 | 20.0000 | 2.0000 | 100.0000 |

- DEV judged-positive coverage from frozen deep-100 rows: 405 queries; original 50 covers 402/454 (misses 52), deeper 100 covers 422/454 (misses 32); queries with zero judged positives in the pool: 46 and 28, respectively.
- INNER judged-positive coverage from frozen deep-100 rows: 100 queries; original 50 covers 102/114 (misses 12), deeper 100 covers 104/114 (misses 10); queries with zero judged positives in the pool: 11 and 9, respectively.

These are judgment-relative descriptive values pending independent audit, not evidence of significance or a novelty claim. Judged-positive misses include only available qrels; unjudged documents are not asserted irrelevant.
