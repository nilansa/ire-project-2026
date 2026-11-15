# Audited exploratory baseline results — 2026-11-15

Reservation `CEOUS-BASELINE-H1H2-001`, run `ceous-approved-study-20261115T175011Z-c8f4e559-9a3a-4d49-9652-f700eec75c19`, frozen code [b62f385](https://github.com/nilansa/ire-project-2026/commit/b62f385f0b8b0fa7b01d056e24ec7fc9a9394eb1). All stages exited 0 in 1377.9446 s (cap 1800 s), 17:54:10–18:17:08 UTC. No fresh encoder or teacher inference and no model-weight updates. DEV contains 206 fixed grid choices × 405 queries, not independent trials. INNER 100 was held for this selection only: all 505 queries had prior exposure. These are exploratory, judgment-relative results, not a fresh holdout.

## Primary MedCPT results

| Arm | DEV nDCG@10 | DEV recall@20 | INNER nDCG@10 | INNER recall@20 |
|---|---:|---:|---:|---:|
| dense | 0.6503740678 | 0.8440329218 | 0.6630479695 | 0.8300000000 |
| rocchio:0.05 | 0.6494235113 | 0.8415637860 | 0.6633732897 | 0.8300000000 |
| rerank20:medcpt | 0.7443376326 | 0.8440329218 | 0.7538223231 | 0.8300000000 |
| rerank50:medcpt | 0.7537853029 | 0.8742798354 | 0.7914690041 | 0.8875000000 |
| weighted:medcpt:1:t4 | 0.7414821751 | 0.8596707819 | 0.7653341447 | 0.8600000000 |
| tour:medcpt:0.8:stop0:t4 | 0.7476013832 | 0.8530864198 | 0.7602604864 | 0.8700000000 |
| h2:medcpt:0.8:10:t4 | 0.7451267839 | 0.8604938272 | 0.7579023377 | 0.8700000000 |

DEV selected each family separately; `rocchio:0.05` lost to dense on DEV, so dense is its deployment fallback although the PRF winner was evaluated on INNER. Deep100 equals dense in the top-20 metrics in both splits. Rerank50 uses 50 teacher pairs; all moved-teacher methods and rerank20 use 20, dense/PRF use zero. Rerank50 is a higher-feedback diagnostic. Weighted has the largest listed feedback-movement INNER nDCG descriptively; this did not select a winner. TOUR beat rerank20 on DEV by 0.003264, below the preregistered +0.01 screen while requiring an extra scan. H1 is a completed baseline phase and evidence for known teacher-guided query movement, not a new method or practical superiority claim. H2 is rejected for this screen and the tested uniform-covariance configuration is parked: it loses to the exact norm-matched TOUR by 0.002475 DEV and 0.002358 INNER and fails the +0.01 DEV screen. This does not reject all geometry. Do not tune on INNER.

## BGE sensitivity (separately DEV-selected)

| Arm | DEV nDCG@10 | DEV recall@20 | INNER nDCG@10 | INNER recall@20 |
|---|---:|---:|---:|---:|
| rerank20:bge | 0.715898 | 0.844033 | 0.736486 | 0.830000 |
| rerank50:bge | 0.713471 | 0.863169 | 0.763028 | 0.890000 |
| weighted:bge:1:t0.5 | 0.720400 | 0.857202 | 0.740604 | 0.850000 |
| tour:bge:0.8:stop0:t0.5 | 0.721733 | 0.850617 | 0.739113 | 0.846667 |
| h2:bge:3.2:1:t4 | 0.720630 | 0.859671 | 0.728287 | 0.850000 |

Teacher is not selected using INNER. Primary MedCPT revision `71caf65d4927987813984f54c284405a13fcca49`; BGE revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. The historical cache contains 50 scored pairs/query/teacher, a separate acquisition overhead; this execution made zero fresh calls. Rankings have no additional final teacher rescore; adding one requires accounting for its cost.

## Where relevant documents move

All 505 have positive qrels: 568 judged-relevant query-document pairs, 504 in the initial 50, 526 in initial 100 (+22), and 42 beyond 100. Zero-positive pools: 57 at 50 and 37 at 100; 22 queries have additional judged relevant documents at 51–100. Unjudged documents are not known negatives. F20 is the original top 20, even for the 50-feedback diagnostic. Gains/losses below count judged-relevant documents versus dense top20 and fixed-pool counterparts rescore only U50.

| Split / method | F20 gain / loss | U50 minus F20 gain / loss | Outside U50 gain / loss | Fixed-pool nDCG@10 / recall@20 |
|---|---:|---:|---:|---:|
| DEV weighted:medcpt:1:t4 | 0 / 2 | 7 / 0 | 3 / 0 | 0.739622 / 0.854733 |
| DEV tour:medcpt:0.8:stop0:t4 | 0 / 5 | 8 / 0 | 3 / 0 | 0.746131 / 0.850617 |
| DEV h2:medcpt:0.8:10:t4 | 0 / 3 | 7 / 0 | 5 / 0 | 0.744900 / 0.850617 |
| INNER weighted:medcpt:1:t4 | 0 / 0 | 3 / 0 | 0 / 0 | 0.765334 / 0.860000 |
| INNER tour:medcpt:0.8:stop0:t4 | 0 / 0 | 4 / 0 | 0 / 0 | 0.760439 / 0.870000 |
| INNER h2:medcpt:0.8:10:t4 | 0 / 0 | 4 / 0 | 0 / 0 | 0.757902 / 0.870000 |

DEV H2 gains 12 and loses 3, 5 of its gains outside U50; TOUR gains 11 and loses 5, 3 outside. INNER each gains 4 and loses 0, with no outside-U50 gains: the recall improvements are within-initial-50 promotion. Rerank50 gains/loses 17/2 DEV and 6/1 INNER. See full region and per-query details in the compressed CSVs and summaries.

## Cost, reproducibility and limitations

Dense/rerank20 use one full scan and 50 returned slots; deep100 uses one scan/100 slots; non-stopped full-corpus feedback methods use two scans/100 slots. Pool variants use one scan/50 slots. See prediction manifests for all per-arm logical budgets, physical work, union sizes and p50/p95; these are not equivalent total compute. Warm CPU INNER p50 TOUR 1.673 ms and H2 5.433 ms (~3.25×), excluding encoder and teacher inference, on a shared Apple M4 Max, 16 CPUs, 64 GB, macOS 26.7.1, one configured thread. These are not end-to-end latencies.

Independent transient audit and duplicate correction are described in [AUDIT_RESULTS.md](AUDIT_RESULTS.md). INNER paired H2−matched TOUR descriptive bootstrap uses Python `random.choices`, seed 20261115, 2000 resamples and sorted order statistics 49 and 1949: approximately [−0.01434, +0.00959]. This differs from interpolated NumPy percentiles and establishes no significance; multiplicity and historical exposure remain limitations.

Run `python3 experiments/inference-time-ceous-20261115/replay_metrics.py` from the repository root to independently reconstruct metric checks on packaged already-computed predictions (not a new experiment). See [package](results/20261115-baseline/) and its original and derived manifests, frozen selection, review certificate, alignment, full execution commands, status, software/hardware and per-query records. Corpus-row IDs and selected qrels contain no abstracts or query text and derive from BEIR SciFact; [SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md) attributes claims/annotations CC BY 4.0, abstracts ODC-By 1.0, and code Apache 2.0; [BEIR](https://github.com/beir-cellar/beir) transformed data. Full retrieval replay also requires the original frozen embeddings and teacher caches at the hash-bound paths in alignment and the execution script, which are not distributed; it is not fresh-machine end-to-end reproducible from this package alone. Figures: [INNER methods](plots/inner-methods.png), [pool coverage](plots/pool-coverage.png).
