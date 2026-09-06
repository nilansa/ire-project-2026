# LADR / Big-ANN sparse-track experiment scaffold

**Status:** scaffold only; no evaluation results are claimed.

## Hypothesis

Compare the original Big-ANN'23 sparse-track algorithms on the MS MARCO/SPLADE 6,980-query set using **both** Big-ANN recall/QPS and MS MARCO qrels metrics: MRR@10 and Recall@1000, with nDCG only where graded qrels exist. The working hypothesis is that Big-ANN's ANN-throughput order will not necessarily equal the downstream relevance order because approximate SPLADE-neighbor overlap measures candidate fidelity to an embedding space, while qrels measure judged relevance and rank position.

## In-scope methods

The original sparse-track target set is:

1. `linscan` baseline
2. `pyanns`
3. `shnsw` / GrassRMA-style sparse HNSW entry
4. `nle`
5. `sustech-whu`
6. `cufe`

Do not substitute entries from the 2024 ongoing/closed-source leaderboard. Big-ANN's original Azure D8lds_v5 full-track QPS order at the published 90% ANN-recall threshold was `pyanns` (6499.65), `shnsw` (5078.45), `NLE-Full` (1314.19), `nle` (1312.96), `sustech-whu` (788.17), `cufe` (97.86), and `linscan` (95.10). Those are reference leaderboard values, not measurements on this Mac.

## Metric contract

- **Big-ANN Recall@10:** mean overlap between the returned ten integer row IDs and the exact SPLADE maximum-inner-product neighbors in `base_full.dev.gt`. It is not relevance recall.
- **Big-ANN QPS:** throughput at an operating point whose Big-ANN Recall@10 is at least 0.90; the published reference uses Azure D8lds_v5 and the full sparse track.
- **MS MARCO MRR@10:** compute from official qrels using the first judged relevant passage in each submitted ranking.
- **MS MARCO Recall@1000:** compute from official qrels over the first 1,000 submitted passages.
- **nDCG:** compute only for query sets with graded judgments, such as TREC DL19/DL20. Do not invent graded labels for MS MARCO Dev small, whose standard qrels are binary.

Keep the two recall columns separate in every result table. A high Big-ANN Recall@10 can coexist with low MS MARCO Recall@1000 or MRR@10 if the SPLADE exact-neighbor set is not the same as the judged relevant set, or if relevant passages are ranked below the cutoff.

## Identity and join investigation

### What the sources establish

- Big-ANN's [`SparseDataset`](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/89a3abaafa63dda46b94b308bdf039e699841b3b/benchmark/datasets.py#L987-L1055) hard-codes `nq = 6980`, loads `queries.dev.csr.gz`, uses `base_full.csr.gz` with 8,841,823 rows, and names the public ground truth `base_full.dev.gt`.
- Big-ANN's [sparse dataset note](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/89a3abaafa63dda46b94b308bdf039e699841b3b/dataset_preparation/sparse_dataset.md) calls the public queries `dev.small` and describes them as MS MARCO queries encoded with `naver/splade-cocondenser-ensembledistil`.
- Microsoft’s [MS MARCO Passage Ranking README](https://github.com/microsoft/MSMARCO-Passage-Ranking/blob/28b369395881cdc50eb69173dcf2f94eb247395e/README.md#query-to-queryid) defines `queries.dev.small.tsv` as the public dev subset and describes the corresponding `qrels.dev.small.tsv` as the qrels for the public set. The README’s data description gives the passage collection as 8,841,823 rows and the public dev subset as roughly 6,800 queries.

Together these sources strongly identify the intended semantic split as MS MARCO Dev small, but they do not prove a row-by-row byte identity between the published SPLADE CSR query rows and `queries.dev.small.tsv` QIDs.

### Blocking join uncertainty

The Big-ANN public format contains CSR values and integer row positions only. `read_sparse_matrix` reads `(data, indices, indptr)` and creates a CSR matrix; it has no QID/PID field. The sparse wrappers insert base rows in iterator order and return integer IDs. The repository contains no passage-PID sidecar, query-QID sidecar, or preprocessing script that proves `row i == MS MARCO PID i` (or records a different permutation). Therefore qrels evaluation is **blocked until an alignment artifact is found or the SPLADE preprocessing is reproduced and verified**.

Do not treat the following as proof: equal row counts, the `dev.small` name, or Big-ANN ANN ground truth. Also retain the [Big-ANN query-embedding issue](https://github.com/harsha-simhadri/big-ann-benchmarks/issues/320) as a data-quality warning; it reports a suspected query/base embedding mismatch but does not resolve the PID/QID mapping.

## Planned extraction and evaluation

The harness returns a matrix of integer IDs through each algorithm's `get_results()` method. A faithful relevance run must request `k=1000`, preserve query order, and write a TREC-compatible run after applying a verified row-to-PID mapping. Then evaluate against the official MS MARCO qrels. The repository's checked-in `res_*.csv` and operating-point text files contain aggregate ANN recall/QPS only; they are not top-k document dumps and cannot be reused for qrels metrics.

The cheapest valid sequence is:

1. Resolve and mechanically validate the row↔PID and query-row↔QID mapping.
2. Reuse any official top-k dumps if a source publishes them; otherwise run `linscan` as the correctness baseline.
3. Run `pyanns`, `shnsw`, `nle`, `sustech-whu`, and `cufe` in original-track order, recording top-1000 predictions, Big-ANN overlap/QPS, and qrels metrics separately.
4. Compare the measured rank orders against the original Big-ANN leaderboard and the published LADR dense TAS-B numbers only with explicit “different objective, model, hardware, and/or candidate pipeline” labels.

## Mac feasibility and blockers

The official Big-ANN harness is Docker/Ubuntu-oriented. Its sparse Dockerfiles build native Rust/C++/Python extensions at image-build time; no ARM64 binaries or top-k result dumps are checked into the benchmark repository. ARM compatibility is therefore unverified for every target method. The current Big-ANN Dockerfile also points `sustech-whu` at a GitHub repository that was not reachable during this audit, so that method has an additional source-availability blocker.

The full CSR download is listed as 5.5 GB compressed for 8,841,823 rows; the query CSR is listed as 1.8 MB. A rough in-memory CSR estimate from ~120 nonzeros per base row is about 8.5 GB before native index overhead, so a full top-five rebuild is not a small laptop experiment. Prefer an alignment artifact and reusable top-k outputs. If those do not exist, the first runnable milestone is `linscan` after mapping resolution; the complete top-five evaluation is estimated at **one working day after the mapping and a compatible build environment are available**, with no result ETA asserted before then.

## Evidence used

- [Big-ANN NeurIPS'23 repository](https://github.com/harsha-simhadri/big-ann-benchmarks)
- [Big-ANN sparse dataset format](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/89a3abaafa63dda46b94b308bdf039e699841b3b/dataset_preparation/sparse_dataset.md)
- [Big-ANN original sparse leaderboard](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/89a3abaafa63dda46b94b308bdf039e699841b3b/neurips23/Azure_D8lds_v5_table.md)
- [MS MARCO Passage Ranking data and qrels](https://github.com/microsoft/MSMARCO-Passage-Ranking)
- [LADR paper](https://arxiv.org/abs/2307.16779) and [LADR implementation](https://github.com/terrierteam/pyterrier_dr/blob/f790d83b420ab70abafe8c608c8d3a086c430f3b/pyterrier_dr/flex/ladr.py)

## Run Status / Results — 2026-09-07 (Asia/Kolkata)

Completed native macOS arm64 runs are ANN-only and not leaderboard-comparable:
Big-ANN Recall@10 is overlap with the SPLADE-neighbor ground truth, not MS
MARCO relevance recall.

| Method | Dataset / queries | k | ANN Recall@10 | QPS | Outcome |
|---|---|---:|---:|---:|---|
| Organizer linscan | `sparse-full` / 6,980 | 1,000 | 0.9999426934 | 72.461 | completed |
| Organizer linscan (`run_linscan.py`) | `sparse-small` / 6,980 | 10 | — | 26,782.815 | completed smoke |
| PyANNS | `sparse-small` / 6,980 | 10 | 0.927521490 | 13,877.079 | completed smoke |
| SHNSW / GrassRMA | `sparse-small` / 6,980 | 10 | 0.689527 | 33,351.2 | completed smoke |
| CUFE | `sparse-small` / 6,980 | 10 | 0.999914040 | 14,995.173 | completed smoke |
| SUSTech-WHU | `sparse-small` / 6,980 | 10 | — | — | not run: source repository 404 |
| NLE | `sparse-small` / 6,980 | 10 | — | — | no completed smoke; ARM build remained detached |

Environment: macOS Apple Silicon (`arm64`), Python 3.12.13, `uv` 0.11.17,
task-local `shared/experiment-run/.venv`; Docker 29.7.2 client was present but
the Colima Docker server was unavailable. The full linscan run used 16 threads
and produced 6,980,000 zero-based row-ID records.

No MS MARCO MRR@10 or Recall@1000 is reported. Query-row → QID and the
100,000-row prefix row → PID were checked, but the full 8,841,823-row
base-row → PID mapping was not exhaustively verified. Joining the full row-ID
run to qrels would therefore be an unverified result. Binary MS MARCO qrels
also do not support a graded nDCG@10 claim here.

Outputs and logs: `shared/experiment-run/results/linscan-full-rowids.trec`,
`shared/experiment-run/results/linscan-full-ann-metrics.json`,
`shared/experiment-run/results/linscan-small.trec`, and
`shared/experiment-run/logs/linscan-full.log`; method-specific evidence is in
`codex-to-chatgpt/pyanns.md`, `shnsw.md`, `cufe.md`, and `sustech.md`.

Next valid step: obtain or reproduce an exact full-base row → PID artifact,
mechanically verify it, then convert the `k=1,000` linscan row-ID run to PIDs
and evaluate the official qrels. Do not infer downstream relevance metrics
from the ANN measurements.
