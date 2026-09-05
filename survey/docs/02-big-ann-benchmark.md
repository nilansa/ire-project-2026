# Big-ANN NeurIPS 2023 Benchmark

## Links

- [Official competition page](https://big-ann-benchmarks.com/neurips23.html)
- [Benchmark repository](https://github.com/harsha-simhadri/big-ann-benchmarks)
- [Track rules and datasets](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md)
- [Results paper — arXiv](https://arxiv.org/abs/2409.17424)
- [Results paper — OpenReview](https://openreview.net/forum?id=dB6W56wQL9)
- [Results paper — PDF](https://openreview.net/pdf?id=dB6W56wQL9)

## What it tests

The benchmark evaluates practical ANN variants under controlled compute, emphasizing search accuracy and efficiency rather than downstream model learning.

| Track | What is being tested | Workload |
|---|---|---|
| **Filtered** | Vector search subject to metadata predicates. | YFCC-10M CLIP vectors; each query also specifies one or two required tags. |
| **Out-of-distribution** | Search when query and database vectors follow different distributions or modalities. | Yandex Text-to-Image: image database vectors and text-query vectors. |
| **Sparse** | Maximum-inner-product search over very high-dimensional sparse learned representations. | 8.8M MS MARCO passages and queries encoded by SPLADE. |
| **Streaming** | Maintaining a compact, accurate index during online insertions, deletions, and searches. | MS Turing runbook with roughly a 4:4:1 insertion/deletion/search ratio. |

## Main evaluation

- **Filtered, OOD, Sparse:** recall-throughput trade-off, commonly reported as QPS at 90% recall; index build is constrained.
- **Streaming:** average recall across search checkpoints while completing the runbook within the time and memory limits.
