# Audit and corrigendum — 2026-11-15

The root reports an actual independent Terra high audit at `/root/literature`: original qrels TSV and source corpus row IDs, raw predictions, no harness imports, file edits, new search or model inference. On 405 DEV and 100 INNER queries × eight primary arms including deep100, per-query nDCG@10 (all-positive IDCG) and recall@20 (all-positive denominator) matched the scored CSV within 1e-10. The auditor verified the supplied hashes and corpus. This publisher did not perform that independent audit; `replay_metrics.py` is a faithful reconstruction, not the original transient auditor's code. Root and Sol review accepted H1 baseline completion and rejected/parked the tested H2 configuration; do not attribute those decisions to this publisher.

## Duplicate corrigendum

The frozen protocol described the four exact-normalized duplicate pairs among the broader 809 as though duplicates could remain within a selected partition. In fact, 85 is in DEV and 86 is outside the selected 505; both members of 89/90, 870/871 and 1291/1292 are outside the 505. There are **zero exact-normalized duplicate pairs within the selected 505**. The frozen protocol and alignment remain byte-identical; no query split or result changes. Historical exposure and multiplicity limitations remain.

## Bindings and descriptive uncertainty

- Result handoff SHA-256 `9bf6784464f21657b8205f831558e107d442776b647d9fe8f249ba739a0a517f`; original file manifest `a88975ab927e66cd4b5e8c7bf73f5724788fa463c4223170d5c78f9a02fd320e`.
- DEV predictions `8a7b6b324cdb75d382cdc805d7a269ee66df33808c498fc76ec3d885a2ca3742`; CSV `9428b4373a801c592007d8328d91d6e7fcaa0506509429ace22d7befde8870c5`; summary `8d546b6eaab85bd9589f74f90b6872e79d16a7abae34ff1c28f17a92e617cc38`.
- INNER predictions `ebc571b37c802ebef53077165d334fd1bf56d4dd8cb6d7fb3c88040cd55da6d8`; CSV `10032940cb6b4b4d4cf4c756a292f5e7bf37a07b27cf683f94d0bf1377889514`; summary `4238867ed44ac2f6095d3adc63e697b9c63744c7533f1f38b4866ef426e66ab7`; prediction manifest `5ffa98d552343674edea14589c802893f4b6eb34bf341681e9c2e77c0581a919`.
- Selection `1e120ea6baf4730fa84d5ef28e8a542680b397c5ea331f25fb5c1af607f08fe9`; alignment `0bf495b424fc52b7bba6fdf6ab32354f4918e7aa52f17e3a68915f0650b4453d`; original train qrels `a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d`; corpus `dec31c8182f3d744c7d2c09423756fd1d17cbef75808db13ba01cc0aab4d1ac6`. Hashes refer to original uncompressed bytes.
- INNER paired bootstrap H2−matched TOUR: sort query IDs, Python `random.choices`, seed 20261115, 2000 resamples, sorted order statistics [49] and [1949], approximately [−0.01433995, +0.00959287]. It is descriptive, not multiplicity-adjusted or evidence of significance.

[Results](RESULTS.md) • [artifact manifest](results/20261115-baseline/artifact-manifest.json).
