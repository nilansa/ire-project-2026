# Independent review record — frozen cache-only study

The execution handoff relays the final verdict of the independent reviewer `/root/baseline_review`, GPT-6.1 Sol at xhigh: “READY — approved for the bounded, cache-only exploratory evaluation against the exact snapshot below. I found no remaining scientific or implementation blocker to the proposed run.” This document records that relayed agent verdict; it is not a human signature or an independent review authored by the integrator. The handoff states that the reviewer did not implement or edit the files, index, or run real data, and inspected the code, protocol, alignment and final test artifact, with only a synthetic runtime check.

## Approved scope and bindings

The approval is for one bounded, frozen-cache exploratory SciFact study under reservation `CEOUS-BASELINE-H1H2-001`, with one shared 1,800-second budget. All partitions have prior exposure; the 405 DEV / 100 INNER division is not a blind holdout. No model is re-encoded or trained. MedCPT is primary and BGE is a separate sensitivity analysis. The 206-arm DEV grid, including pool counterparts and temperatures, is frozen; INNER uses frozen DEV choices and does not select settings. The review notes that 2000-draw descriptive bootstrap intervals are neither multiplicity- nor duplicate-group-adjusted.

| Reviewed artifact | SHA-256 |
|---|---|
| [alignment.py](alignment.py) | `3c101b8f18c61078ececbecf3198f6d98f38bacd1789c4a598ee459167613718` |
| [harness.py](harness.py) | `1afd2c5bf62052336cbed6d26ff5137bdf2f535de94b42bf8857c93761bc386a` |
| [test_harness.py](test_harness.py) | `7b37024f2b1b5b2cd1a36be5a19c9ce1a55f7ab23e40ffc50fbe6dad90c957ae` |
| [PROTOCOL.md](PROTOCOL.md) | `dfa00c8bfc250de22b5c7b9887b00b041b80b8f392e5fdcbb452744101e3fe54` |
| [FINAL_REVIEW.md](FINAL_REVIEW.md) | `01dd38dda95bea7afe40824848c5b04edcaea89921d02644f3a103351e4960ca` |
| [README.md](README.md) | `5c5a6768764c007936e1485b2c7cabd591dbdd9794ab6f96d1b525dc430009ca` |
| Alignment record | `0bf495b424fc52b7bba6fdf6ab32354f4918e7aa52f17e3a68915f0650b4453d` |
| Seven-check synthetic test log | `664522bf67525ab76f41533ed4dbf49dffd31610860a109453f6528b5a988237` |
| Train qrels, pre-run binding | `a53f2114831916c096b6c37d9e54da68cef4efdcdbd5ed46533601af972acf1d` |

The integrator verified these hashes and inspected the seven-test log. The reviewer reports correct all-qrel ideal normalization, qrel-free prediction, frozen selection before INNER, exclusive output paths and completion manifests, retention of all family winners, matched TOUR/H2 full search and complete displacement including decay, F20 / U50-minus-F20 / outside-U50 decomposition, depth-50/100 accounting, and separate logical and physical work. Four duplicate query groups do not cross the 405/100 split. The handoff reports that the reviewer accepted the dataset provenance and attribution for this study.

The handoff reports a synthetic one-thread runtime check: 200 random 50×384 SVDs averaged 2.759 ms (p95 4.063 ms); 1,000 5,183×384 matvec/top-50 operations averaged 0.446 ms (p95 0.734 ms). Its extrapolation was about 652 seconds at means, 999 seconds at p95 values and 1,304 seconds at twice the mean, excluding overhead. These are synthetic estimates, not measured end-to-end deployment or proof that the run will finish. The fixed budget is not extended if exceeded. The earlier [validation record](VALIDATION.md) reports the earlier three-check stage; the seven-check log is the later record. [FINAL_REVIEW.md](FINAL_REVIEW.md) preserves the implementer's pre-approval status; this separate record captures the later relayed verdict without changing reviewed source.
