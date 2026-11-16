# Ceous — independent evaluation and reproducibility handoff

**Status: draft reviewer guide (1 November 2026), not a final experiment release.** This file documents the evidence actually inspected, distinguishes historical pilots from the newer teacher-feedback experiments, and identifies the remaining requirements for a TA to rerun the implementation. It does **not** alter the submitted midpoint PDF, claim a new result, or authorize access to private workspaces. The final report and recorded presentation are due **15 November 2026**.

## 1. What is and is not committed

- The `main` branch at `619aeedfcf49efd261f607c259f7cbcd32bb1c56` (12 October 2026) is chiefly a research plan, surveys, standalone explainers and pilot write-ups. It has **no executable C5/C6 training or evaluation release, dependency lockfile, or end-to-end reproducer**. A TA cannot reproduce the current results from `main` alone.
- The separate branch `experiment/100-direction-screen-20260928` contains a **historical lightweight feature/logistic-scoring screen**, not the current full bi-encoder experiments. It reported a best direction only **+0.0009** mean nDCG@10 against its top-candidate control, with a slightly negative worst-dataset difference. Do not describe that screen as a confirmed teacher-distillation gain.
- The branch `experiment/dense-feedback-validation-20260929` contains a different **query-side transformer experiment script, requirements and workflow**, but lacks its expected `experiments/dense-feedback-validation/final-results/` artifacts in the inspected branch tree. Its instructions are *not* the executed C5/C6 protocol and are not evidence that its proposed validation completed.

## 2. Keep the September pilots distinct

| Artifact | Relevant difference | Interpretation |
| --- | --- | --- |
| `survey/experiments/02-ann-backend-negative-selection-sanity.md` | 50k MS MARCO; ANCE; 1,000 train / 200 calibration / 500 eval queries; HNSW, IVF-Flat and exact; one seed; 300 optimizer updates; **no cross-encoder teacher** | Candidate top-200 HNSW–IVF Jaccard ≈0.917, but their selected-20 **intersection/20** was 0.4344 (not selected-pool Jaccard). All three trained models lost to untouched ANCE. |
| `experiments/ann-micro-pilot-20260907/RESULTS.md` | Separate 50k ANCE follow-up, **MiniLM cross-encoder** teacher, 32,312 unique scored query–passage pairs, 100 updates | Untouched ANCE MRR@10 0.8908; HNSW+teacher 0.8667; IVF+teacher 0.8559. Both trained models lost. |

Neither pilot is the current multi-domain C6 study. ANN `Recall@200` in these pilots measures agreement with *exact nearest neighbors*, not qrel-relevance recall. A large identity change under a rank-position selector does not establish semantic diversity or a general ANN effect. Never substitute the 0.917 top-200 Jaccard for the 0.4344 selected-20 shared fraction.

## 3. Current local research: C5 and C6 (read-only evidence snapshot)

The experiments below were inspected in the authorized Ceous research workspace on 1 November 2026. Their frozen run records date to **31 October 2026**. These sources are **not in `main`**, and no newer results are implied. An independent evaluator requires a separate authorized, hash-verified evidence bundle; do not publish private corpus/model caches or raw records indiscriminately.

**C5 — SciFact full bi-encoder.** Frozen `ire-fullencoder-c5-20261031/shared/PROTOCOL_C5.md`: 505 TRAIN and 200 HOLDOUT queries (HOLDOUT was already consulted in C2; C5 is the declared final second look), 5,183-document corpus, shared `all-MiniLM-L6-v2` query/document encoder, full corpus re-encoding after training, three seeds **17/29/41**. G-full trains with qrel positives (InfoNCE); KD-full adds cached MedCPT teacher KL on non-positive slate documents, with teacher temperature **4.0**. Three seed-specific CSVs and `results-d4-combined/g5.json` support the final calculation. `G5` compares KD-full versus G-full on nDCG@10, requiring mean improvement at least 0.010, paired 95% CI lower >0 and positive improvement in every seed. The saved gate **failed**. Keep the four declared runtime deviations D1–D4 (Ada switch, GPU memory/batch adjustment, disabled document truncation fallback, and split seed jobs) in the evidence release. Do **not** run the original combiner in its completed destination; it creates outputs, rather than merely inspecting them.

**C6 — FiQA and NFCorpus full bi-encoder, label-scarcity design.** Frozen `ire-multidomain-c6-20261031/shared/PROTOCOL_C6.md`, clarification amendment A1 and deviations D-C6-1/D-C6-2:

| Field | Actual frozen procedure |
| --- | --- |
| Data | BEIR official FiQA-2018, corpus 57,638 docs, official test **648** queries; NFCorpus, corpus 3,633 docs, official test **323** queries |
| Selection | Per domain, SHA256-ordered **2,000** positive-labeled training-pool queries; **200** from the pool for validation; remaining **1,800** for model training; **180** SHA256-selected training queries in the 10%-gold condition. Query-ID intersections train/validation/test were verified empty; text-level duplicate and pretraining-exposure audits remain outstanding. |
| Student | Shared `sentence-transformers/all-MiniLM-L6-v2` encoder, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, both towers updated, exact initial top-50 slates, cosine/0.05, re-encode corpus; 3 epochs, AdamW LR 2e-5, WD 0.01, effective batch 16, query length 64 and document length 256. |
| Teachers | NFCorpus: `MedCPT-Cross-Encoder` revision `71caf65d4927987813984f54c284405a13fcca49`, frozen tau=4; FiQA: `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, frozen tau=1. Both passed a pre-training teacher-versus-student **validation** qualification gate. Teacher scores were used only for training/validation slates, not official test qrels. |
| Supervision arms | **A0** untouched; **G100** full qrel-gold InfoNCE; **K100** G100 + teacher KL on non-positive slate documents; **G10** 10% gold-only, sampled cyclically over G100-matched optimizer steps; **K10** the same gold batches **plus teacher KL on all 1,800 train queries**. Gold means provided relevance labels/qrels; teacher scores are **model-generated**, not gold annotations. |
| Feedback accounting | Teacher scoring **100,000 pairs/domain** = 2,000 pool queries × 50 candidates, **200,000 unique scored pairs total**. This is a *fixed teacher scoring set*, not an experiment comparing 4/12/32 teacher-pair budgets. The 10% restriction concerns gold query labels; K10 additionally sees teacher-scored queries unavailable to G10. |
| Evaluation | Official untouched test sets; full-corpus exact retrieval; graded NFCorpus relevance preserved in nDCG@10 using `2^grade−1`; also MRR@10 and Recall@100. Seeds **17,29,41**, paired per-query, seed-averaged **10,000-draw bootstrap** (seed `20261031`), pooled across **971 test queries**. The recorded `RUN_STATE.json` says test qrels were read at final evaluation only; direct independently observed filesystem-audit proof of every runtime read is not available. |
| Execution | `code/c6_pipeline_fix2.py` is the **executed** pipeline (SHA256 `d0e44520a138eaff44870619b65150e19ca71ddbce67437d8f615d00b1d53bd1`), not the original `code/c6_pipeline.py` (SHA256 `04fd73ec05dfc61dc7d97d83fff6f57ff2725461883944471d902a1637b1326c`). Teacher sharding and split GPU/identity training are documented deviations, not silent protocol changes. |

C6 passed its preregistered K10−G10 label-scarcity contrast but **failed** the K100−G100 full-gold teacher-added-value contrast. **Causal limitation:** K10 sees teacher scores for many more distinct train queries than G10 and G10 performed below the untouched A0 control; this is not a cost-matched teacher feedback-selection comparison. K10 also fell short of full-gold G100 in the recorded pooled post-hoc comparison. Additional evaluation should be explicitly designated new work, rather than retroactively modifying these frozen gates.

## 4. Integrity verification without retraining

The companion [read-only verifier](tools/verify_ceous_evidence.py) uses the Python standard library. It checks C5 CSV source hashes, matching 200-query holdouts and recomputed seed-averaged C5 mean gains; C6 frozen gate SHA, stated protocol hash, recorded qualification/tau, all 26 model-evaluation files, matching per-query IDs, recomputed per-seed and pooled **mean** differences, consistency of stored gate decisions, and (when supplied) hash-ordered C6 split manifests. It does **not** rerun teacher scoring, model fitting, test retrieval or confidence-interval resampling.

```bash
# From the repository checkout:
python3 tools/verify_ceous_evidence.py --self-test

# Only with an authorized, already-staged read-only copy of the local evidence:
export CEOUS_SCRATCH_ROOT=/path/to/authorized/ire-project
python3 tools/verify_ceous_evidence.py \
  --c5-run "$CEOUS_SCRATCH_ROOT/ire-fullencoder-c5-20261031/shared/run-002-ada" \
  --c6-run "$CEOUS_SCRATCH_ROOT/ire-multidomain-c6-20261031/shared/run-001"
```

Expected frozen C5: 200 aligned holdout queries, C5 KD-full−G-full mean **+0.0052551998**, G-full−A0 **+0.0759844876** (and G5 FAIL). Expected C6: FiQA 648/NFCorpus 323 aligned official test queries; pooled K100−G100 **+0.0034300250** (FAIL), K10−G10 **+0.0666169538** (PASS), K10−G100 **−0.0160343686**. These are historical 31 October evidence means, **not** guarantees on other environment reruns. Stored CIs were separately checked during research but are *not* independently reconstructed by the lightweight verifier. Frozen C6 `gates.json` SHA256 is `ec7ea2a6450add488615f907b28fd06d97171c46b9112c7739ce8298c03295a4`.

## 5. Remaining evaluator blockers (do not silently paper over)

1. **Code availability:** Commit a reviewed exact executed C5/C6 source snapshot (including C5 D4-specific scripts, C6 fix2 and helpers), protocols, amendments and deviations, or provide the TA a permissioned, immutable source bundle. There is presently no public end-to-end C5/C6 implementation to review.
2. **Environment:** Supply pinned dependencies, CUDA/PyTorch and driver compatibility, correct Python interpreter versions, exact model snapshot hashes and executable one-shot setup. The C6 captured runtime was Python 3.12.15, torch 2.14.0+cu126, transformers 4.57.6, sentence-transformers 5.7.0, NumPy 2.5.4 on GTX 1080 Ti. These are *observed*, not a tested reinstall lockfile.
3. **Frozen evidence manifest:** Release SHA256/size/inventory for actual executed code, dataset archives, split manifests, teacher logits and hyperparameters, 24 trained model outputs, 26 test-result JSONs, C5 seed CSVs, runtime logs and the frozen gates. The early C6 transfer index only covers four setup artifacts, not the final results. No shared scratch or original result should be silently overwritten.
4. **Execution commands and permissions:** Show `--help`, a tiny self-test, a single-seed deterministic smoke run with an *independent output directory*, all required inputs, and the exact full-run Slurm commands. Do not rerun research in-place or use the evaluation test qrels to tune parameters. Reproducing the full 200k teacher-pair experiment requires appropriate authorized GPUs and model/data access.
5. **Test coverage:** `test_c6.py` imports the original `c6_pipeline` module, not the executed `c6_pipeline_fix2`; verify the final runtime changes directly and add checks for split no-overlap, graded relevance, matched G10/K10 gold batches, teacher-score accounting, no test-qrel training reads, artifact integrity and rerun idempotence. Existing tests are useful unit checks, *not* proof of a complete fresh reproduction.
6. **Claim support:** A stronger validation-selected/early-stopped G10 control, strict query-exposure and teacher-cost accounting, independent unused test evaluation, a broader domain design and prior-art comparison are still scientific gaps. Report them as limitations; do not create post-hoc "confirmatory" gates.
7. **Final course deliverables:** Final report, recorded presentation and a TA code walkthrough are separate from the already-submitted midpoint PDF. Prepare the final outputs by **15 November 2026** without modifying the midpoint receipt or submission.

## 6. TA implementation walkthrough

Explain each line of the actual chain before discussing performance: **(1)** read official training qrels, filter positive train queries, deterministically freeze train/validation/gold10 membership; **(2)** compute untouched full-corpus embeddings and exact top-50 training slates; **(3)** obtain/freeze teacher scores on those slates and select temperature on *validation*; **(4)** apply gold InfoNCE and, for K arms, KL over non-positive slate candidates; **(5)** train the same shared query/document encoder with matched gold batches; **(6)** re-encode the *entire* document corpus with each trained model; **(7)** evaluate exact retrieval on the untouched official test once, preserving graded qrels; **(8)** aggregate per-query seed differences, bootstrap them, apply the predeclared gates and disclose execution deviations. Distinguish qrel-gold supervision, cross-encoder teacher logits and ANN-neighbor agreement; these are three different quantities.

**Review boundary:** This guide and verifier are proposed documentation/tests only. They do not modify the midpoint submission, execute research training, sync private data, or assert that the final code evaluation has been completed.
