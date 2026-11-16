# Ceous — IRE evaluation and experimental framework

**Version:** 1.1 · 16 November 2026 (IST) — revised minimal-training-first research narrative  
**Purpose:** A cohesive, evidence-backed research evaluation and oral-defense checklist for M26CS4.406.  
**Scope:** Documentation and synthesis only. This document does not launch an experiment, claim a new result, modify the canonical research-status tracker, or transfer files to a local Codex workspace.  
**Task reference:** ire-eval-defense-20261116-0521  
**Important:** This is a proposed evaluation-readiness framework, **not** an official point-weighted rubric.

## 1. The research goal, stated precisely

> **Can limited, expensive relevance feedback improve a frozen retriever's top-K shortlist at inference time beyond strong, resource-matched reranking or feedback baselines? If not, can we diagnose whether the limiting factor is original-pool coverage, in-pool ranking, teacher quality, or compute/feedback cost?**

We are not trying to prove that every experiment beats the baseline. We are trying to produce a **controlled answer to a well-posed question**, including negative results.

### The one-minute project narrative — minimal training first

1. **The retrieval problem:** A cheap first-stage retriever selects candidate documents; a stronger, costly reranker can inspect only a limited shortlist. Useful documents below that cutoff are unavailable to the expensive downstream stage.
2. **Why revisit training?** The Microsoft Research challenge raises the question of how to use costly reward-model feedback for retriever post-training. Retraining involves collecting/scoring training pairs and updating the retriever. But expensive feedback is also available at query time, so full model-weight updates need not be the first option explored.
3. **Literature-driven refinement:** After reviewing inference-time feedback and retrieval-coverage literature (including TOUR, ReFIT, related reranker-feedback work and DISCo), we narrowed the research question: **Can a fixed retriever use a small amount of teacher feedback per query to produce a better top-K shortlist within an accountable inference budget?** Existing related work means no blanket novelty claim.
4. **Minimal-additional-training setup:** Keep encoder and index weights frozen; retrieve an initial pool, score a selected subset with a teacher, adapt only query-specific state and/or the candidate-selection policy, and evaluate the final shortlist. Per-query fitting is still optimization, **not model-weight retraining** and not “zero computation.”
5. **Measured answer so far:** In the audited exploratory SciFact screen, a TOUR-style transfer gained +0.003264 DEV nDCG@10 over direct rerank20 but used extra retrieval work and fell below the team's +0.01 practical screen. H2, the tested local-covariance mechanism, lost to an exact matched TOUR control.
6. **Research deliverable:** Fair quality–cost comparisons, an audited and reproducible method screen, query-level error decomposition, and defensible limits. H3–H6 are hypotheses requiring separate reviews and experiments.

**Conceptual contrast (do not conflate the two paths):**

```text
Conventional retriever post-training:
training queries/pairs -> costly reward feedback -> update retriever weights -> use updated retriever

Current Ceous research:
query -> frozen retriever / initial pool -> limited teacher feedback
      -> per-query adaptation / candidate selection -> top-K output
      -> independently judged evaluation and full cost accounting
```

**Story boundary:** Preliminary ANN/ANCE trials were exploratory domain reconnaissance. They are deliberately **not used to motivate or evaluate the present inference-time research question**. Their historical logs remain in the repository; there is no claim that retriever post-training is generally ineffective.

**Do not claim:** new algorithmic novelty; a generalizable win; a blind held-out result; end-to-end speed improvement; or a solution to Microsoft's post-training problem from a frozen-weight inference study.

## 2. What the course evidence actually establishes

The evidence below comes from the uploaded IRE proposal form, 2026 class recordings in the user's IRE Otter channel, and official Moodle announcements. The recordings are automatically transcribed, so wording should not be treated as verbatim proof of a named speaker's identity.

| Source | Supported teaching/evaluation expectation | Interpretation for Ceous |
|---|---|---|
| Initial IRE project proposal form, August 2026 | Science **or** engineering complexity is valid; projects need importance, literature gap, executable plan, mentor scoping; evaluation is individual | Defend why the question matters, what alternatives exist, and individual ownership |
| IRE recording, **24 Aug**, “Document Similarity Analysis” | Instructor intended to attend at least one evaluation per student to calibrate differences | Each team member must understand the work |
| IRE recording, **3 Sep**, “Note” (project scoping and benchmarking) | Scope carefully; understand agent-generated code; document what worked and failed; benchmark hardware, throughput, latency and component costs, and track important changes | Maintain an experiment ledger and a reproducible benchmark sheet |
| IRE recording, **10 Sep**, “Note” (assignment grading discussion) | Code is an artifact, while explanation/understanding is central to assignment evaluation | Useful evidence for oral-defense preparation, **not proof of an identical project marking scheme** |
| IRE recording, **15 Oct**, “System Evaluation and Experimental Design” | Explicitly calls for a well-formed **hypothesis**, appropriate **data/ground-truth measurement**, and a **controlled experiment** that isolates a factor | This is the backbone of the Ceous checklist |
| IRE recordings, **12 Nov**, A/B-testing and statistical testing | Isolate intervention; discuss variance, effect size, sample size and power; paired offline ranker comparisons; avoid p-value and non-significance misconceptions | Report per-query paired differences and uncertainty, not just one aggregate number |
| TA Moodle announcement, **14 Nov** | Final report maximum **2 pages**, standard readable fonts | Check the *actual submitted file*, not only a working draft |
| Moodle confirmations, **15 Nov**, plus Cal.com | A named final-submission PDF was received; team evaluation booking confirmed for **Mon 16 Nov, 5:40–6:00 PM IST, F5 MTNLP** | Prepare a concise defense; do not assume another submission is necessary |

**What remains unknown:** No official numerical project rubric or promise that negative results receive a particular mark was found in these sources. The checklist below is a risk-reduction strategy based on explicit course guidance and standard empirical research practice.

Anil Nelakanti's public IIIT profile and institutional biography place his work in algorithm design, machine learning and applied search/retrieval. This background helps explain why practical quality–latency tradeoffs are relevant, but it is **not** itself evidence of grading criteria.

## 3. Make the research falsifiable: one trunk, three questions

**Trunk question:** At a fixed total retrieval and feedback budget, does per-query teacher-informed adaptation produce a meaningful improvement over the best eligible baseline?

- **RQ-A — Where is the recoverable error?** Measure how many judged positives are outside the initial pool, inside the pool but below the shortlist, or already present but misranked.
- **RQ-B — Does feedback help beyond simply using more compute?** Compare query movement, fixed-pool reranking, deeper retrieval and plain feedback against matched budget vectors.
- **RQ-C — When does feedback fail?** Report teacher-noise sensitivity, zero-positive initial pools, query-level harms, and whether a result survives an unseen evaluation partition.

A null/control hypothesis must appear next to every proposal: **additional scans, extra candidate slots, direct reranking, or noisy qrels may explain the apparent gain without the proposed mechanism.**

### Registered branch map: never confuse proposals with results

| Branch | Question / mechanism | Current status from repo | Evidence or required gate |
|---|---|---|---|
| H1 | Known teacher-guided query movement (TOUR-style transfer) | **Completed baseline** | Screen failed to establish practical superiority |
| H2 | Uniform local-covariance movement versus exact norm-matched TOUR | **Tested configuration rejected/parked** | DEV difference negative; do not reject all geometric methods |
| H3 | Bounded kernel residual on initial U50 | **Proposed** | Compare same pool, teacher budget, kernel/linear controls; requires preregistration and review |
| H4 | Adaptive allocation of retrieval/feedback budget | **Proposed** | Fixed equal-total-cost controls; coordinate overlapping P5 work |
| H5 | Score-blind diversity selection of exactly 20 feedback pairs | **Proposed** | Compare first20, uniform, rank-stratified, diversity controls with identical adaptation |
| H6 | Multi-direction max fusion | **Proposed** | Three-scan/slot accounting; distinct from existing P5 RRF study |
| P5 | Separately directed lexical/decomposition/coverage investigation | **Previously active; live final state unverified here** | Do not import incomplete partial logs as results or duplicate ownership |

Sources: [research status](../../RESEARCH_STATUS.md), [branch designs](BRANCH_DESIGNS.md), [literature/audit boundaries](LITERATURE_AND_AUDIT.md).

## 4. One immutable experimental contract

**Unit of comparison:** individual query; both arms score the same query and the same relevance judgments.

1. **Dataset authority.** Identify corpus, qrels source, query IDs, and exact train/development/test partitions, with hashes. Document all prior exposure; previously inspected queries are **not blind holdout**.
2. **Primary metric.** Predeclare nDCG@10 for the current SciFact screen; secondary Recall@K, MRR@10, relevant-document coverage and per-query changes. Explain why each metric suits this task. Unjudged documents are not known irrelevant.
3. **Common setup.** Freeze encoder, index, initial pool, evaluation queries, teacher model, final K, and pre-processing. Per-query state optimization is allowed but counted; **model-weight retraining is not performed**.
4. **Fair budget.** Record separate totals for teacher query-document scores, ANN/full scans, scanned candidates/probes, returned slots (including duplicates), unique candidates, CPU/GPU and latency. No claim of “same cost” when only teacher-call count matches.
5. **Baselines.** At least raw dense, direct same-pool rerank20, and the strongest eligible feedback comparator (e.g. TOUR). Include rerank50 as a **higher-budget diagnostic**, not a peer unless costs are equalized.
6. **Single intervention.** When evaluating H2 against TOUR, hold everything fixed except the H2 movement mechanism; similarly isolate selection vs update rule vs number of scans. Use separate ablations for independent factors.
7. **Positive and negative controls.** Required synthetic planted-truth test; shuffled-teacher or random-selection control; unchanged-query/dense fallback. Log both passing and failing controls.
8. **Uncertainty and robustness.** Paired per-query deltas, seeded bootstrap confidence intervals, query-level wins/losses, effect magnitude and limitations; explain multiplicity and sample-size restrictions. A confidence interval overlapping zero is not proof of equivalence.
9. **Resource report.** Include hardware, model/cache residency, cold/warm path, full latency p50/p95 if measured, throughput, component timings, and **explicit exclusions**. Cached teacher scores must never be presented as free online inference.
10. **Reproducibility.** Capture source commit, exact command, configuration, seed, hashes of data/model/predictions, complete result table, run log and independent metric replay. Preserve invalid and negative runs with correction records.

**Internal decision rule already written in the repo:** For an exploratory DEV screen, seek at least **+0.01 absolute nDCG@10** versus the strongest eligible comparator with no greater teacher-pair, full-scan and returned-slot budgets. This is a **team research screen**, not an official IRE grading threshold. A candidate failing it is **PARK/PIVOT**, not secretly tuned on INNER.

## 5. The evidence we can defend today

### Audited exploratory MedCPT SciFact study

| Arm | DEV nDCG@10 (405) | INNER nDCG@10 (100) | Teacher pairs | Cost interpretation |
|---|---:|---:|---:|---|
| Dense retrieval | 0.650374 | 0.663048 | 0 | One scan; starting point |
| Direct rerank20 | 0.744338 | 0.753822 | 20 | One scan; essential comparator |
| TOUR-style query movement | 0.747601 | 0.760260 | 20 | Two scans; not equal total cost to rerank20 |
| H2 uniform-covariance | 0.745127 | 0.757902 | 20 | Two scans; loses to matched TOUR |
| Direct rerank50 | 0.753785 | 0.791469 | 50 | Higher teacher budget, diagnostic only |

- **H1 vs rerank20 on DEV:** +0.003264 nDCG@10, below +0.01 with additional retrieval work. No practical superiority claim.
- **H2 vs exact norm-matched TOUR:** **−0.002475 DEV**, **−0.002358 INNER**. Paired INNER descriptive bootstrap interval approximately **[−0.01434, +0.00959]**; no demonstrated difference at the reported uncertainty.
- **Coverage bottleneck:** Across 505 queries and 568 judged positives, 504 are in initial top50, 526 in top100, and 42 beyond top100; 57 queries have **zero** judged positives in the first 50. This supports separate analysis of *ranking*, *pool coverage*, and *unrecoverable-within-pool* errors, subject to incomplete judgments.
- **Timing limit:** warm CPU p50 TOUR 1.673 ms vs H2 5.433 ms, excluding encoder and teacher inference. They are **not online end-to-end latencies**.
- **Leakage limit:** the 405/100 SciFact partition was previously exposed; the study is exploratory. No fresh teacher/model inference was used in this cache-only run.

References: [RESULTS.md](../../experiments/inference-time-ceous-20261115/RESULTS.md), [AUDIT_RESULTS.md](../../experiments/inference-time-ceous-20261115/AUDIT_RESULTS.md), [PROTOCOL.md](../../experiments/inference-time-ceous-20261115/PROTOCOL.md).

### Scientifically responsible conclusion

> “On the tested, previously exposed SciFact queries, query adaptation produced only a small DEV advantage over direct rerank20 and did not meet our cost-aware screen; the tested geometric variant underperformed a matched existing method. However, the controlled analysis identifies distinct bottlenecks and defines where more targeted feedback selection or retrieval-budget allocation could be useful. Generalization remains unverified.”

This is a conclusion supported by current evidence. “We beat retrieval SOTA” is not.

## 6. Checklist to be ready for Anil's evaluation

Legend: **[x]** evidence verified in linked records; **[ ]** action or verification still needed. “Prepared” is not the same as “submitted.”

### P0 — Before today's oral evaluation

- [x] IRE final **submission receipt** exists from 15 Nov 2026, with a named project PDF. **Receipt only**; actual file content was not independently opened here.
- [x] Cal.com confirmation lists **Monday 16 Nov, 5:40–6:00 PM IST, F5 MTNLP**.
- [ ] Open the **actual submitted PDF**; confirm maximum two readable pages; verify its numbers/claims against the audited repo and avoid contradictions with the working report.
- [ ] Prepare **one 45-second story**: two-stage retrieval and expensive feedback → literature-driven minimal-training question → controlled result → limitation.
- [ ] Prepare **one honest result table** (Section 5) with a cost column and split labels; explain rerank50's additional teacher calls.
- [ ] Prepare **one failure-mode figure**: initial F20 → U50 → outside-U50/100, label judged vs unjudged relevance.
- [ ] Be ready to show the existing replay script and explain how qrels are kept out of per-query adaptation.
- [ ] Verify whether P5 has a later independently audited final record; if not, say **pending/unverified**, not “completed.”
- [ ] Agree with teammates on accurate **individual contributions** and who can answer what; never invent authorship.
- [ ] Have a 2-minute and a 7-minute version; leave time for examiner questions.

### P1 — Scientific validity

- [x] H1/H2 protocol, cached results and independent raw-metrics audit documented.
- [x] Original MSR motivation and a literature-driven **inference-time scope refinement** documented; no failed-pilot causal narrative.
- [x] Existing H2 negative result and its decision rule documented.
- [ ] Clearly mark every plot/table as **historical**, **exploratory**, **audited**, or **proposed**.
- [ ] Add a per-query gain/loss example if time permits, but do not manufacture qualitative explanations without inspecting examples.
- [ ] State prior SciFact exposure and absence of fresh blind confirmation.
- [ ] Distinguish teacher preference from independent qrel relevance.
- [ ] Distinguish warm cached CPU timing from deployable end-to-end latency.

### P2 — If continuing research after evaluation

- [ ] Choose **one** next question based on measured failure mode, not novelty brainstorming.
- [ ] Read live RESEARCH_STATUS and owner/reservation ledger; avoid duplicating ongoing P5 work.
- [ ] Register counterhypothesis, budget cap, same-cost baselines, seed, data split, minimum effect, stop criterion and reviewer **before** execution.
- [ ] Use genuinely fresh unexposed queries for confirmatory claims if feasible; don't rename previously exposed partitions as holdout.
- [ ] Retain all negative results and record CONTINUE / PIVOT / PARK / STOP with evidence.
- [ ] Recompute the report table from archived predictions after each accepted run; never silently overwrite previous outcomes.

## 7. Suggested 12–15 minute oral defense

| Time | Show | Examiner question it answers |
|---|---|---|
| 0–1 min | One diagram and research question | What exactly is the problem? |
| 1–3 min | Minimal-training motivation and literature-driven question refinement | Why this scope, and how is it related to literature? |
| 3–5 min | Pipeline and controllable interventions | What changes in your proposed method? |
| 5–8 min | Frozen dataset, baselines, teacher/scans/slots and result table | Is the experiment fair and interpretable? |
| 8–10 min | H1 small gain, H2 failed result, bootstrap interval | What did you learn rather than merely observe? |
| 10–12 min | Coverage decomposition and known limitations | What would falsify or change your conclusions? |
| 12–15 min | Individual technical explanation, reproducibility and Q&A | Do you understand the project and the code? |

**Expected hard questions and prepared answers**

1. **Why not rerank 50?** It has more teacher scores; compare the gain and full cost, and recognize that it currently performs strongly.
2. **Is H1 a novel improvement?** No. It is a known-method baseline/transfer, and its +0.003264 DEV advantage is below our practical screen.
3. **Why is H2 worse?** We have empirical evidence of worse scores versus an exact matched control, but not yet a causally verified explanation; show the observed failure, uncertainty and controls.
4. **How is this related to Microsoft post-training?** Shared expensive-feedback selection motivation; inference-time adaptation is an **alternative**, not a completed post-training answer.
5. **Are these test results blind?** No. All 505 selected queries were previously exposed; claims are exploratory.
6. **How can the teacher be trusted?** Teacher scores guide adaptation; independent qrels judge retrieval quality, with incomplete-judgment caveats.
7. **Can you reproduce it?** Packaged predictions permit metric replay; complete fresh end-to-end inference still needs original cached vectors/models at hash-bound paths.
8. **What did each person do?** Use a factual ownership log and explain one's own algorithm, command, failure modes and metrics.

## 8. Ongoing experiment/report operating loop

For every proposed change:

**PRE-REGISTER → FREEZE DATA/CONTROLS/COST → RUN ONCE → AUDIT → PER-QUERY ANALYSIS → DECIDE → UPDATE REPORT.**

Each experiment note must record: hypothesis, distinct intervention, null/control, expected mechanism, dataset/split exposure, exact baselines, metrics, budget-vector, implementation version, seed/hashes, run commands, raw outputs, independent audit, effect/interval, limitations, final decision and next question. Keep rejected and invalid attempts append-only rather than rewriting history.

## 9. Sources and claim-boundaries register

**Course evidence:** Initial IRE project proposal form (uploaded PDF); IRE 2026 Otter recordings dated 24 Aug, 3 Sep, 10 Sep, 15 Oct and 12 Nov (titles above, available in the user's connected account); official Moodle announcements “Project evaluations and end-sem syllabus” (14–15 Nov 2026) and “Project Submission Details” (14 Nov 2026); Cal.com event confirmation and Moodle submission receipt. **No formal percentage rubric verified.**

**Current scientific evidence:** [RESEARCH_STATUS.md](../../RESEARCH_STATUS.md), [RESEARCH_PLAN.md](../../RESEARCH_PLAN.md), [FINAL_REPORT_WORKING.md](../../FINAL_REPORT_WORKING.md), [2026-11-15 results](../../experiments/inference-time-ceous-20261115/RESULTS.md), [audit](../../experiments/inference-time-ceous-20261115/AUDIT_RESULTS.md), [literature and scope](LITERATURE_AND_AUDIT.md).

**External:** [Microsoft research challenge](https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/), [Anil Nelakanti's IIITH faculty profile](https://www.iiit.ac.in/faculty/anil-nelakanti/), [IIITH profile of his applied search background](https://blogs.iiit.ac.in/monthly_news/dr-anil-nelakanti/).

**Version hygiene:** This checklist is a synthesis of sources available on 16 Nov 2026. The submitted PDF, live P5 state, and anyone's current verbal expectations must be checked separately; no unstated tool access, continuing monitoring, grade assurance, fresh experiment, or local-device synchronization is implied.
