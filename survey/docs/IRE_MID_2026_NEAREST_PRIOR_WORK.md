# IRE Mid-2026 Checkpoint: Nearest Prior Work and Novelty Boundaries

**Team:** Ceous · **Task:** `ire-mid-checkpoint-20261022-001` · **Literature checked:** 22 October 2026

## Question and short answer

> Holding the initial retriever, teacher, candidate depth, selection rule, per-query scoring budget, training objective, optimizer, and evaluation fixed, can the ANN backend used for candidate generation change the query–document supervision selected for retraining and the retriever's final exact-search performance?

The broad idea that retrieved candidates or negative samples shape dense-retriever training is established. ANCE makes full-corpus ANN retrieval part of hard-negative training; ADORE makes dynamic retrieval part of a ranking-optimization method; RocketQA, RocketQAv2, SimANS, and later studies demonstrate that candidate filtering, selection, or teacher signals can affect training. A claim that “ANN affects retriever training” or “this is the first use of ANN for training examples” would therefore be too broad.

The narrower, still testable question is whether **the ANN index/search backend itself** changes the selected teacher-labelled pairs and final retriever when the retriever snapshot, teacher, selector, score budget, loss, and evaluation are fixed—and when HNSW/IVF operating points are matched on exact-neighbor recall. The papers below establish close mechanisms and strong counterarguments; the reviewed sources do not establish this exact controlled comparison. This is a nearest-work map, not proof that no such comparison exists anywhere. Do not use “first” without a wider systematic search.

## Closest primary work

### Dense retrieval training, candidate selection, and teacher supervision

1. **ANCE — Xiong et al., ICLR 2021, [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval](https://openreview.net/forum?id=zeFrfgyZln).**
   - **Contribution:** Explains the weakness of uninformative in-batch negatives and trains with hard negatives retrieved globally from the corpus using an ANN index that is refreshed as the retriever changes.
   - **Overlap / boundary:** Closest conceptual prior: ANN results become training negatives and influence the learned retriever. The target here is narrower than ANCE's changing-model training loop: vary the search backend while freezing the initial retriever and the rest of the supervision pipeline.
   - **Falsifier:** At matched exact-neighbor Recall@200 and fixed candidate/teacher budgets, if backend-specific selected IDs and teacher targets show no reproducible change—or final exact-search metrics fall within paired-seed variation—the proposed backend-specific effect is unsupported.

2. **RocketQA — Qu et al., NAACL 2021, [An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://aclanthology.org/2021.naacl-main.466/).**
   - **Contribution:** Combines cross-batch negatives, cross-encoder-denoised hard negatives, and data augmentation; it is an especially close precedent for retrieved candidates followed by expensive teacher filtering/scoring.
   - **Overlap / boundary:** Establishes the candidate → cross-encoder → retriever-training pathway. It changes a package of training interventions and does not isolate index/search backend as the sole treatment.
   - **Falsifier:** Run the same fixed cross-encoder and selection rule on backend-specific candidate pools at identical per-query score counts. If the retained identities/teacher labels and trained outcomes are equivalent, this pipeline offers no evidence for a distinct backend effect.

3. **ADORE / STAR — Zhan et al., SIGIR 2021, [Optimizing Dense Retrieval Model Training with Hard Negatives](https://doi.org/10.1145/3404835.3462880).**
   - **Contribution:** Analyzes hard-negative training; STAR adds random negatives for stability, while ADORE uses dynamic query-side negatives to optimize ranking performance.
   - **Overlap / boundary:** Strong novelty counterargument: dynamically retrieved negatives already link retrieval behavior to final ranking. ADORE changes the training/sampling method and adapts the query encoder; the proposed test keeps the candidate-generating checkpoint fixed and compares index backends under one unchanged objective.
   - **Falsifier:** Freeze the mining checkpoint and selector, match backend recall and budget, and pair training seeds. If backend effects disappear under this control, there is no separable index-backend effect beyond the known dynamic-mining/training interaction.

4. **RocketQAv2 — Ren et al., EMNLP 2021, [A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://aclanthology.org/2021.emnlp-main.224/).**
   - **Contribution:** Jointly trains a dense retriever and cross-encoder reranker using dynamic listwise distillation and hybrid data augmentation.
   - **Overlap / boundary:** Shows that candidate lists and teacher relevance distributions can co-evolve and affect both modules. Its teacher is jointly adapting, unlike the fixed-teacher comparison proposed here.
   - **Falsifier:** Keep a single teacher frozen and fix its scoring budget; if HNSW/IVF/exact candidates produce the same selected targets and final retrieval within paired-seed variability, the proposed fixed-teacher effect is falsified for this setting.

5. **SimANS — Zhou et al., EMNLP Industry Track 2022, [Simple Ambiguous Negatives Sampling for Dense Text Retrieval](https://aclanthology.org/2022.emnlp-industry.56/).**
   - **Contribution:** Samples negatives near positives using a probability distribution designed to favor informative, less obviously false negatives over too-easy or too-hard samples.
   - **Overlap / boundary:** Directly establishes that the *selection rule* over candidates matters. It does not isolate the ANN family when the candidate selector and scoring budget are held fixed.
   - **Falsifier:** Apply the same frozen SimANS-style (or project-fixed) selector to recall-matched backend pools; if the selected supervision and exact-search outcomes do not differ reproducibly, backend identity adds no demonstrated effect beyond selection policy.

6. **ANCE-Tele — Sun et al., EMNLP 2022, [Reduce Catastrophic Forgetting of Dense Retrieval Training with Teleportation Negatives](https://aclanthology.org/2022.emnlp-main.445/).**
   - **Contribution:** Addresses instability in iterative hard-negative training with momentum negatives from earlier steps and lookahead negatives that approximate future iterations.
   - **Overlap / boundary:** Shows that the timing and history of mined negatives can affect convergence and forgetting. It studies temporal sampling dynamics, not a controlled HNSW-versus-IVF choice with the teacher and objective fixed.
   - **Falsifier:** Replay identical cached candidate IDs and teacher scores through nominal backend branches with the same seed. Any model difference in this replay is an implementation/training confound; without a clean replay, a backend-causality claim fails.

7. **PROD — Lin et al., The Web Conference (WWW) 2023, [Progressive Distillation for Dense Retrieval](https://doi.org/10.1145/3543507.3583421).**
   - **Contribution:** Progressively improves teachers and training data; its data-progressive stage selects queries the student finds confusing and distills from stronger teachers, with regularization against forgetting.
   - **Overlap / boundary:** Closely connects selected candidate data and teacher quality to final retrieval. It changes teacher strength and data curriculum, so it does not identify an index-backend effect under a fixed teacher and selection objective.
   - **Falsifier:** Hold teacher, confusing-query set, selected-pair budget, and student objective constant while changing only the backend. If paired runs show no stable target or exact-retrieval differences, the backend-specific hypothesis is not supported.

8. **Rajapakse, Yates & de Rijke — SIGIR 2024, [Negative Sampling Techniques for Dense Passage Retrieval in a Multilingual Setting](https://doi.org/10.1145/3626772.3657854).**
   - **Contribution:** Reproduces and compares BM25, teacher-based clustering, periodically updated dense-index negatives, and iterative clustering across multilingual in-/out-of-distribution settings; proposes ICT and reports that sampler effectiveness depends on setting.
   - **Overlap / boundary:** A strong empirical counterargument to novelty claims about sampling: training negatives can change final retrieval, and comparisons across samplers are established. The study varies sampling methods and generalization conditions, not only the ANN backend under one shared fixed supervision procedure.
   - **Falsifier:** Repeat a fixed English setup with exact/HNSW/IVF candidate generators, matched on calibration recall and score count. If selected labels and paired-seed downstream metrics agree, no additional backend effect is shown beyond known sampler effects.

### Search/index work and the strongest deployment counterpoint

9. **Big ANN — Simhadri et al., NeurIPS 2025 Datasets and Benchmarks Track, [Results of the Big ANN: NeurIPS’23 competition](https://papers.neurips.cc/paper_files/paper/2025/hash/63092d79154adebd7305dfd498cbff70-Abstract-Datasets_and_Benchmarks_Track.html).**
   - **Contribution:** Benchmarks ANN indexing/search systems on practical filtered, out-of-distribution, sparse, and streaming workloads, measuring search accuracy and efficiency.
   - **Overlap / boundary:** Establishes that ANN systems have different accuracy/efficiency operating points. It evaluates search systems, not teacher-labelled training pools or retrained students.
   - **Falsifier:** If backend differences at the chosen operating points do not change selected IDs/teacher targets, or any downstream difference vanishes across paired seeds, the benchmark's retrieval trade-offs do not translate into a training effect here.
   - **Date correction:** The competition was held at NeurIPS 2023; the proceedings paper is in **NeurIPS 2025**, not NeurIPS 2023 and not 2026.

10. **Barman et al., ICLR 2026, [Welfarist Formulations for Diverse Similarity Search](https://proceedings.iclr.cc/paper_files/paper/2026/hash/abaab8d9908c3df048fbfc0802dc778e-Abstract-Conference.html).**
    - **Contribution:** Defines welfare-based nearest-neighbor objectives that adaptively balance relevance and attribute diversity, with algorithms that can use standard ANN methods as a subroutine.
    - **Overlap / boundary:** Relevant if the project later varies which candidates are selected, but it changes the retrieval objective toward diversity. It does not test fixed-objective teacher-supervised retriever training.
    - **Falsifier:** For the present claim, keep the project objective fixed; if backend-specific candidates selected under that same objective yield equivalent labels and final models, diversity-aware search does not rescue a backend effect. Treat diversity selection as a separate follow-on question.

11. **Elkiran & Rasheed, IEEE Access 2026, [An Empirical Evaluation of Retrieval, Reranking, and Similarity for a Q&A-Based RAG System](https://ieeexplore.ieee.org/abstract/document/11396639), DOI [10.1109/ACCESS.2026.3664852](https://doi.org/10.1109/ACCESS.2026.3664852).**
    - **Contribution:** Varies index structures, retrievers, rerankers, and similarity measures in a modular RAG pipeline; its abstract reports index choice has negligible impact across most measured metrics, while retriever and similarity choices dominate.
    - **Overlap / boundary:** The strongest direct counterpoint to practical impact: index choice may matter little to end-to-end retrieval/RAG outcomes. It evaluates an inference-time RAG configuration, not candidate generation used to construct fixed-budget training supervision; it therefore neither proves nor disproves the training-mediated effect.
    - **Falsifier:** If a controlled teacher-scoring/retraining study also finds no reliable supervision or exact-search outcome differences, this counterpoint aligns with a null result and the proposed practical claim should be dropped. The IEEE Xplore record verifies publication on 16 February 2026 (vol. 14, pp. 26053–26066).

## Strongest novelty counterarguments

- **The mechanism is old:** ANCE uses ANN retrieval to choose negatives for training. The project cannot claim novelty from that pipeline alone.
- **Sampling already has a large literature:** ADORE, RocketQA, SimANS, ANCE-Tele, PROD, and the SIGIR 2024 study vary how candidates/negatives are generated, selected, denoised, or distilled, and report training consequences.
- **Teacher-scored candidates are not new:** RocketQA and RocketQAv2 are close structural precedents. The defendable distinction is holding the teacher fixed and attributing any effect to the search backend.
- **ANN quality alone does not imply training value:** Big ANN measures search quality/efficiency. The IEEE Access 2026 study reports mostly negligible index impact on its RAG metrics. Retrieval recall or latency cannot stand in for selected-supervision quality or post-training gains.
- **The current pilot does not demonstrate improvement:** It has one seed, and all retrained branches are below the untouched checkpoint. Do not claim backend superiority, useful post-training, statistical significance, or semantic difference between candidate pools.

## What the two project pilots support

The project records describe **two different experiments**, not two seeds of one experiment:

- The [ANN-backend negative-selection sanity record](../experiments/02-ann-backend-negative-selection-sanity.md) uses no cross-encoder teacher, rank-position selection, and 300 updates. It reports HNSW/IVF top-200 exact-neighbor agreement of 0.956/0.962, candidate-set Jaccard 0.9174, and only 43.44% shared fraction among selected-20 entries. Its trained MRR@10 values (exact 0.8261, HNSW 0.8333, IVF 0.8505) are all below the untouched checkpoint's 0.8908.
- The [teacher-scored micro-pilot](../../experiments/ann-micro-pilot-20260907/RESULTS.md) uses a cross-encoder, soft-label cross-entropy, and 100 updates. It also reports 43.44% selected-pool shared fraction, but the teacher's top passage matches for 993/1,000 queries and the mean aligned-distribution JSD is 0.00523. Its MRR@10 values (HNSW 0.8667, IVF 0.8559) are again below the same untouched 0.8908 baseline; the two trained branches reverse order relative to the no-teacher run.

Safe interpretation: candidate identities differ substantially under the tested fixed selector, but the teacher's aggregate/aligned outputs are close in this run, and neither retraining branch beat the starting model. These are motivation and feasibility observations, not a benefit claim. Keep the two configurations and metric tables separate in the checkpoint; explain the different teacher/loss/update settings rather than pooling them.

## Minimal falsifiable test matrix

Use the existing teacher-scored setup as the target. Freeze one initial retriever snapshot for *all candidate generation*, one teacher checkpoint, corpus, query split, positive filtering, selector, per-query scoring count, soft-label loss, optimizer/schedule, and exact evaluation. Select HNSW and IVF settings on calibration queries to match exact-neighbor Recall@200 as closely as practical; report recall and latency separately. Do not tune settings against evaluation relevance labels.

| Arm / check | Candidate and training condition | Minimum replication | What it answers |
|---|---|---:|---|
| Untouched baseline | Initial checkpoint, no updates; same exact evaluation | 1 | Whether any retraining beats the starting model |
| Exact | Exact top-200 candidates → same fixed selector and teacher budget → same soft-label objective | 3 paired seeds | Reference candidate generator |
| HNSW | Same pipeline and candidate depth; calibration setting matched to exact-neighbor recall | 3 paired seeds | Backend-specific supervision and outcome |
| IVF-Flat | Same pipeline and candidate depth; calibration setting matched to exact-neighbor recall | 3 paired seeds | Backend-specific supervision and outcome |
| Shared-supervision replay | Feed identical saved pair IDs and teacher scores through two nominal branches with the same seed | 1 pair | Implementation control: outputs should agree within deterministic tolerance; otherwise backend attribution is confounded |

Report per query and per seed: exact-neighbor Recall@200 (not relevance recall), candidate and selected-ID overlap, teacher-score/soft-label differences aligned by query–passage ID, exact-search MRR@10 as the primary retrieval endpoint, and Recall@100/1000 as secondary endpoints. Three paired seeds are a minimum diagnostic, not by themselves a strong significance claim; show each seed and the paired spread. A null result on labels and final metrics is a valid answer.

## Concrete amendments for the one-page checkpoint

- **Question:** “With the starting retriever snapshot, teacher, fixed 20-passage/query selection rule, teacher-score budget, soft-label objective, optimizer, and exact evaluation held fixed, does replacing exact/HNSW/IVF candidate generation change selected teacher supervision and the final retriever?”
- **Literature position:** “Prior work establishes that mined negatives, candidate filtering, and teacher-guided selection can affect dense-retriever training (ANCE; RocketQA; ADORE; SimANS; PROD). Our narrower test isolates the ANN search backend under a fixed supervision pipeline; we make no ‘first’ claim.”
- **Pilot evidence:** State the two experiments separately. In the teacher-scored pilot, HNSW/IVF selected-pool shared fraction was 43.44%, teacher top passage matched on 993/1,000 queries, and both retrained MRR@10 scores were below untouched ANCE (0.8908). Label this **one-seed motivation evidence**.
- **Claim boundary:** “The pilot shows candidate identity sensitivity under one selector. It does not show that the candidates are semantically different, that teacher supervision changes materially, that retraining helps, or that one backend is superior.”
- **Counterpoint:** Cite the 2026 IEEE Access RAG study as evidence that index choice can be negligible for inference-time RAG metrics; explain that its estimand differs from training-supervision selection.
- **Next test:** Name exact/HNSW/IVF, matched calibration Recall@200, fixed per-query teacher budget, at least three paired seeds, shared-supervision replay, and exact-search MRR@10. State in advance that unchanged teacher targets and outcomes would falsify the backend-effect hypothesis.
- **Bibliographic correction:** Cite the Big ANN competition as held at NeurIPS 2023 and published in the NeurIPS 2025 Datasets and Benchmarks Track.

## Source verification notes

Venue/year and contribution statements link to primary proceedings or publisher records: OpenReview/ICLR, ACL Anthology, ACM DOI/proceedings, NeurIPS proceedings, ICLR proceedings, and IEEE Xplore. The Big ANN year distinction and IEEE Access publication date were checked against the official proceedings/publisher pages. The novelty boundary is deliberately scoped to the sources reviewed; it is not an exhaustive systematic review, and “no prior work” should not be stated from this document alone.
