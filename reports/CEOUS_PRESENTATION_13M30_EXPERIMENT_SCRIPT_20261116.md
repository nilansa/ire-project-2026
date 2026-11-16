# Ceous — 20-slide research presentation and speaker script

**Date:** 16 November 2026  
**Target runtime:** **13 minutes 30 seconds** (minimum acceptable duration: 10 minutes).  
**Format:** full-screen, one-slide-per-screen HTML presentation; forward/back navigation; optional speaker-notes panel; progress indicator.  
**Status:** SCRIPT / PRESENTATION SOURCE, not a claim that the final website or recording is already published.  
**Main narrative:** Microsoft Research motivation → bounded inference-time feedback → TOUR → ReFIT → TTT-Embed → three separately labeled experiments → complementary feedback hypothesis and honest conclusion.

## Presenter rules

- This is a coherent ~13:30 talk, **not** a three-minute synopsis. Slow down for diagrams and metric interpretations. Target 125–145 spoken words per minute; never race through numerical slides.
- Separate **published papers' results** from **our own results**; label independently verified, exploratory and proposed work.
- We make **no claim** of a new state-of-the-art method, reliable positive held-out improvement, faithful DISCo implementation, or reproduction of TTT-Embed global-wise results.
- Avoid historical ANN/ANCE exploration in the main story; that is not the motivation for the scoped frozen-weight experiment.
- For all experiments, distinguish additional retrieval/search cost, teacher-scoring cost and ranking-quality metrics.
- The 20 headings below are the slide titles. **Display text** is minimal; **speaker text** is read or paraphrased orally and must not crowd the slides.
- Suggested visual style: clean restrained light background, near-black ink, serif display heading and humanist sans-serif annotations; one mathematical idea or result per screen, no decorative AI-looking gradients.

### Slide 01 — Can better feedback fix a frozen retriever? (00:00–00:30; 0:30)

**Display:** "Same retriever. Same index. Smarter feedback." / Ceous · IIIT Hyderabad · IRE 2026

**Speaker script:** Our project studies a simple question: once a retriever and its index already exist, can expensive feedback from a stronger reranker help us use them better **at inference time**, without retraining the encoder? We will introduce the problem, explain three relevant inference-time approaches, present our experimental findings, and then describe the complementary-feedback hypothesis they motivate.

### Slide 02 — The two-stage bottleneck (00:30–01:05; 0:35)

**Display:** Corpus → fast retriever → top B candidates → expensive reranker → results; highlight top-K access restriction.

**Speaker script:** In most retrieval systems a fast first-stage model searches a large corpus. A much more expensive cross-encoder reranker can inspect only a small number of returned documents. This restriction creates two problems. The reranker may prefer a document that was ranked too low to reach it, or a useful document might not enter the retrieved pool at all. These are not the same kind of failure, and they call for different remedies.

### Slide 03 — Where our question started (01:05–01:40; 0:35)

**Display:** Microsoft Research challenge: data selection / loss / computational efficiency.

**Speaker script:** The Microsoft Research challenge on efficient post-training of retrieval models proposes learning from stronger reward models, including LLM-based cross-encoders. It asks which query-document pairs should receive expensive feedback, how to formulate the learning loss, and how to make repeated training iterations affordable. That challenge provided our motivation. But for the final project we narrowed the goal to a more controlled question we could test with frozen model weights.

### Slide 04 — The scoped research question (01:40–02:15; 0:35)

**Display:** "Can teacher feedback improve retrieval at test time without changing encoder or index weights?"

**Speaker script:** Our experiments do not train a new retrieval model end to end. We keep the encoder, corpus embeddings and search index fixed. We may optimize a query-specific correction or make a better choice of documents sent for scoring, but we do not update the encoder parameters. It is important to say **frozen model weights**, not "no optimization": some methods optimize a small temporary vector.

### Slide 05 — What must remain fixed? (02:15–02:50; 0:35)

**Display:** Controlled comparison: corpus, queries, initial retriever, teacher, candidate depth, evaluation; count teacher pairs and scans.

**Speaker script:** We want to isolate the effect of feedback selection or query adaptation. So every comparison needs the same dataset, query split, initial retriever, feedback teacher, and evaluation definition. A fair cost comparison must also count how many teacher query-document scores we purchase, how many corpus scans or index searches we perform, and the latency introduced. It would be misleading to compare a method requiring two searches with a baseline requiring only one and call the teacher budgets equal.

### Slide 06 — TOUR: move the query, not the model (02:50–03:30; 0:40)

**Display:** Original q → top documents → teacher preferences → corrected q′ → search again.

**Speaker script:** The first relevant family is TOUR-style inference-time query movement. We retrieve an initial candidate set, obtain relevance feedback from a stronger model, and move the query embedding in a direction that is more compatible with that feedback. Then we search again using the changed query vector. Encoder and document-vector parameters stay frozen. The hope is that moving the query changes which documents are discovered, instead of merely rearranging the original candidates.

### Slide 07 — ReFIT: use reranker feedback for query-time retrieval (03:30–04:10; 0:40)

**Display:** Reranker scores → feedback distribution → query representation update → retrieval. "No backbone update."

**Speaker script:** ReFIT is another inference-time relevance-feedback direction. A strong reranker scores a limited candidate set, and those scores supervise a change in the retrieval query representation. The optimization encourages the retriever's scores to agree more closely with the stronger model's preference distribution. This is different from simply sorting by reranker score. ReFIT tries to alter retrieval itself, so the next candidate set can be different. We must evaluate on independent relevance judgments, not just agreement with the teacher's own preferences.

### Slide 08 — TTT-Embed: a small reusable residual (04:10–04:50; 0:40)

**Display:** q′ = q + δ; frozen embeddings + teacher scores + learned correction.

**Speaker script:** TTT-Embed gives us a related but distinctive design. Rather than changing all encoder weights, it learns a small correction in embedding space from a budget of teacher-scored query-document pairs. The corrected embedding then retrieves from the same frozen document index. The crucial distinction is **scope**: a correction can be learned separately for a query, shared across a task, or learned more globally. Those sharing scopes determine which feedback examples are allowed to influence later queries.

### Slide 09 — What the KL objective means (04:50–05:35; 0:45)

**Display:** p_teacher(d | q) compared with p_retriever(d | q + δ); fit δ while keeping encoder fixed.

**Speaker script:** Here is the intuition behind the objective. First, take a small set of documents and convert the teacher's scores into a probability distribution over them. Next, take retriever similarities using the adjusted query and convert those to another distribution. A KL-based matching loss tries to bring those preferences closer together. We optimize the small residual, not the millions or billions of original model parameters. Whether this correction generalizes depends on the feedback examples and on what queries share that residual.

### Slide 10 — Three scopes, three generalization questions (05:35–06:20; 0:45)

**Display:** Query-wise / Task-wise / Global-wise; diagram of which feedback is pooled.

**Speaker script:** Query-wise adaptation has feedback tied to one test query. Task-wise adaptation learns from queries from a common task and then uses that correction on other queries in the task. Global-wise adaptation aggregates broader information. These are not three entirely unrelated algorithms; they are different decisions about sharing the correction. The key scientific question for our project is whether a residual fit to some teacher-labeled pairs can actually help queries whose labels were not seen in its optimization.

### Slide 11 — Experimental contract (06:20–06:55; 0:35)

**Display:** Four blocks: frozen retriever / teacher / train or feedback split / held-out metric.

**Speaker script:** Now to our measurements. We use different datasets and therefore report different metrics; we do not average them into one performance number. SCIDOCS tests ReFIT with Recall at 100, FiQA tests task-wise TTT-Embed with nDCG at 10, and SciFact tests a TOUR-style per-query movement baseline. For each case I will first show the setup, then the strongest relevant comparator, then the result and its limitation.

### Slide 12 — Experiment A: ReFIT on SCIDOCS (06:55–07:40; 0:45)

**Display:** Frozen Contriever; MiniLM teacher; Recall@100 = 37.8067% → 37.8717%.

**Speaker script:** On SCIDOCS, we tested a query-wise ReFIT procedure using frozen Contriever document and query representations and a MiniLM cross-encoder as the teacher. The baseline Recall at 100 was 37.8067 percent. After the adaptation it reached 37.8717 percent, a gain of only 0.065 percentage points. This is statistically inconclusive, and we did not reproduce the stronger published reference result. The honest message is that this implementation did not produce a convincing improvement.

### Slide 13 — Experiment B: TTT-Embed on FiQA (07:40–08:25; 0:45)

**Display:** 57,638 docs; 648 held-out test queries; five-fold cross-fit; nDCG@10 = 0.46722 → 0.46313.

**Speaker script:** For a separate test, we evaluated task-wise TTT-Embed on the FiQA retrieval task. The corpus contains 57,638 documents. We used Qwen3-Embedding-0.6B, Qwen3-Reranker-4B feedback scores and five-fold cross-fitting to distinguish fitting a residual from testing it on unseen queries. Although the optimization improved scores on its own feedback or training queries, held-out nDCG at 10 moved from 0.46722 to 0.46313. That is a slight decline, not evidence of robust generalization.

### Slide 14 — Why the FiQA result matters (08:25–09:10; 0:45)

**Display:** Training-fit gain ≠ held-out retrieval gain. Optional separate direct reranking reference 0.4866, clearly labeled.

**Speaker script:** This negative result is scientifically useful. A correction can learn to imitate a teacher on the very query-document pairs used to fit it and still fail when transferred to new queries. We therefore need to test on unscored queries, and we must not use training-set agreement as the success metric. A direct reranking reference reached about 0.4866 nDCG at 10 in a separate comparison, but it has a different operational role and should not be presented as cost-matched without measuring its complete feedback budget.

### Slide 15 — Experiment C: TOUR-style movement on SciFact (09:10–10:00; 0:50)

**Display:** DEV nDCG@10: dense 0.6504; direct rerank20 0.7443; TOUR 0.7476; rerank50 0.7538.

**Speaker script:** Our independently audited SciFact exploratory experiment starts from frozen dense vectors and aligned cached teacher scores. Dense retrieval alone scored 0.6504 nDCG at 10. Direct reranking of 20 candidates scored 0.7443. TOUR-style query movement achieved 0.7476, a gain of about 0.0033 over rerank20, but it required another full retrieval scan. Reranking 50 candidates instead gave 0.7538, although it used 50 teacher pairs rather than 20. These cost differences change the interpretation.

### Slide 16 — The SciFact control that did not win (10:00–10:55; 0:55)

**Display:** H2 local-covariance = 0.7451; TOUR = 0.7476; practical screen = +0.01 nDCG@10.

**Speaker script:** We also tested a local-covariance variant of query movement, called H2. It scored 0.7451 on the development queries, below the simpler, norm-matched TOUR correction at 0.7476. We therefore parked that tested configuration. The TOUR gain over rerank20 also fell below the predeclared practical threshold of 0.01 absolute nDCG at 10, and its extra retrieval work matters. One more caveat: these SciFact queries were previously exposed, and the run used cached scores, so its reported timings are not end-to-end teacher-inference latency.

### Slide 17 — What these experiments collectively tell us (10:55–11:40; 0:45)

**Display:** Three observations: small/inconclusive ReFIT gain; no FiQA held-out gain; cost-sensitive TOUR gain.

**Speaker script:** Across three datasets we see a consistent research challenge rather than a victory claim. Query-time adaptation can fit local feedback, but a better training objective value does not guarantee better retrieval on unseen questions. Small apparent gains may disappear when compared to deeper reranking or extra search work. This makes the **quality and diversity of the feedback examples** a plausible variable to isolate next. The experiments do not by themselves prove that our proposed selector will work.

### Slide 18 — The precise new hypothesis: complementary feedback (11:40–12:20; 0:40)

**Display:** Same teacher + same number of scored pairs + same residual optimization; alter only which documents are selected. Dense / BM25 / ColBERT / DISCo-style.

**Speaker script:** Here is our contribution hypothesis. If we always choose the nearest dense candidates, many feedback examples might be redundant. Could we use a complementary retriever such as BM25, a token-level retriever such as ColBERT, or a DISCo-inspired coverage selector to surface different useful feedback documents? We would keep the teacher, residual optimization and total teacher pair budget fixed. Only the document selection set changes. This is a proposed experimental direction, not a measured improvement.

### Slide 19 — How we would actually falsify it (12:20–13:05; 0:45)

**Display:** Controls: top-ranked / random / rank-stratified / MMR / deeper retrieval / complementary; measures: held-out nDCG, Recall, teacher pairs, scans, latency.

**Speaker script:** The clean experiment compares complementary selection against simple baselines: dense top-ranked selection, random selection, rank-stratified sampling, diversity-aware MMR and deeper retrieval. We must measure unseen-query nDCG and Recall, relevant documents newly entering the candidate pool, number of teacher pairs, duplicated retrieval slots, search work and latency. If complementary selection helps only because it performs more searches, or cannot beat a simple equal-budget baseline, we should reject or revise the hypothesis. We also must not claim an officially faithful DISCo implementation without building and validating one.

### Slide 20 — Takeaway and limitations (13:05–13:30; 0:25)

**Display:** "Feedback quality matters. Evidence first." Three lines: frozen weights / negative results / testable next step.

**Speaker script:** The project began with expensive-feedback post-training but evolved into an inference-time question under frozen model weights. We reproduced and tested relevant mechanisms, documented small or negative results, and identified a controlled next experiment: whether complementary retrieval improves the usefulness of a fixed feedback budget. The conclusion is a sharper research question, not a claimed breakthrough. Thank you.

## Timing summary

| Segment | Slides | Duration |
|:--|:--|--:|
| Problem and motivation | 01–05 | 2:50 |
| Prior inference-time methods | 06–10 | 3:30 |
| Experimental evidence | 11–17 | 5:20 |
| Hypothesis and conclusion | 18–20 | 1:50 |
| **Total** | **20** | **13:30** |

## Source and claim ledger

1. Microsoft Research challenge, "Efficient and accurate post-training of retrieval models": https://www.microsoft.com/en-us/research/academic-program/microsoft-research-fellowship/research-challenges/
2. In-repository audited SciFact report, experiment protocol and audit: [RESULTS](../experiments/inference-time-ceous-20261115/RESULTS.md), [AUDIT_RESULTS](../experiments/inference-time-ceous-20261115/AUDIT_RESULTS.md), [PROTOCOL](../experiments/inference-time-ceous-20261115/PROTOCOL.md).
3. In-repository coordinated research report: [FINAL_REPORT_WORKING.md](../FINAL_REPORT_WORKING.md). SciFact is cache-only and not a blind holdout.
4. SCIDOCS ReFIT and FiQA TTT-Embed results: carried forward from [the existing video-script record](CEOUS_VIDEO_SCRIPT_COMPLEMENTARY_FEEDBACK_20261116.md). These are separate experiments, evaluated with different metrics, and no new rerun is implied by this source.
5. DISCo-like complementary feedback is a **hypothesis** and not a verified DISCo system or proven experimental gain. See [RESEARCH_STATUS.md](../RESEARCH_STATUS.md), H3–H6 proposals.
6. Published TOUR, ReFIT, TTT-Embed and DISCo results must be attributed to their respective papers in the visual site; do not label their metrics as this team's measurements.
7. Any future additions require exact model, corpus, split, seed, metric, run provenance and status; never invent missing values.

## Website implementation acceptance checks

- True full-viewport slide mode, keyboard arrows, click next/back and 20-page progress marker.
- Every page has a speaker-notes toggle with **this page's complete text** and visible target time; default presentation view hides notes from the audience.
- Optional presenter timer counts elapsed talk time (does not advance pages automatically).
- All four research statuses are visually separated: prior published method; completed experiment; failed/negative result; proposed hypothesis.
- Numerals and plotted graphs must come from sourced values above; no fabricated statistics.
- Works on phone and laptop without external sign-in and without requiring a network connection after the HTML loads.
