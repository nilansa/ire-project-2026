# P5 branch OOD_COVDECOMP_20261115 — PREREGISTRATION (frozen before any data access)

Status: FIRST PILOT, frozen retriever weights, no training, no tuning. Does NOT claim the research direction must be training-free.
Separate branch; does not modify existing P5 final submission, Stage-1/2 files, or gold_HIDDEN.json (never opened).

## Question
On unseen domains, does (i) complementary token-coverage set selection or (ii) natural-language decomposition + fusion improve TRUE relevance
(full qrels Recall@10/20, nDCG@10) and multi-hop all-supporting-evidence@K over frozen monolithic retrieval, at reported cost?

## Prior-art positioning (verified 2026-11-15)
- DISCo (Nair et al., ICLR 2026, proceedings PDF sha256-prefix 01ae56862fd6d5ea): facility-location coverage over CONTEXTUAL ColBERT query-token
  embeddings; marginal gain = sum over query atoms of positive improvement over current best; query vectors are "edited" each round; PLAID-style
  lifted index. NO explicit LLM sub-questions (0 occurrences of "decompos"/"sub-quer"). Evaluated on BEIR MS-MARCO/HotpotQA/FEVER/NFCorpus and LoTTE
  forum queries; headline metric is coverage objective vs latency; gold-set check only on HotpotQA K=2. Not designed for teacher/reranker top-K agreement.
- Yin et al., arXiv 2606.08577 (Findings EMNLP 2026): decomposition often HARMS first-stage retrieval (misses targets) but helps reranking.
- Not novel in themselves: set coverage, sufficiency detection, stopping (DISCo, SetR, Sufficient Context, HALT, BEAR).
- Our arm A5/A6 = "DISCo-inspired" exact greedy on a candidate pool; NOT official DISCo (official needs PLAID index; CPU build failed at Stage-2).

## Mechanisms (causally distinct)
M1 Set complementarity (in-pool): coverage greedy promotes pool docs that cover not-yet-covered query atoms. Prediction: A5 > A4 on MuSiQue
   all-support@10, ~0 or negative on single-relevant SciFact. Counter: gains are a scorer effect (M2). Comparator: A4 (same encoder, same pool, relevance-only).
M2 Representation/scorer shift (in-pool, not a set effect): late-interaction token scoring beats single-vector dense OOD. Prediction: A4 > A3 and A5 ≈ A4.
M3 Out-of-pool discovery by query reformulation (LLM decomposition A7/A7b or residual-query mutation A6): reaches gold absent from the monolithic
   top-200 pool. Prediction: out-of-pool share of top-K gold > 5% and all-support@20 gain on MuSiQue; Yin predicts harm for decomposition-only (A7b)
   and on single-facet OOD queries.

## Data (sampling unit = query; no tuning, so all sampled queries are evaluation)
- BEIR SciFact test (300 q) and BEIR NFCorpus test (323 q): 100 each, order by sha256('p5-ood-20261115:'+qid), take first 100.
  Note: these test sets were used for a single confirmatory look in the separate IRE P1 project (different method, teacher KD); no P5 design choice uses them.
- MuSiQue-Ans dev 2-hop (local stage1_musique/data/musique_ans_v1.0_dev.jsonl), eligible = 2hop__ ids with exactly 2 distinct supports, EXCLUDING the
  150 Stage-1 ids (read ids only from stage1_musique/sample_questions.json); 100 by same salted hash. Corpus = stage1_musique/corpus.jsonl (21,100 docs,
  hash recorded). Supports derived from raw dev jsonl in score phase only. gold_HIDDEN.json is never read.
- LoTTE: not used in this pilot (multi-GB corpora); listed as next step.

## Frozen arms (K in {10,20}, pool B=200)
A1 BM25 (rank_bm25 BM25Okapi defaults; lowercase \w+ tokenization).
A2 Dense: sentence-transformers/msmarco-distilbert-base-tas-b (MS MARCO-only; pinned revision recorded), dot product, exact search.
A3 RRF(A1,A2), k=60 -> base pool P = top-200.
A4 ColBERTv2 MaxSim rerank of P (local pinned colbertv2.0 checkpoint; standard query augmentation, query maxlen 32, doc maxlen 180).
A5 DISCo-inspired coverage greedy on P: F(S)=sum_q max_{d in S, x in d} q.x over ColBERT query embeddings; exact greedy order for 20 slots.
A6 Residual-query mutation (one round): after A5 picks 3 docs, residual words = non-stopword query words whose ColBERT tokens' coverage is below the
   median coverage of non-special query tokens; residual query -> RRF(BM25,dense) top-100 added to P; continue A5 greedy for remaining slots.
A7 LLM decomposition + fusion (Yin "keep original"): qwen3.5:4b-mlx via local Ollama, temperature 0, seed 0, thinking off, fixed prompt, <=3 sub-queries;
   each sub-query and the original -> RRF(BM25,dense) top-200; fuse all lists with RRF k=60.
A7b Decomposition-only fusion (original excluded) — Yin replication arm.
A8 Coverage greedy (A5 objective, original-query atoms) on pool = union of A7 lists (top-200 each).
Prompt fixed in code before running; format sanity-checked on 5 SciFact TRAIN queries (output format only, no metrics).

## Metrics (scored ONLY by score.py, which alone reads qrels/supports)
Primary: MuSiQue all-support@10 (both supports in top-10). Also all-support@20, Recall@10/20, nDCG@10 (graded where qrels graded) on every dataset.
BEIR complete@K reported for queries with 2<=|rel|<=K.
Attribution: for each arm, relevant docs in top-K split into in-pool (in P) vs out-of-pool, and promotions (P-rank > K moved into top-K).
Pool ceilings: Recall@200 and all-support@200 of P.
Cost: per-query wall latency per arm (incl. encoding), #LLM calls, LLM output tokens, #ColBERT doc encodings, #retrieval calls.
Secondary (time cap 25 min, optional): top-10 overlap with bge-reranker-v2-m3 teacher ranking of the union of all arms' top-20 — agreement is NOT relevance.
Statistics: paired bootstrap over queries, 1000 resamples, seed 0, 95% percentile CI, per dataset; no pooling across datasets.

## Decision rules (fixed now)
D1 (M1): Delta = A5 - A4 MuSiQue all-support@10. Support if >= +0.05 and CI lower > 0.
D2 (M3): A7 - A3 MuSiQue all-support@20 >= +0.05, CI lower > 0; Yin-consistent harm if A7b - A3 nDCG@10 <= -0.02 with CI upper < 0 on a BEIR set.
D3 non-regression: A5 - A4 and A7 - A3 nDCG@10 on SciFact and NFCorpus >= -0.02 (point estimate).
D4 (M2): A4 - A3 nDCG@10 / all-support@10 reported; if M2 positive and D1 null -> gain is scorer, not coverage.
Verdict: CONTINUE if D1 or D2 supported with D3; PIVOT if only M2 or only out-of-pool discovery without all-support gain; STOP if all null/harmful.
n=100 per dataset: CI half-width ~0.05-0.08; this is a pilot, effect sizes < 0.05 are not resolvable.

## Controls
Positive (planted truth): synthetic 2-facet query with 5 near-duplicate facet-X docs + 1 facet-Y doc + distractors encoded by the real ColBERT;
  expected: A5 top-2 contains facet-Y doc, A4 top-2 does not. Metric positive control: oracle ranking -> recall=1, all-support=1.
Negative N1: qrels permuted across queries -> all arms near 0. N2: coverage greedy with random Gaussian token embeddings must score below A5 and
  near random pool reordering. N3 leakage: retrieval phase runs before qrels/supports files are created; static check that retrieval code never
  references qrels/supports/gold.

## Budget / safety
Local CPU/MPS only; total wall cap 2.5 h; downloads limited to TAS-B model and BEIR scifact/nfcorpus zips (public); no paid APIs, no GPU cluster,
no retraining, no loops beyond one residual round. One run; no reruns after seeing scores except quarantined-with-reason.
Independent cold review (separate Terra task, not the implementer) before any claim.
