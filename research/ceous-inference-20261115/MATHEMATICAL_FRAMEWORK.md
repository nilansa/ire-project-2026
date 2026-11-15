# Ceous inference-time retrieval: mathematical framework

Status: research design and conditional mathematics, not an empirical result or novelty claim. The root's Astra theory reviewer reported a completed analysis; implementation and protocol review remain pending. Definitions and proposed comparisons below must be frozen for a particular run before evaluation. The companion [coordination tracker](../../RESEARCH_STATUS.md) records ownership, evidence and changes.

## Setup and what can be observed

For textual query `x`, let `q₀` be a fixed encoder's vector and `dᵢ` a frozen corpus vector. Initial dense score is `q₀ᵀdᵢ`. A teacher produces `rᵢ` for the original text `x` and scored documents, while independent relevance `yᵢ` is unknown to the predictor. Let `D` be the full corpus, `U` the set exposed by a retrieval procedure, `F ⊆ U` the teacher-scored subset, `C ⊆ U` the final candidate set of at most K items, and `L` the displayed ranking of at most k items selected from `C`. A moved query or additional direction can expose new documents, so define `U` separately for each method. Encoders and the index stay fixed; per-query state may be optimized.

Cost is a vector, not one scalar: `(final K, teacher query–document scores including final reranking, returned slots including duplicates, unique documents, ANN probes/work, wall time)`, supplemented by CPU/GPU work. A full-corpus scan is work even if no additional teacher score is requested. For cached scores distinguish *logical* requested pairs, *physical fresh* teacher calls, and historical cache production. Cache-only timing excludes teacher inference and is not full end-to-end deployment latency.

## Objective and gap decomposition

Take nonnegative nonincreasing rank weights `wⱼ`, nonnegative gain `g`, and a fixed positive normalization `Z` computed from the same full judgment set. Pad short lists with zero gain. Define

`J(L; y) = Σⱼ wⱼ g(y[Lⱼ]) / Z`, and `J*(A; y)` as the maximum over lists of up to k distinct items drawn from `A`.

For nested `D ⊇ U ⊇ C ⊇ L`, the identity

`J*(D) − J(L) = [J*(D) − J*(U)] + [J*(U) − J*(C)] + [J*(C) − J(L)]`

separates outside-exposure, selection, and ordering gaps. Each term is nonnegative under these assumptions because maximizing over a larger feasible set cannot be worse. `J*(D)` is a judgment-based oracle, not a prediction or a true upper bound under unknown relevance. For binary judged relevant set `R` with `|R| > 0`, oracle candidate recall at K is `min(K, |U ∩ R|)/|R|`. Unjudged documents are not proven irrelevant; all oracle and gain statements are relative to available judgments.

Compute nDCG's IDCG from **all benchmark qrels for the query**, never merely qrels whose documents happen to occur in the candidate set. Regression example: at nDCG@2, ranked `[A, X]`, with A and B both relevant and X not relevant, scores `1/(1 + 1/log₂(3)) ≈ 0.613147`, not 1.

## Geometry and expressivity

For any query-dependent matrix `A_q` that is fixed across documents at scoring time,

`qᵀ A_q d = (A_qᵀq)ᵀd`.

This remains true when feedback determines `A_q`. Conversely, for nonzero `q` and any target vector `v`, choosing `A_q = qvᵀ/(qᵀq)` gives `A_qᵀq = v`. Thus a bilinear transform changes parameterization, constraints or regularization; it does not extend the ranking family of an unconstrained single vector. A fixed weighted sum of linear scores from multiple vectors also collapses to one vector.

A maximum need not collapse: for `d₊=(1,0)`, `d₋=(−1,0)`, `d₀=(0,0)` and `q₊=(1,0)`, `q₋=(−1,0)`, `max(q₊ᵀd,q₋ᵀd)` ranks both extremes at 1 above the midpoint at 0; one linear vector cannot strictly rank both extremes above the midpoint. Likewise, a relevant document that is a convex combination of irrelevant vectors cannot strictly outrank all of them under one linear query. These are geometric capacity observations, not a claim that multi-query search improves relevance.

An exception to the bilinear collapse is `cos(q, A_q d)`: the document-dependent denominator `||A_q d||` generally changes scores. Exploiting it can require a changed index or search procedure; account for that work. A scalar linear residual in `d` is just query movement. A genuinely nonlinear residual can change within-pool ordering but cannot discover outside-pool items without another search mechanism.

## A conditional movement bound

Let `H` be symmetric positive definite and `F` have an H-metric smoothness bound in the relevant neighborhood:

`F(z + Δ) ≥ F(z) + ∇F(z)ᵀΔ − (L_H/2)||Δ||²_H`.

Suppose `||ĝ − ∇F(z)||_{H⁻¹} ≤ ε` and `a = ||ĝ||_{H⁻¹} > 0`. For `Δ = r H⁻¹ĝ/a`, Cauchy–Schwarz gives

`F(z + Δ) − F(z) ≥ r(a − ε) − L_H r²/2`.

If `a > ε`, `L_H > 0`, and the smoothness/error conditions hold throughout the proposed step, choose `r = min(ρ, (a − ε)/L_H)` for trust radius `ρ > 0`. This lower bound concerns the **specified smooth, fixed-support objective** only: it does not guarantee improved top-K ranking, generalization, or unseen relevance. If the assumptions or error bound are unknown, the formula supplies no empirical guarantee.

For fixed feedback set `F`, `pᵢ(z) = softmax(zᵀdᵢ/τ)` with `τ > 0`, and teacher utility `F_r(z) = Σᵢ pᵢ(z)rᵢ`,

`∇F_r(z) = (1/τ) Σᵢ pᵢ(z)(rᵢ − F_r(z))dᵢ`.

If on this support `|rᵢ − yᵢ| ≤ δ` and `||dᵢ||_{H⁻¹} ≤ D₀`, then `||∇F_r − ∇F_y||_{H⁻¹} ≤ 2D₀δ/τ`. The calibration assumption between teacher scores and relevance is unverified, and the bound says nothing about labels outside F. Indeed, two worlds with identical observed vectors, teacher scores and histories can assign opposite labels to unseen documents; no method can guarantee an unseen relevance improvement in both without extra linking assumptions.

## Six falsifiable mechanisms

| Branch | Proposed mechanism and counterhypothesis | Essential comparison and limit |
|---|---|---|
| H1 | Teacher-guided movement finds judged positives outside initial U; alternatively, equal-cost deeper retrieval or reranking explains any gain. | Untouched, dense/teacher Rocchio, and TOUR-inspired optimization; track gained **and lost** judged positives. |
| H2 | Shrinkage/local covariance changes step direction beneficially; alternatively isotropic movement with the same gradient and norm suffices. | Same initial gradient and Euclidean norm of the complete applied displacement after weight decay and clipping; declare any final renormalization and apply it consistently. Declare centering, divisor, dtype and shrinkage; test Woodbury small-matrix against a direct solve. |
| H3 | Nonlinear residuals improve ordering inside a fixed pool; alternatively teacher reranking or a linear query shift suffices. | Same-pool reranking and matched total costs; no discovery claim without another search. |
| H4 | Adaptive allocation concentrates search/teacher effort on queries with high marginal value; alternatively a static schedule performs as well at equal aggregate cost. | Static schedule, per-query and total-cost accounting; proposed Claude ownership requires explicit ACK. |
| H5 | Disagreement, leverage or exploration identifies useful feedback; alternatively top-score or random selection is sufficient. | Known sampling propensities where applicable, random/top-score controls and selection-bias analysis; independent labels. Defer until core baselines are reviewed. |
| H6 | Nonlinear max across directions discovers complementary relevant documents; alternatively equal-work deep single-query or random directions suffice. | Count all probes, returned slots and duplicates; coverage is not relevance. Proposed Claude ownership requires explicit ACK. |

Each mechanism needs a prespecified prediction, comparator, control, development cap and rejection decision. Independent review and planted-truth/negative controls precede interpretable findings; rejected and invalidated results remain in the ledger.

## Curvature and the H2 direction comparison

For a fixed set of `n` feedback documents and positive integer embedding dimension `d`, let `D ∈ ℝⁿˣᵈ` contain one frozen document vector per row, `q ∈ ℝᵈ` be the query vector, `T > 0` be the student temperature, and `p ∈ ℝⁿ` be a fixed teacher probability vector. Define `s(q) = softmax(Dq/T) ∈ ℝⁿ`, the student probability vector over those same `n` documents, and `L(q) = KL(p || s(q))`, a scalar loss. With fixed `p` and support, differentiating the softmax cross-entropy gives gradient `∇L(q) = Dᵀ(s − p)/T`. The softmax Jacobian is `(diag(s) − ssᵀ)/T` with respect to the unscaled scores `Dq`: `diag(s)` puts the probabilities on the diagonal, while `ssᵀ` is their outer product. Thus the exact Hessian, the `d × d` matrix of second derivatives of this scalar loss, is

`∇²L(q) = Dᵀ(diag(s) − ssᵀ)D / T²`, with `rank(∇²L) ≤ min(d, n − 1)`.

The inner matrix is the student-probability-weighted covariance of the fixed feedback document vectors; the rank bound follows because it annihilates the all-ones vector. Uniform covariance over the initial 50 documents, used as the H2 proposal, is not generally this Hessian or the exact Fisher: it uses a different support and weights unless those happen to coincide. For the directional control, equal Euclidean step means equal length of the **complete applied displacement** after weight decay and clipping, with any subsequent normalization accounted for; matching a raw optimizer update alone is insufficient.

## TOUR as an essential prior, not a novelty claim

In [TOUR](https://aclanthology.org/2023.findings-acl.354.pdf), the soft-label instance uses teacher probabilities `p = softmax(t/0.5)` and student probabilities `s = softmax(Dq)` with unit student temperature. Minimizing `KL(p || s)` by a plain gradient step gives `q⁺ = q + ηDᵀ(p − s)` when rows of D are feedback vectors. The paper's DPR passage setup reports learning rate 0.2, one iteration, retrieval and reranking depth 100, momentum 0.99, weight decay 0.01, and linear learning-rate scheduling; actual optimizer details affect an implementation. The [authors' repository](https://github.com/dmis-lab/TouR) should also be checked for code-specific behavior. The paper's soft-label early stop checks whether the top dense-retrieved item has the maximum teacher score; its hard-label variant uses membership in a nucleus-style pseudo-positive set. In the released phrase-retrieval code, the soft path can also use nucleus-based stopping. Declare the exact code path and stopping condition; the paper description and released code are not interchangeable.

The proposed single-step MiniLM/MedCPT study using 20 feedback scores is **TOUR-inspired transfer**, not an exact reproduction of the paper's DPR/depth-100 settings; the real experiment has not run. [Reddy et al. (SIGIR 2025)](https://research.ibm.com/publications/a-large-scale-study-of-reranker-relevance-feedback-at-inference) is direct prior work. [IBM's task-aware embedding refinement repository](https://github.com/IBM/task-aware-embedding-refinement) and the [TTT-Embed preprint](https://tianyucodings.github.io/publications/) are close additional work; detailed method-by-method comparison remains pending.

## Exploratory SciFact study: design checks, not results

The recovered full-corpus SciFact cache is reported to contain 5,183 × 384 normalized MiniLM corpus vectors, 809 mining/train query vectors and 300 test vectors; a separate teacher cache covers 505 train queries × 50 documents for MedCPT and BGE with query/document IDs. Test-query teacher feedback is not available in that cache. Validate hashes, identities, row alignment, exact corpus/query mapping and access terms before use. The 505 previously exposed queries and any subsets are **exploratory**, not a blind holdout. A full-corpus re-retrieval comparison can be new, but is not a full MS MARCO experiment.

For an initial pool `U₅₀` and scored `F₂₀`, distinguish `F₂₀`, `U₅₀ \ F₂₀`, and `D \ U₅₀`. Only the last is outside the initial pool. Report judged relevant gained and lost. Reranking the same 20 uses the same logical feedback but is not equal compute to a query update followed by a full-corpus scan; reranking 20 cannot change Recall@20 when all 20 remain. Screen duplicate normalized query-text groups before metric computation, exclude cross-split groups without using relevance labels and report the actual sample size. Prespecified development qrels can select from a finite global hyperparameter set; evaluation qrels never guide adaptation, tuning or stopping.

Hash all predictions before opening evaluation qrels. Use paired seeded bootstrap as exploratory uncertainty, not a best-arm significance test. Stop weak branches by the predeclared development cap. The older P0-prime design remains failed and must not be run; a corrected study has its own protocol and review, not an automatic carry-over of that invalid design.
