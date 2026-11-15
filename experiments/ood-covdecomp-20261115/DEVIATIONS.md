# Deviations and setup notes

- The checkpoint directory has no `CHECKPOINT.json`; the run will hash the available weight, config, tokenizer, and `artifact.metadata` files directly.
- The existing Stage-2 virtual environment is reused read-only; no packages are installed. `rank_bm25` is absent, so BM25Okapi's documented default formula is implemented locally with `k1=1.5`, `b=0.75`, `epsilon=0.25`, and lowercase `\w+` tokenization.
- Preparation reads only the first BEIR qid column to identify the frozen test/train query sets; it does not read relevance values or write label files. Full relevance labels and MuSiQue supports are read/written only by `score.py` after retrieval.
- An initial train-only format probe returned a JSON array and failed before output-format validation. One API call was made; its token count and raw text were not retained by that failed implementation. The prompt was clarified with an explicit object schema before the successful five-query smoke. No test labels or metrics were involved.
- The successful five-query SciFact train smoke returned valid JSON objects for all five calls (25 output tokens); each contained zero subqueries. This was recorded as output-format sanity only and did not trigger prompt changes or metrics.
- The prior IRE_P5 adapter refers to the earlier hidden-label boundary. This branch follows the explicit task authorization for public MuSiQue dev annotations in `score.py`; `gold_HIDDEN.json` remains out of scope and untouched.
- The optional teacher comparison is deferred; independent cold review remains pending before any scientific claim.
