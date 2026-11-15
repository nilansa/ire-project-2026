# Ceous IRE report — research and drafting record

**Report:** Reranker-Guided Test-Time Retrieval with a Fixed Index  
**Team:** Nilanjan Sarkar and Raman Rayaprolu, IIIT Hyderabad  
**Draft date:** 15 November 2026  
**Status:** two-page conference-style draft; **not submitted**; **no new experiments run for this draft**.

## Read the current draft

- [Two-page PDF on Drive](https://drive.google.com/file/d/1Lz5QaN9burvXNryGBQ0hKmy1LAOuRmY9/view?usp=drivesdk) — verified file metadata, 313,648 bytes.
- [Overleaf-compatible LaTeX source ZIP on Drive](https://drive.google.com/file/d/1kkkaD9z3ttAoW-0uo6JULRPmttKpXhNW/view?usp=drivesdk) — contains `main.tex`, `references.bib`, `main.bbl`, a README, and this research-history note.
- [Presentation upload folder](https://drive.google.com/drive/folders/1fDPo9M7gxhj0um6OW5E_nASB81JyA44-) — created but upload pending; anonymous public access has **not yet been enabled or verified**.

## How the question changed

1. The original Microsoft Research challenge addressed efficient post-training of retrieval models from costly reward-model or cross-encoder feedback: which pairs to score, which loss to use, and how to minimize expensive repeated training.
2. The archived Ceous ANN experiment asked whether HNSW versus IVF candidate selection changed the supervision obtained under a fixed teacher budget. On a positive-enriched, 50,000-passage MS MARCO diagnostic, recorded MRR@10 was 0.890777 for untouched ANCE, 0.866675 for HNSW+teacher, and 0.855858 for IVF+teacher. Both teacher-trained branches were worse in this **single-seed, short-run** diagnostic. Do not promote it to a general claim about training.
3. The present inference-time question is whether the fixed retriever can recover relevant documents missed by its initial candidate set by exploiting a limited number of reranker scores, without retraining encoders or rebuilding the document index.
4. TOuR (Findings ACL 2023), ReFIT (arXiv 2023), and Reddy et al. (SIGIR 2025) already perform/assess reranker-guided inference-time retrieval updates. Query steering per se is **not novel**. DISCo (ICLR 2026) is collective token-coverage retrieval, not proof of reranker-guided recall gains. TTT-Embed and Q-REACT (2026 preprints) further constrain novelty claims.
5. The study must compare deeper retrieval, direct same-pool reranking, ordinary and teacher-weighted pseudo-relevance feedback, and a TOuR/ReFIT-like baseline at matching teacher-pair, retrieval-work, duplicate-slot, and latency budgets. External qrels—not the reranker's own judgments—determine effectiveness.

## Evidence and limits

The current `RESEARCH_STATUS.md` records a historical SciFact cached descriptive audit: 5,183 corpus vectors, 505 teacher-scored queries, 504 of 568 judged positive query–document pairs retrieved in the initial top 50, and 57 queries with no judged positive there. **These figures were not newly reproduced or independently audited for this draft**; none is a result of the proposed test-time method.

No new training, test-time optimization, reranking, inference, plotting, or statistical testing was executed while writing this report. The working paper is a research hypothesis, mathematical formulation, relevant literature, prior negative evidence, and prospective matched-cost evaluation protocol.

## Sources reviewed

Uploaded Microsoft Research proposal text; *Results of the Big ANN: NeurIPS'23 competition* (arXiv:2409.17424), related Big ANN slide PDF and PPTX; earlier Ceous Drive drafts and evidence ledger; `RESEARCH_STATUS.md` in this repository; primary research publications and preprints from ACL, SIGIR, ICLR, NeurIPS, and arXiv.

## Publication and access

The PDF and editable LaTeX ZIP are located in the existing private Ceous IRE final-submission folder on the `ramforzero@gmail.com` Drive. They are readable by that account; public-read sharing is not claimed for the PDF/ZIP. The video folder is real, but currently private, awaiting a recording and an explicit Google Drive "Anyone with the link – Viewer" share setting. The Overleaf MCP starts locally, but authenticated account/project operations are not verified, so the report was compiled with local pdfLaTeX rather than directly modified in an Overleaf project.

Existing research coordination continues under `ire-final-submission-20261115-1943`. No running experiment or peer review was interrupted or duplicated by this drafting update.
