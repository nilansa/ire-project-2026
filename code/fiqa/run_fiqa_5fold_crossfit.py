#!/usr/bin/env python3
"""FiQA2018 five-fold cross-fitted task-wise TTT-Embed (CPU only, cache only).

Task ID: ire-tttembed-fiqa-5fold-crossfit-20261116-1125
Spec:    shared/RUN_SPEC.md (sha256 8b5ae8d6489b5f7347ed25df39be2c916458160fc8881f529881b24ed76814fa)

Fold 0 = the exact existing 518/130 split of task ire-tttembed-fiqa-8q-heldout-20261116-1121.
Folds 1-4 = the 518 fold-0 rewarded IDs partitioned by numpy.random.RandomState(1006).permutation(518)
into consecutive groups of 130,130,129,129. Each fold fits one vector on its training queries'
cached top-10 teacher rewards only. Qrels are opened only after all five vectors are fitted.
Every held-out query is searched against all 57,638 frozen document embeddings.
Zero embedding calls, zero reranker calls, no network, no GPU.
"""
from __future__ import annotations
import hashlib, json, math, os, platform, shutil, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

TASK_ID = "ire-tttembed-fiqa-5fold-crossfit-20261116-1125"
ROOT = Path("/Users/nilan/Documents/ire-project-2026/artifacts/fiqa-qwen3-tttembed-20261116")
TASK_ROOT = Path("/Users/nilan/.codex/scratch/cross-surface/ire-project") / TASK_ID
OUT = Path(os.environ.get("CROSSFIT_OUT", str(TASK_ROOT / "shared/results")))
STOP_BEFORE_QRELS = os.environ.get("CROSSFIT_STOP_BEFORE_QRELS") == "1"
SPEC = TASK_ROOT / "shared/RUN_SPEC.md"
SPEC_SHA = "8b5ae8d6489b5f7347ed25df39be2c916458160fc8881f529881b24ed76814fa"
PRIOR = Path("/Users/nilan/.codex/scratch/cross-surface/ire-project/ire-tttembed-fiqa-8q-heldout-20261116-1121/shared/results_80pct")

DOC_FILE = ROOT / "fiqa_documents_qwen3_0.6b_f32.npy"
DOC_IDS_FILE = ROOT / "fiqa_doc_ids.json"
QUERY_FILE = ROOT / "fiqa_queries_qwen3_0.6b_f32.npy"
TOP50_FILE = ROOT / "fiqa_top50.json"
TEACHER_FILE = ROOT / "fiqa_qwen3_reranker_top10_teacher.jsonl"
QUERY_INPUT = ROOT / "input/queries_test.json"
QRELS_FILE = ROOT / "input/qrels_test.json"
EMBED_MANIFEST = ROOT / "embedding_manifest.json"
FINAL_MANIFEST = ROOT / "FINAL_MANIFEST.json"
BASELINE_RESULTS = ROOT / "TTT_BASELINE_RESULTS.json"
CHECKSUMS = ROOT / "CHECKSUMS.sha256"

SPLIT_SEED = 1006          # paper rule 1000 + t, FiQA2018 t = 6 (as in fold-0 run)
FOLD_SEED = 1006           # RUN_SPEC: recorded seed for folds 1-4
RNG_IMPL = "numpy.random.RandomState (MT19937), .permutation"
FOLD_SIZES_1_4 = [130, 130, 129, 129]
DEPTH = 10
TEACHER_TEMP, STUDENT_TEMP, LAMBDA, LR, STEPS = 0.02, 0.05, 0.3, 0.01, 300
BOOT_SEED, BOOT_REPS = 20261116, 10000
N_DOCS, DIM, N_TEST, N_QROWS = 57638, 1024, 648, 1148


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s); st.flush()
    def flush(self):
        for st in self.streams: st.flush()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ndcg10(doc_ids, rel):  # identical to fold-0 run implementation
    ideal = sorted((int(x) for x in rel.values()), reverse=True)[:10]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    if idcg == 0:
        return 0.0
    dcg = sum((2**int(rel.get(str(d), rel.get(d, 0))) - 1) / math.log2(i + 2)
              for i, d in enumerate(doc_ids[:10]))
    return dcg / idcg


def paired_bootstrap(delta, seed=BOOT_SEED, reps=BOOT_REPS):  # identical to fold-0 run
    rng = np.random.default_rng(seed)
    n = len(delta)
    samples = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        samples[i] = np.mean(delta[rng.integers(0, n, n)])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(reps=reps, seed=seed, rng="numpy.random.default_rng (PCG64)",
                ci95_low=float(lo), ci95_high=float(hi),
                bootstrap_probability_positive=float(np.mean(samples > 0)))


def stratified_bootstrap(deltas_by_fold, seed=BOOT_SEED, reps=BOOT_REPS):
    """Resample queries within each fold (fold sizes fixed); pooled mean over 648."""
    rng = np.random.default_rng(seed)
    total = sum(len(d) for d in deltas_by_fold)
    samples = np.empty(reps)
    for i in range(reps):
        s = 0.0
        for d in deltas_by_fold:
            s += d[rng.integers(0, len(d), len(d))].sum()
        samples[i] = s / total
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return dict(reps=reps, seed=seed, ci95_low=float(lo), ci95_high=float(hi),
                bootstrap_probability_positive=float(np.mean(samples > 0)))


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "code").mkdir()
    log_f = (OUT / "run_progress.log").open("w")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    t0 = time.time()
    torch.set_num_threads(min(8, os.cpu_count() or 8))  # same as fold-0 run
    torch.manual_seed(0)  # not used by the deterministic fit; recorded for completeness
    env = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
               tqdm=__import__("tqdm").__version__, platform=platform.platform(),
               machine=platform.machine(), torch_threads=torch.get_num_threads(),
               executable=sys.executable)
    print("TASK", TASK_ID, "| ENV", json.dumps(env), flush=True)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)

    # ------------------------------------------------------------- integrity gate
    em = json.loads(EMBED_MANIFEST.read_text())
    fm = json.loads(FINAL_MANIFEST.read_text())
    ck = {}
    for line in CHECKSUMS.read_text().splitlines():
        if line.strip():
            d, name = line.split(None, 1); ck[name.strip()] = d
    expected = {
        SPEC: [SPEC_SHA],
        DOC_FILE: [em["doc_matrix_sha256"], ck["fiqa_documents_qwen3_0.6b_f32.npy"]],
        DOC_IDS_FILE: [em["doc_ids_sha256"], ck["fiqa_doc_ids.json"]],
        QUERY_FILE: [em["query_matrix_sha256"], ck["fiqa_queries_qwen3_0.6b_f32.npy"]],
        TOP50_FILE: [em["top50_sha256"], ck["fiqa_top50.json"]],
        TEACHER_FILE: [fm["teacher_sha256"], ck["fiqa_qwen3_reranker_top10_teacher.jsonl"]],
        QUERY_INPUT: [ck["input/queries_test.json"]],
        QRELS_FILE: [ck["input/qrels_test.json"]],  # hash only; contents NOT parsed until after fits
        EMBED_MANIFEST: [ck["embedding_manifest.json"]],
        FINAL_MANIFEST: [ck["FINAL_MANIFEST.json"]],
        BASELINE_RESULTS: [ck["TTT_BASELINE_RESULTS.json"]],
    }
    prior_art = json.loads((PRIOR / "ARTIFACTS_SHA256.json").read_text())
    for name in ["selected_feedback_queries.json", "RESULTS.json", "heldout_rankings.json",
                 "ttt_fiqa_8q_seed1006_v_f32.npy", "RUN.md"]:
        expected[PRIOR / name] = [prior_art[name]["sha256"]]
    input_hashes = {}
    for path, digests in tqdm(expected.items(), desc="sha256 inputs", file=sys.stderr):
        got = sha(path)
        if any(got != d for d in digests):
            raise RuntimeError(f"SHA256 mismatch: {path}: {got} vs {digests}")
        input_hashes[str(path)] = dict(sha256=got, bytes=path.stat().st_size,
                                       verified_against=len(digests))
    print("SHA_OK", len(input_hashes), "files", flush=True)

    docs = np.load(DOC_FILE, mmap_mode="r")
    doc_ids = [str(d) for d in json.loads(DOC_IDS_FILE.read_text())]
    qvec = np.load(QUERY_FILE, mmap_mode="r")
    top50 = json.loads(TOP50_FILE.read_text())
    test_queries = json.loads(QUERY_INPUT.read_text())
    assert docs.shape == (N_DOCS, DIM) and docs.dtype == np.float32, docs.shape
    assert len(doc_ids) == N_DOCS and len(set(doc_ids)) == N_DOCS
    assert qvec.shape == (N_QROWS, DIM) and qvec.dtype == np.float32 and len(top50) == N_QROWS
    assert [r["split"] for r in top50] == ["test"] * N_TEST + ["train_extra"] * (N_QROWS - N_TEST)
    test_rows = top50[:N_TEST]
    test_qids = [str(r["query_id"]) for r in test_rows]
    assert len(test_queries) == N_TEST and len(set(test_qids)) == N_TEST
    assert [str(q["id"]) for q in test_queries] == test_qids, "query file / top50 / embedding order mismatch"
    doc_arr = np.asarray(docs, dtype=np.float32)
    q_test = np.asarray(qvec[:N_TEST], dtype=np.float32)
    assert np.isfinite(doc_arr).all() and np.isfinite(q_test).all()
    dn, qn = np.linalg.norm(doc_arr, axis=1), np.linalg.norm(q_test, axis=1)
    did_to_row = {d: i for i, d in enumerate(doc_ids)}
    for r in test_rows:
        c = r["candidates"]
        assert len(c) == 50 and [x["rank"] for x in c] == list(range(1, 51))
        assert all(str(x["doc_id"]) in did_to_row for x in c)
    shape_checks = dict(doc_matrix=list(docs.shape), doc_dtype=str(docs.dtype), unique_doc_ids=N_DOCS,
                        query_matrix=list(qvec.shape), test_query_rows="0..647", unique_test_qids=N_TEST,
                        doc_norm_min=float(dn.min()), doc_norm_max=float(dn.max()),
                        query_norm_min=float(qn.min()), query_norm_max=float(qn.max()),
                        order_check="queries_test.json ids == top50[:648].query_id (== query-matrix rows)")
    print("SHAPES_OK", json.dumps(shape_checks), flush=True)

    # Teacher cache: exactly 10 records per test query == cached top-10 in rank order.
    teacher = [json.loads(l) for l in TEACHER_FILE.open()]
    teacher_test = [x for x in teacher if x["split"] == "test"]
    assert len(teacher_test) == 6480
    teacher_map, per_q = {}, {}
    for x in teacher_test:
        key = (str(x["query_id"]), str(x["doc_id"]))
        assert key not in teacher_map, "duplicate teacher pair"
        r = float(x["teacher_reward"]); assert 0.0 <= r <= 1.0 and math.isfinite(r)
        teacher_map[key] = r
        per_q.setdefault(key[0], []).append((int(x["base_rank"]), key[1]))
    assert set(per_q) == set(test_qids)
    for r in test_rows:
        qid = str(r["query_id"])
        got = [d for _, d in sorted(per_q[qid])]
        assert len(got) == DEPTH and got == [str(c["doc_id"]) for c in r["candidates"][:DEPTH]], qid
    print("TEACHER_OK 6480 unique test pairs; 10 per query; equal to cached top-10 in rank order", flush=True)

    # ------------------------------------------------------------- folds
    prior_sel = json.loads((PRIOR / "selected_feedback_queries.json").read_text())
    perm648 = np.random.RandomState(SPLIT_SEED).permutation(N_TEST)
    regen_rewarded = [test_qids[int(i)] for i in perm648[:518]]
    assert regen_rewarded == [str(x) for x in prior_sel["rewarded_query_ids"]], "fold-0 split regen mismatch"
    fold0_held = [str(x) for x in prior_sel["heldout_query_ids"]]
    assert set(fold0_held) == set(test_qids) - set(regen_rewarded) and len(fold0_held) == 130
    perm518 = np.random.RandomState(FOLD_SEED).permutation(518)
    bounds = np.cumsum([0] + FOLD_SIZES_1_4)
    held = [fold0_held] + [[regen_rewarded[int(i)] for i in perm518[bounds[k]:bounds[k + 1]]] for k in range(4)]
    qid_row = {q: i for i, q in enumerate(test_qids)}
    held = [sorted(h, key=qid_row.get) for h in held]  # stored in test-row order
    order648 = [test_qids[int(i)] for i in perm648]      # training order = fold-0 permutation order
    folds = []
    for k in range(5):
        hs = set(held[k])
        train = [q for q in order648 if q not in hs]
        folds.append(dict(fold=k, heldout_query_ids=held[k], train_query_ids=train))
    # proofs
    for a in range(5):
        assert not set(folds[a]["train_query_ids"]) & set(folds[a]["heldout_query_ids"])
        assert set(folds[a]["train_query_ids"]) | set(folds[a]["heldout_query_ids"]) == set(test_qids)
        for b in range(a + 1, 5):
            assert not set(held[a]) & set(held[b])
    union = [q for h in held for q in h]
    assert len(union) == N_TEST and set(union) == set(test_qids)
    assert folds[0]["train_query_ids"] == regen_rewarded  # identical ORDER to fold-0 run
    sizes = [(len(f["train_query_ids"]), len(f["heldout_query_ids"])) for f in folds]
    assert sizes == [(518, 130), (518, 130), (518, 130), (519, 129), (519, 129)], sizes
    fold_proofs = dict(pairwise_disjoint_heldout=True, union_equals_648_test_ids=True,
                       each_query_heldout_exactly_once=True, train_heldout_disjoint_every_fold=True,
                       train_plus_heldout_equals_648_every_fold=True, sizes_train_heldout=sizes,
                       fold0_regenerated_from_seed_matches_prior_file=True)
    (OUT / "fold_assignments.json").write_text(json.dumps(dict(
        task_id=TASK_ID, split_seed=SPLIT_SEED, fold_seed=FOLD_SEED, rng=RNG_IMPL,
        fold0_source=str(PRIOR / "selected_feedback_queries.json"),
        construction=("fold0 held-out = prior 130 held-out IDs; folds1-4 held-out = prior 518 rewarded IDs "
                      "(file order) indexed by RandomState(1006).permutation(518), consecutive chunks 130,130,129,129; "
                      "train = other IDs in RandomState(1006).permutation(648) order"),
        proofs=fold_proofs, folds=folds), indent=2))
    print("FOLDS_OK", json.dumps(fold_proofs), flush=True)

    # ------------------------------------------------------------- fit (no qrels in memory)
    assert "qrels" not in dir()
    vectors, fit_info = [], []
    for f in tqdm(folds, desc="fit folds", file=sys.stderr):
        k, train = f["fold"], f["train_query_ids"]
        hs = set(f["heldout_query_ids"])
        fold_teacher = {key: v for key, v in teacher_map.items() if key[0] not in hs}
        assert {key[0] for key in fold_teacher} == set(train) and len(fold_teacher) == 10 * len(train)
        rows = [qid_row[q] for q in train]
        x_train = torch.from_numpy(q_test[rows].copy())
        Z = torch.from_numpy(np.stack([doc_arr[[did_to_row[str(c["doc_id"])] for c in test_rows[i]["candidates"][:DEPTH]]]
                                       for i in rows]))
        rew = torch.tensor([[fold_teacher[(test_qids[i], str(c["doc_id"]))] for c in test_rows[i]["candidates"][:DEPTH]]
                            for i in rows], dtype=torch.float32)
        assert Z.shape == (len(train), DEPTH, DIM) and rew.shape == (len(train), DEPTH)
        p_t = torch.softmax(rew / TEACHER_TEMP, dim=-1)
        v = torch.zeros(DIM, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([v], lr=LR)
        trace, ts = [], time.time()
        for step in tqdm(range(STEPS), desc=f"fold{k} adam", file=sys.stderr, leave=False, mininterval=0.5):
            opt.zero_grad(set_to_none=True)
            a = F.normalize(x_train + v.unsqueeze(0), p=2, dim=-1)        # Eq. 3: no alpha in training
            logp_s = torch.log_softmax(torch.einsum("bd,bkd->bk", a, Z) / STUDENT_TEMP, dim=-1)
            kl = torch.sum(p_t * (torch.log(p_t.clamp_min(1e-12)) - logp_s), dim=-1).mean()
            loss = kl + LAMBDA * torch.sum(v * v)
            loss.backward(); opt.step()
            if step == 0 or (step + 1) % 50 == 0:
                trace.append(dict(step=step + 1, loss=float(loss.detach()), kl=float(kl.detach()),
                                  vector_norm=float(torch.linalg.vector_norm(v.detach()))))
        v_np = v.detach().numpy().astype(np.float32)
        assert np.isfinite(v_np).all()
        n = len(train); alpha = n / (n + 1.0)
        path = OUT / f"fold{k}_v_f32.npy"; np.save(path, v_np)
        with torch.no_grad():  # KL at v=0 for the training set (label-free diagnostic)
            a0 = F.normalize(x_train, p=2, dim=-1)
            lp0 = torch.log_softmax(torch.einsum("bd,bkd->bk", a0, Z) / STUDENT_TEMP, dim=-1)
            kl0 = float(torch.sum(p_t * (torch.log(p_t.clamp_min(1e-12)) - lp0), dim=-1).mean())
        vectors.append(v_np)
        fit_info.append(dict(fold=k, n_train=n, n_heldout=len(hs), teacher_pairs_used=10 * n, alpha=alpha,
                             vector_norm=float(np.linalg.norm(v_np)), train_kl_at_v0=kl0,
                             train_kl_final=trace[-1]["kl"], loss_trace=trace,
                             fit_seconds=time.time() - ts, vector_file=path.name, vector_sha256=sha(path)))
        print(f"FIT fold{k} n={n} |v|={np.linalg.norm(v_np):.6f} KL0={kl0:.6f} KLfin={trace[-1]['kl']:.6f}", flush=True)
    prior_v = np.load(PRIOR / "ttt_fiqa_8q_seed1006_v_f32.npy")
    fold0_v_maxabs = float(np.max(np.abs(prior_v - vectors[0])))
    print("FOLD0_VECTOR_MAXABS_DIFF_VS_PRIOR", fold0_v_maxabs, flush=True)

    # ------------------------------------------------------------- full-corpus search (still no qrels)
    doc_t = torch.from_numpy(doc_arr)
    def full_search(qm, desc, k=10, bs=64):
        ids, scores = [], []
        for s in tqdm(range(0, qm.shape[0], bs), desc=desc, file=sys.stderr, leave=False):
            sims = qm[s:s + bs] @ doc_t.T
            top = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
            ids.extend([[doc_ids[int(j)] for j in row] for row in top.indices.numpy()])
            scores.extend(top.values.numpy().tolist())
        return ids, scores
    q_all = torch.from_numpy(q_test.copy())
    # Identity / null control: v = 0 through the same deployment path as the adapted arm.
    q_null = F.normalize(q_all + (518 / 519) * torch.zeros(1, DIM), p=2, dim=-1)
    null_ids50, null_sc50 = full_search(q_null, "null v=0 search", k=50)
    cached10 = [[str(c["doc_id"]) for c in r["candidates"][:10]] for r in test_rows]
    cached50 = [[str(c["doc_id"]) for c in r["candidates"]] for r in test_rows]
    mism10 = [i for i in range(N_TEST) if null_ids50[i][:10] != cached10[i]]
    mism50 = [i for i in range(N_TEST) if null_ids50[i] != cached50[i]]
    set10 = [i for i in range(N_TEST) if set(null_ids50[i][:10]) != set(cached10[i])]
    cos_diff = max(abs(null_sc50[i][j] - test_rows[i]["candidates"][j]["cosine"])
                   for i in range(N_TEST) for j in range(50) if null_ids50[i][j] == cached50[i][j])
    mism_detail = []
    for i in mism10:
        sc = null_sc50[i][:11]
        mism_detail.append(dict(query_id=test_qids[i], cpu_top10=null_ids50[i][:10], cached_top10=cached10[i],
                                min_adjacent_score_gap_top11=float(min(abs(sc[j] - sc[j + 1]) for j in range(10)))))
    identity = dict(path="normalize(x_q + alpha*0) @ D^T, torch.topk over all 57,638 docs (same as adapted arm)",
                    queries=N_TEST, top10_order_mismatches=len(mism10), top10_set_mismatches=len(set10),
                    top50_order_mismatches=len(mism50),
                    max_abs_cosine_diff_vs_cached_on_matching_positions=float(cos_diff),
                    exact_top10_reproduction=len(mism10) == 0, mismatch_detail=mism_detail)
    (OUT / "identity_control.json").write_text(json.dumps(identity, indent=2))
    print("IDENTITY_CONTROL", json.dumps({k: v for k, v in identity.items() if k != "mismatch_detail"}), flush=True)
    null10 = [r[:10] for r in null_ids50]

    adapted = {}   # (fold, qid) -> top10 for every test query under that fold's vector
    adapted_scores = {}
    for k in tqdm(range(5), desc="adapted full-corpus search", file=sys.stderr):
        alpha = fit_info[k]["alpha"]
        qa = F.normalize(q_all + alpha * torch.from_numpy(vectors[k]).unsqueeze(0), p=2, dim=-1)
        ids, sc = full_search(qa, f"fold{k} search")
        for i, q in enumerate(test_qids):
            adapted[(k, q)] = ids[i]; adapted_scores[(k, q)] = sc[i]

    # ------------------------------------------------------------- qrels: evaluation only
    if STOP_BEFORE_QRELS:
        print("DRY_RUN_STOP_BEFORE_QRELS", round(time.time() - t0, 2), flush=True); return
    t_fits_done = time.time()
    qrels = json.loads(QRELS_FILE.read_text())
    assert len(qrels) == N_TEST and set(qrels) == set(test_qids)
    grades = sorted({int(g) for r in qrels.values() for g in r.values()})
    print("QRELS_LOADED_AFTER_ALL_FITS", dict(queries=len(qrels), grades=grades), flush=True)

    raw_cached_all = np.array([ndcg10(cached10[i], qrels[q]) for i, q in enumerate(test_qids)])
    raw_null_all = np.array([ndcg10(null10[i], qrels[q]) for i, q in enumerate(test_qids)])
    prior_raw = float(json.loads(BASELINE_RESULTS.read_text())["results"]["raw_retriever_ndcg10"])
    assert abs(raw_cached_all.mean() - prior_raw) < 1e-12, "raw metric drift"

    per_query, fold_metrics, deltas_by_fold = [], [], []
    base50 = {q: set(cached50[i]) for i, q in enumerate(test_qids)}
    for k, f in enumerate(folds):
        rows = [qid_row[q] for q in f["heldout_query_ids"]]
        raw_c = np.array([raw_cached_all[i] for i in rows]); raw_n = np.array([raw_null_all[i] for i in rows])
        ad = np.array([ndcg10(adapted[(k, test_qids[i])], qrels[test_qids[i]]) for i in rows])
        dr = []  # REFERENCE ONLY: direct teacher rerank of held-out top-10 = teacher inference at held-out time
        for i in rows:
            q = test_qids[i]
            rr = sorted(test_rows[i]["candidates"][:10], key=lambda c: (-teacher_map[(q, str(c["doc_id"]))], int(c["rank"])))
            dr.append(ndcg10([str(c["doc_id"]) for c in rr], qrels[q]))
        dr = np.array(dr)
        tr_rows = [qid_row[q] for q in f["train_query_ids"]]
        tr_ad = np.array([ndcg10(adapted[(k, test_qids[i])], qrels[test_qids[i]]) for i in tr_rows])
        delta = ad - raw_n
        deltas_by_fold.append(delta)
        changed = sum(adapted[(k, test_qids[i])] != null10[i] for i in rows)
        outside50 = sum(1 for i in rows for d in adapted[(k, test_qids[i])] if d not in base50[test_qids[i]])
        fold_metrics.append(dict(
            fold=k, n_train=len(f["train_query_ids"]), n_heldout=len(rows),
            raw_ndcg10_heldout=float(raw_n.mean()), raw_cached_ndcg10_heldout=float(raw_c.mean()),
            adapted_ndcg10_heldout=float(ad.mean()), delta_ndcg10_heldout=float(delta.mean()),
            heldout_paired_bootstrap=paired_bootstrap(delta),
            wins=int((delta > 0).sum()), losses=int((delta < 0).sum()), ties=int((delta == 0).sum()),
            heldout_queries_with_changed_top10=int(changed),
            heldout_top10_docs_outside_cached_top50=int(outside50),
            REFERENCE_direct_teacher_rerank_ndcg10_heldout=float(dr.mean()),
            insample_train_raw_ndcg10=float(np.mean([raw_null_all[i] for i in tr_rows])),
            insample_train_adapted_ndcg10=float(tr_ad.mean()),
            vector_norm=fit_info[k]["vector_norm"], vector_sha256=fit_info[k]["vector_sha256"]))
        for j, i in enumerate(rows):
            q = test_qids[i]
            per_query.append(dict(query_id=q, fold=k, raw_ndcg10=float(raw_n[j]), raw_cached_ndcg10=float(raw_c[j]),
                                  adapted_ndcg10=float(ad[j]), delta=float(delta[j]),
                                  REFERENCE_direct_rerank_ndcg10=float(dr[j]),
                                  raw_top10=null10[i], adapted_top10=adapted[(k, q)],
                                  adapted_top10_scores=[float(s) for s in adapted_scores[(k, q)]]))
        print(f"FOLD{k} heldout n={len(rows)} raw={raw_n.mean():.6f} adapted={ad.mean():.6f} delta={delta.mean():+.6f}", flush=True)

    # Fold-0 regression vs prior 518/130 run.
    prior_res = json.loads((PRIOR / "RESULTS.json").read_text())
    prior_rank = {r["query_id"]: r for r in json.loads((PRIOR / "heldout_rankings.json").read_text())}
    f0 = fold_metrics[0]
    f0_rank_mism = [q for q in folds[0]["heldout_query_ids"] if adapted[(0, q)] != prior_rank[q]["adapted_top10"]]
    f0_raw_rank_mism = [q for q in folds[0]["heldout_query_ids"] if null10[qid_row[q]] != prior_rank[q]["raw_top10"]]
    regression = dict(
        prior_raw_heldout=prior_res["raw_ndcg10_heldout640"], prior_adapted_heldout=prior_res["adapted_ndcg10_heldout640"],
        prior_delta=prior_res["heldout_gain_ndcg10"], prior_bootstrap=prior_res["heldout_paired_bootstrap"],
        now_raw_cached_heldout=f0["raw_cached_ndcg10_heldout"], now_raw_cpu_null_heldout=f0["raw_ndcg10_heldout"],
        now_adapted_heldout=f0["adapted_ndcg10_heldout"], now_delta=f0["delta_ndcg10_heldout"],
        now_bootstrap=f0["heldout_paired_bootstrap"],
        vector_maxabs_diff=fold0_v_maxabs, adapted_top10_mismatches_vs_prior=len(f0_rank_mism),
        raw_top10_mismatches_vs_prior=len(f0_raw_rank_mism),
        abs_diff_adapted=abs(f0["adapted_ndcg10_heldout"] - prior_res["adapted_ndcg10_heldout640"]),
        abs_diff_delta=abs(f0["delta_ndcg10_heldout"] - prior_res["heldout_gain_ndcg10"]),
        tolerance="vector max|diff| <= 1e-6, identical adapted top-10 lists, |metric diff| <= 1e-12")
    regression["passed"] = bool(fold0_v_maxabs <= 1e-6 and not f0_rank_mism and regression["abs_diff_adapted"] <= 1e-12
                                and regression["abs_diff_delta"] <= 1e-12)
    print("FOLD0_REGRESSION", json.dumps({k: v for k, v in regression.items() if "bootstrap" not in k}), flush=True)
    if not regression["passed"]:
        (OUT / "FOLD0_REGRESSION_FAILED.json").write_text(json.dumps(regression, indent=2))
        raise SystemExit("STOP: fold 0 did not reproduce the prior 518/130 run; pooled results withheld.")

    all_delta = np.concatenate(deltas_by_fold)
    raw_pool = np.array([p["raw_ndcg10"] for p in per_query]); ad_pool = np.array([p["adapted_ndcg10"] for p in per_query])
    fd = np.array([m["delta_ndcg10_heldout"] for m in fold_metrics])
    pooled = dict(
        n_queries=len(per_query), unique_queries=len({p["query_id"] for p in per_query}),
        raw_ndcg10=float(raw_pool.mean()), crossfit_adapted_ndcg10=float(ad_pool.mean()),
        mean_paired_delta=float(all_delta.mean()),
        paired_query_bootstrap=paired_bootstrap(all_delta),
        fold_stratified_bootstrap=stratified_bootstrap(deltas_by_fold),
        wins=int((all_delta > 0).sum()), losses=int((all_delta < 0).sum()), ties=int((all_delta == 0).sum()),
        REFERENCE_direct_teacher_rerank_ndcg10=float(np.mean([p["REFERENCE_direct_rerank_ndcg10"] for p in per_query])),
        per_fold_delta=fd.tolist(), per_fold_delta_mean=float(fd.mean()),
        per_fold_delta_sd_ddof1=float(fd.std(ddof=1)), per_fold_delta_min=float(fd.min()),
        per_fold_delta_max=float(fd.max()), folds_positive=int((fd > 0).sum()), folds_negative=int((fd < 0).sum()))
    ci = pooled["paired_query_bootstrap"]
    pooled["ci_excludes_zero"] = bool(ci["ci95_low"] > 0 or ci["ci95_high"] < 0)
    V = np.stack(vectors); Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    vec_cos = (Vn @ Vn.T).tolist()
    print("POOLED", json.dumps({k: v for k, v in pooled.items()}), flush=True)

    # ------------------------------------------------------------- write outputs
    (OUT / "per_query_heldout.json").write_text(json.dumps(per_query, indent=1))
    with (OUT / "per_query_heldout.csv").open("w") as fh:
        fh.write("query_id,fold,raw_ndcg10,adapted_ndcg10,delta,raw_cached_ndcg10,REFERENCE_direct_rerank_ndcg10\n")
        for p in per_query:
            fh.write(f"{p['query_id']},{p['fold']},{p['raw_ndcg10']:.10f},{p['adapted_ndcg10']:.10f},{p['delta']:.10f},"
                     f"{p['raw_cached_ndcg10']:.10f},{p['REFERENCE_direct_rerank_ndcg10']:.10f}\n")
    (OUT / "fit_info.json").write_text(json.dumps(fit_info, indent=2))
    (OUT / "INPUT_CHECKSUMS.json").write_text(json.dumps(input_hashes, indent=2, sort_keys=True))
    results = dict(
        task_id=TASK_ID, task="FiQA2018", design="5-fold cross-fitted task-wise TTT-Embed; fold 0 = prior 518/130 split",
        counts=dict(folds=5, unique_test_queries=N_TEST, heldout_exactly_once=True,
                    fold_train_sizes=[m["n_train"] for m in fold_metrics],
                    fold_heldout_sizes=[m["n_heldout"] for m in fold_metrics],
                    corpus_documents_searched_per_query=N_DOCS, unique_cached_teacher_pairs=6480,
                    logical_teacher_pair_uses_in_fitting=int(sum(f["teacher_pairs_used"] for f in fit_info)),
                    teacher_pairs_per_fold=[f["teacher_pairs_used"] for f in fit_info],
                    new_embedding_calls=0, new_reranker_calls=0, downloads=0, gpu_or_modal_jobs=0, paid_api_calls=0),
        hyperparameters=dict(depth=DEPTH, teacher_temperature=TEACHER_TEMP, student_temperature=STUDENT_TEMP,
                             lambda_l2=LAMBDA, lr=LR, steps=STEPS, optimizer="Adam (torch defaults betas/eps)",
                             init="zeros", loss="forward KL(p_T||p_S) mean over queries + lambda*||v||^2",
                             train_query="normalize(x_q+v)", deploy_query="normalize(x_q+alpha*v), alpha=n/(n+1)"),
        seeds=dict(split_seed=SPLIT_SEED, fold_seed=FOLD_SEED, rng=RNG_IMPL, bootstrap_seed=BOOT_SEED,
                   bootstrap_reps=BOOT_REPS),
        primary_raw_arm="CPU full-corpus search with v=0 (same implementation as adapted arm)",
        identity_control={k: v for k, v in identity.items() if k != "mismatch_detail"},
        fold_proofs=fold_proofs, fold0_regression=regression, per_fold=fold_metrics, pooled=pooled,
        fold_vector_pairwise_cosine=vec_cos, fit=[{k: v for k, v in f.items() if k != "loss_trace"} for f in fit_info],
        qrels_loaded_after_all_fits=True, qrels_load_utc_offset_seconds=t_fits_done - t0,
        environment=env, input_hashes={Path(k).name if str(ROOT) in k else k: v["sha256"] for k, v in input_hashes.items()},
        interpretation_guardrails=[
            "Cross-fitted estimate over five fold-specific vectors, not one deployed vector.",
            "Single task (FiQA2018); not a reproduction of the paper's 15-task averages.",
            "REFERENCE_direct_teacher_rerank uses held-out teacher scores (teacher inference at held-out time); "
            "it is a context ceiling, not a matched-budget comparator.",
            "All 648 test queries were already exposed in earlier runs (in-sample baseline, 8q, 80/20); "
            "fold 0 is a re-run of a known result."],
        runtime_seconds=time.time() - t0)
    (OUT / "RESULTS.json").write_text(json.dumps(results, indent=2))
    print("RUN_SECONDS", round(time.time() - t0, 2), flush=True)
    print("DONE_PENDING_RUN_MD_AND_MANIFEST", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
