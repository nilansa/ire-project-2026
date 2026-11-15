from __future__ import annotations

import csv
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
import statistics
import sys
import time
import zipfile
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

import numpy as np

from common import DATA, MUSIQUE_DEV, RAW, ROOT, STAGE1_SAMPLE, read_jsonl, read_json, salted, sha256_file, write_json_new

ARMS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A7b", "A8"]
KS = [10, 20]
BOOTSTRAPS = 1000
N1_LIMIT = 0.05
N2_NEAR_RANDOM = 0.03


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qid_column(zf: zipfile.ZipFile, dataset: str, split: str) -> list[str]:
    names = [n for n in zf.namelist() if n.endswith(f"{dataset}/qrels/{split}.tsv")]
    if len(names) != 1:
        raise RuntimeError(f"cannot locate {dataset}/{split} judgments")
    out = {}
    with zf.open(names[0]) as f:
        first = next(f, None)
        lines = f if first is not None and first.lower().startswith(b"query-id\t") else chain([first], f) if first is not None else iter(())
        for raw in lines:
            q, d, score = raw.decode("utf-8").strip().split("\t")[:3]
            out.setdefault(q, {})[d] = int(float(score))
    return out


def norm(s: str) -> str:
    return " ".join(str(s).casefold().split())


def musique_labels(prepared_ids: list[str]) -> tuple[dict, dict, list[str], str | None]:
    with STAGE1_SAMPLE.open(encoding="utf-8") as f:
        excluded = {str(x["id"]) for x in json.load(f)}
    text_to_ids: dict[tuple[str, str], list[str]] = {}
    known_ids = set()
    with (DATA / "musique" / "corpus.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            known_ids.add(str(d["doc_id"]))
            text_to_ids.setdefault((norm(d.get("title", "")), norm(d.get("text", ""))), []).append(str(d["doc_id"]))
    actual: dict[str, list[str]] = {}
    ambiguous = []
    all_eligible = []
    with MUSIQUE_DEV.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row.get("id", ""))
            if not qid.startswith("2hop__") or qid in excluded:
                continue
            paras = row.get("paragraphs", [])
            chosen = [p for p in paras if p.get("is_supporting") is True]
            if not chosen:
                refs = {str(x.get("paragraph_support_idx")) for x in row.get("question_decomposition", []) if x.get("paragraph_support_idx") is not None}
                chosen = [p for p in paras if str(p.get("idx")) in refs]
            docs = []
            for p in chosen:
                direct = p.get("doc_id")
                if direct is not None:
                    if str(direct) in known_ids:
                        docs.append(str(direct))
                    continue
                key = (norm(p.get("title", "")), norm(p.get("paragraph_text", p.get("text", ""))))
                matches = text_to_ids.get(key, [])
                if len(matches) > 1:
                    ambiguous.append(qid)
                elif matches:
                    docs.append(matches[0])
            docs = sorted(set(docs))
            if len(docs) == 2:
                actual[qid] = docs
                all_eligible.append(qid)
    exact_ids = sorted(all_eligible, key=lambda q: (salted(q), q))[:100]
    if len(exact_ids) != 100:
        return {}, {}, exact_ids, f"only {len(exact_ids)} eligible ids found"
    if prepared_ids != exact_ids:
        return {}, {}, exact_ids, "retrieved MuSiQue ids do not equal the first 100 eligible ids"
    if set(ambiguous) & set(exact_ids):
        return {}, {}, exact_ids, "ambiguous corpus mapping for one or more selected support passages"
    supports = {q: actual[q] for q in exact_ids}
    qrels = {q: {d: 1 for d in actual[q]} for q in exact_ids}
    return qrels, supports, exact_ids, None


def nrel(qrel: dict[str, int]) -> dict[str, int]:
    return {d: int(v) for d, v in qrel.items() if int(v) > 0}


def ndcg(rel: dict[str, int], ranking: list[str], k: int) -> float:
    gains = [int(rel.get(d, 0)) for d in ranking[:k]]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(gains))
    ideal = sorted((int(v) for v in rel.values()), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def query_metrics(dataset: str, qid: str, arm: str, k: int, rec: dict, qrels: dict, supports: dict) -> dict:
    rel = nrel(qrels.get(qid, {}))
    ranking = rec["rankings"].get(arm, [])
    top = ranking[:k]
    hit = len(set(top) & set(rel))
    recall = hit / len(rel) if rel else 0.0
    complete = float(hit == len(rel)) if dataset in {"scifact", "nfcorpus"} and 2 <= len(rel) <= k else None
    support = supports.get(qid)
    all_support = float(set(support).issubset(top)) if support else None
    pool = rec["base_pool"]
    pool_set = set(pool)
    in_count = len(set(top) & set(rel) & pool_set)
    out_count = len((set(top) & set(rel)) - pool_set)
    pool_rank = {d: i + 1 for i, d in enumerate(pool)}
    promotions = sum(1 for d in set(top) & set(rel) if d in pool_rank and pool_rank[d] > k)
    pool_hit = len(set(pool) & set(rel))
    pool_recall = pool_hit / len(rel) if rel else 0.0
    pool_all = float(bool(support) and set(support).issubset(pool_set)) if support else None
    total_gold = in_count + out_count
    return {
        "dataset": dataset, "qid": qid, "arm": arm, "K": k,
        "recall": recall, "ndcg": ndcg(qrels.get(qid, {}), ranking, k),
        "all_support": all_support, "complete": complete,
        "in_pool_count": in_count, "out_of_pool_count": out_count,
        "out_of_pool_share": out_count / total_gold if total_gold else 0.0,
        "promotions": promotions, "pool_recall_200": pool_recall,
        "pool_all_support_200": pool_all,
        "latency_s": rec["latency_s"].get(arm),
        "llm_calls": rec["llm_calls"].get(arm, 0),
        "llm_tokens": rec["llm_tokens"].get(arm, 0),
        "colbert_doc_encodings": rec["doc_encodings"].get(arm, 0),
        "retrieval_calls": rec["retrieval_calls"].get(arm, 0),
    }


def mean(values):
    vals = [float(x) for x in values if x is not None]
    return sum(vals) / len(vals) if vals else None


def ci(diff: list[float], indices: np.ndarray) -> list[float]:
    vals = np.asarray(diff, dtype=np.float64)[indices].mean(axis=1)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def contrast(name: str, dataset: str, metric: str, k: int, a: str, b: str, rows: list[dict], boot_idx: np.ndarray) -> dict:
    by = {(r["qid"], r["arm"]): r for r in rows if r["dataset"] == dataset and r["K"] == k}
    qids = sorted({qid for qid, arm in by if arm == a and (qid, b) in by and by[(qid, a)][metric] is not None and by[(qid, b)][metric] is not None})
    diffs = [float(by[(q, a)][metric]) - float(by[(q, b)][metric]) for q in qids]
    if not diffs:
        return {"name": name, "dataset": dataset, "metric": metric, "K": k, "a": a, "b": b, "n": 0, "estimate": None, "ci95": None}
    return {"name": name, "dataset": dataset, "metric": metric, "K": k, "a": a, "b": b, "n": len(diffs), "estimate": mean(diffs), "ci95": ci(diffs, boot_idx)}


def result_means(rows: list[dict], datasets: list[str]) -> dict:
    out = {}
    for ds in datasets:
        out[ds] = {"n": len({r["qid"] for r in rows if r["dataset"] == ds}), "arms": {}}
        for arm in ARMS:
            out[ds]["arms"][arm] = {}
            for k in KS:
                subset = [r for r in rows if r["dataset"] == ds and r["arm"] == arm and r["K"] == k]
                out[ds]["arms"][arm][str(k)] = {
                    metric: mean([r[metric] for r in subset]) for metric in ["recall", "ndcg", "all_support", "complete", "in_pool_count", "out_of_pool_count", "out_of_pool_share", "promotions", "pool_recall_200", "pool_all_support_200", "latency_s", "llm_calls", "llm_tokens", "colbert_doc_encodings", "retrieval_calls"]
                }
                out[ds]["arms"][arm][str(k)]["complete_n"] = sum(r["complete"] is not None for r in subset)
    return out


def oracle_check(qrels: dict, supports: dict) -> dict:
    max_recall_error = 0.0
    max_ndcg_error = 0.0
    max_support_error = 0.0
    n = 0
    for qid, rel0 in qrels.items():
        rel = nrel(rel0)
        perfect = sorted(rel, key=lambda d: (-rel[d], d))
        recall = len(set(perfect[:len(rel)]) & set(rel)) / max(1, len(rel))
        max_recall_error = max(max_recall_error, abs(1.0 - recall))
        max_ndcg_error = max(max_ndcg_error, abs(1.0 - ndcg(rel0, perfect, len(rel))))
        if qid in supports:
            k = len(supports[qid])
            val = float(set(supports[qid]).issubset(perfect[:k]))
            max_support_error = max(max_support_error, abs(1.0 - val))
        n += 1
    ok = n > 0 and max(max_recall_error, max_ndcg_error, max_support_error) < 1e-12
    return {"status": "PASS" if ok else "FAIL", "expected": "oracle recall=1, nDCG=1, all-support=1 at matching K", "observed": {"queries": n, "max_recall_error": max_recall_error, "max_ndcg_error": max_ndcg_error, "max_all_support_error": max_support_error}}


def negative_one(records: list[dict], qrels: dict, supports: dict, datasets: list[str]) -> dict:
    rng = np.random.default_rng(0)
    observed = {}
    passed = True
    for ds in datasets:
        qids = sorted(qrels[ds])
        if len(qids) < 2:
            observed[ds] = {"status": "FAIL", "reason": "fewer than two queries"}; passed = False; continue
        order = np.arange(len(qids))
        while True:
            perm = rng.permutation(len(qids))
            if np.all(perm != order):
                break
        shuffled = {qids[i]: qrels[ds][qids[int(perm[i])]] for i in range(len(qids))}
        shuffled_support = {qids[i]: supports.get(ds, {}).get(qids[int(perm[i])], []) for i in range(len(qids))}
        max_recall, max_all = 0.0, 0.0
        recall_by_arm, all_by_arm = {}, {}
        for arm in ARMS:
            recalls, all_values = [], []
            for rec in retrieval_rows_by_ds[ds]:
                q = rec["qid"]
                rel = nrel(shuffled.get(q, {}))
                top = rec["rankings"].get(arm, [])[:20]
                score = len(set(top) & set(rel)) / len(rel) if rel else 0.0
                recalls.append(score)
                ss = shuffled_support.get(q, [])
                if ss:
                    all_values.append(float(set(ss).issubset(top)))
            recall_by_arm[arm] = mean(recalls) or 0.0
            all_by_arm[arm] = mean(all_values) or 0.0
            max_recall = max(max_recall, recall_by_arm[arm])
            max_all = max(max_all, all_by_arm[arm])
        ok = max_recall <= N1_LIMIT and (ds != "musique" or max_all <= N1_LIMIT)
        observed[ds] = {"status": "PASS" if ok else "FAIL", "max_recall_at_20": max_recall, "mean_recall_at_20_by_arm": recall_by_arm, "max_all_support_at_20": max_all if ds == "musique" else None, "mean_all_support_at_20_by_arm": all_by_arm if ds == "musique" else None, "tolerance": N1_LIMIT, "seed": 0}
        passed = passed and ok
    return {"status": "PASS" if passed else "FAIL", "expected": f"all arms <= {N1_LIMIT} after a seeded derangement of qrels", "observed": observed}


def negative_two(records: list[dict], supports: dict, blocked: bool) -> dict:
    if blocked or "musique" not in supports:
        return {"status": "FAIL", "expected": f"random Gaussian coverage below A5 and within {N2_NEAR_RANDOM} of random pool order", "observed": "MuSiQue sample unavailable"}
    rows = [r for r in records if r["dataset"] == "musique" and r["K"] == 10]
    actual, randcov, randorder = [], [], []
    for r in rows:
        q = r["qid"]; ss = set(supports["musique"][q])
        rec = retrieval_rows_by_ds["musique"][q]
        actual.append(float(ss.issubset(rec["rankings"]["A5"][:10])))
        randcov.append(float(ss.issubset(rec["random_a5"][:10])))
        randorder.append(float(ss.issubset(rec["random_order"][:10])))
    a, rc, ro = mean(actual), mean(randcov), mean(randorder)
    ok = rc < a and abs(rc - ro) <= N2_NEAR_RANDOM
    return {"status": "PASS" if ok else "FAIL", "expected": f"random coverage < A5 and absolute difference from random ordering <= {N2_NEAR_RANDOM}", "observed": {"A5_all_support_at_10": a, "random_gaussian_all_support_at_10": rc, "random_order_all_support_at_10": ro, "margin": N2_NEAR_RANDOM}}


def write_report(results: dict, contrasts: dict, controls: dict, blocked_reason: str | None, out_path: Path) -> None:
    def fmt(value):
        return "NA" if value is None else f"{value:.4f}"

    lines = ["# OOD_COVDECOMP_20261115 results", "", f"Run completed (UTC): {now()}", "", "## Dataset means (per-query averages)"]
    for ds, item in results["datasets"].items():
        if item.get("status") == "BLOCKED":
            lines.append(f"- {ds}: BLOCKED — {item['reason']}")
            continue
        lines.append(f"- {ds} (n={item['n']}; R=Recall, N=nDCG, AS=all-support):")
        packed = []
        for arm in ARMS:
            a10 = item["arms"][arm]["10"]
            a20 = item["arms"][arm]["20"]
            packed.append(f"{arm} R10={fmt(a10['recall'])} R20={fmt(a20['recall'])} N10={fmt(a10['ndcg'])} AS10={fmt(a10['all_support'])} AS20={fmt(a20['all_support'])}")
        for start in range(0, len(packed), 3):
            lines.append("  " + " | ".join(packed[start:start + 3]))
    lines += ["", "## Preregistered contrasts"]
    for name, c in contrasts.items():
        if c.get("estimate") is None:
            lines.append(f"- {name}: BLOCKED (n=0)")
        else:
            lo, hi = c["ci95"]
            lines.append(f"- {name}: delta={c['estimate']:.4f}, 95% paired bootstrap CI [{lo:.4f}, {hi:.4f}], n={c['n']}")
    lines += ["", "## Decision rules"]
    for k, v in results["decision_rules"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Controls"]
    for name, c in controls.items():
        lines.append(f"- {name}: {c.get('status', 'BLOCKED')} — expected={c.get('expected')} — observed={c.get('observed')}")
    lines += ["", "## Deviations and blockers", "- Checkpoint CHECKPOINT.json was absent; local checkpoint files were hashed directly.", "- Existing Stage-2 venv reused read-only; BM25 formula port used because rank_bm25 is unavailable; optional teacher skipped.", "- See DEVIATIONS.md for the failed initial smoke probe and the successful five-query format check.", "- Independent cold review remains pending; no scientific claim is certified by this run."]
    if blocked_reason:
        lines.append(f"- {blocked_reason}")
    if len(lines) > 60:
        raise AssertionError(f"REPORT.md must be <=60 lines, got {len(lines)}")
    with out_path.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    start = time.perf_counter()
    if not (ROOT / "retrieval" / "retrieval.jsonl").exists():
        raise FileNotFoundError("retrieval phase has not completed")
    hidden = ROOT / "hidden_labels"
    hidden.mkdir(exist_ok=False)
    retrieval_rows = list(read_jsonl(ROOT / "retrieval" / "retrieval.jsonl"))
    global retrieval_rows_by_ds
    retrieval_rows_by_ds = {ds: {r["qid"]: r for r in retrieval_rows if r["dataset"] == ds} for ds in ["scifact", "nfcorpus", "musique"]}
    prep = read_json(DATA / "prepare_manifest.json")
    qrels: dict[str, dict] = {}
    supports: dict[str, dict] = {}
    for ds in ["scifact", "nfcorpus"]:
        with zipfile.ZipFile(RAW / f"{ds}.zip") as zf:
            all_qrels = qid_column(zf, ds, "test")
        selected = set(prep["beir"][ds]["query_ids"])
        qrels[ds] = {q: all_qrels[q] for q in selected}
        supports[ds] = {}
    mu_qrels, mu_supports, exact_ids, blocked_reason = musique_labels(prep["musique"]["query_ids"])
    if blocked_reason is None:
        qrels["musique"], supports["musique"] = mu_qrels, mu_supports
    else:
        qrels["musique"], supports["musique"] = {}, {}
    write_json_new(hidden / "qrels.json", qrels)
    write_json_new(hidden / "supports.json", supports)

    usable = ["scifact", "nfcorpus"] + ([] if blocked_reason else ["musique"])
    qids_by_ds = {ds: sorted(qrels[ds], key=lambda q: (salted(q), q)) for ds in usable}
    rows = []
    for ds in usable:
        for qid in qids_by_ds[ds]:
            rec = retrieval_rows_by_ds[ds].get(qid)
            if rec is None:
                raise RuntimeError(f"retrieval output missing {ds}/{qid}")
            for arm in ARMS:
                for k in KS:
                    rows.append(query_metrics(ds, qid, arm, k, rec, qrels[ds], supports[ds]))

    csv_path = ROOT / "per_query.csv"
    fields = ["dataset", "qid", "arm", "K", "recall", "ndcg", "all_support", "complete", "in_pool_count", "out_of_pool_count", "out_of_pool_share", "promotions", "pool_recall_200", "pool_all_support_200", "latency_s", "llm_calls", "llm_tokens", "colbert_doc_encodings", "retrieval_calls"]
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    means = result_means(rows, ["scifact", "nfcorpus", "musique"])
    if blocked_reason:
        means["musique"] = {"status": "BLOCKED", "reason": blocked_reason, "candidate_queries_retrieved": len(retrieval_rows_by_ds["musique"])}
    boot_idx = {ds: np.random.default_rng(0).integers(0, len(qids_by_ds[ds]), size=(BOOTSTRAPS, len(qids_by_ds[ds]))) for ds in usable}
    contrasts = {}
    if "musique" in usable:
        contrasts["D1_M1"] = contrast("D1_M1", "musique", "all_support", 10, "A5", "A4", rows, boot_idx["musique"])
        contrasts["D2_M3"] = contrast("D2_M3", "musique", "all_support", 20, "A7", "A3", rows, boot_idx["musique"])
        contrasts["D4_M2_nDCG"] = contrast("D4_M2_nDCG", "musique", "ndcg", 10, "A4", "A3", rows, boot_idx["musique"])
        contrasts["D4_M2_all_support"] = contrast("D4_M2_all_support", "musique", "all_support", 10, "A4", "A3", rows, boot_idx["musique"])
    for ds in ["scifact", "nfcorpus"]:
        contrasts[f"D2_Yin_harm_{ds}"] = contrast(f"D2_Yin_harm_{ds}", ds, "ndcg", 10, "A7b", "A3", rows, boot_idx[ds])
        contrasts[f"D3_A5_nonreg_{ds}"] = contrast(f"D3_A5_nonreg_{ds}", ds, "ndcg", 10, "A5", "A4", rows, boot_idx[ds])
        contrasts[f"D3_A7_nonreg_{ds}"] = contrast(f"D3_A7_nonreg_{ds}", ds, "ndcg", 10, "A7", "A3", rows, boot_idx[ds])

    preflight = read_json(ROOT / "preflight_controls.json")
    control_dict = {x["name"]: x for x in preflight["controls"]}
    oracle_qrels = {f"{ds}::{q}": v for ds in usable for q, v in qrels[ds].items()}
    oracle_supports = {f"{ds}::{q}": v for ds in usable for q, v in supports[ds].items()}
    oracle = oracle_check(oracle_qrels, oracle_supports)
    n1 = negative_one(rows, qrels, supports, usable)
    n2 = negative_two(rows, supports, bool(blocked_reason))
    control_dict["oracle_metric"] = {"name": "oracle_metric", **oracle}
    control_dict["N1_shuffled_qrels"] = {"name": "N1_shuffled_qrels", **n1}
    control_dict["N2_random_embeddings"] = {"name": "N2_random_embeddings", **n2}
    controls = {x["name"]: {k: v for k, v in x.items() if k != "name"} for x in control_dict.values()}
    write_json_new(ROOT / "controls.json", controls)

    d1 = contrasts.get("D1_M1", {})
    d2 = contrasts.get("D2_M3", {})
    d1_fire = d1.get("estimate") is not None and d1["estimate"] >= 0.05 and d1["ci95"][0] > 0
    d2_fire = d2.get("estimate") is not None and d2["estimate"] >= 0.05 and d2["ci95"][0] > 0
    d3_items = [contrasts[f"D3_A{x}_nonreg_{ds}"] for ds in ["scifact", "nfcorpus"] for x in [5, 7]]
    d3_pass = all(c.get("estimate") is not None and c["estimate"] >= -0.02 for c in d3_items)
    yin_harm = any(c.get("estimate") is not None and c["estimate"] <= -0.02 and c["ci95"][1] < 0 for c in [contrasts[f"D2_Yin_harm_{ds}"] for ds in ["scifact", "nfcorpus"]])
    m2 = [contrasts.get("D4_M2_nDCG", {}).get("estimate"), contrasts.get("D4_M2_all_support", {}).get("estimate")]
    m2_positive = any(x is not None and x > 0 for x in m2)
    a7_mu20 = [r for r in rows if r["dataset"] == "musique" and r["arm"] == "A7" and r["K"] == 20]
    m3_den = sum(r["in_pool_count"] + r["out_of_pool_count"] for r in a7_mu20)
    m3_out_share = sum(r["out_of_pool_count"] for r in a7_mu20) / m3_den if m3_den else 0.0
    m3_discovery = m3_out_share > 0.05
    if blocked_reason:
        verdict = "PARK / REVISIT"
    elif (d1_fire or d2_fire) and d3_pass and all(c.get("status") == "PASS" for c in controls.values()):
        verdict = "CONTINUE"
    elif m2_positive or m3_discovery:
        verdict = "PIVOT"
    else:
        verdict = "STOP"
    decision_rules = {
        "D1_M1_supported": bool(d1_fire), "D2_M3_supported": bool(d2_fire), "D2_Yin_consistent_harm": bool(yin_harm),
        "D3_nonregression_pass": bool(d3_pass), "D4_M2_positive": bool(m2_positive),
        "D2_A7_out_of_pool_share_gt_5pct": bool(m3_discovery), "D2_A7_out_of_pool_share": m3_out_share,
        "verdict": verdict,
    }
    results = {"datasets": means, "contrasts": contrasts, "decision_rules": decision_rules,
               "bootstrap": {"resamples": BOOTSTRAPS, "seed": 0, "interval": "paired percentile 95%"},
               "controls": {k: v["status"] for k, v in controls.items()}, "completed_utc": now()}
    write_json_new(ROOT / "results.json", results)
    write_report(results, contrasts, controls, blocked_reason, ROOT / "REPORT.md")

    deviations = [
        "The checkpoint directory lacked CHECKPOINT.json; MANIFEST.json hashes model.safetensors and the actual config/tokenizer/artifact.metadata files.",
        "The existing Stage-2 environment was reused read-only, as allowed by the task; no packages were installed.",
        "The optional bge-reranker-v2-m3 teacher comparison was skipped; it is optional and capped at 25 minutes.",
        "The IRE_P5 adapter describes the prior hidden-label boundary. This branch follows the branch-specific authorization: public MuSiQue dev supports are read only during score.py; gold_HIDDEN.json was never accessed.",
        "The independent cold review is pending and must precede any scientific claim.",
    ]
    if blocked_reason:
        deviations.append("MuSiQue confirmatory scoring was blocked: " + blocked_reason + ".")
    with (ROOT / "DEVIATIONS.md").open("a", encoding="utf-8") as f:
        f.write("\n## Full run\n\n" + "\n".join(f"- {x}" for x in deviations) + "\n")
    write_manifest(results, usable, blocked_reason, start)
    print(json.dumps({"phase": "score_done", "pid": os.getpid(), "rows": len(rows), "verdict": verdict, "blocked_reason": blocked_reason, "wall_s": time.perf_counter() - start, "at": now()}), flush=True)


def write_manifest(results: dict, datasets: list[str], blocked_reason: str | None, started: float) -> None:
    prep = read_json(DATA / "prepare_manifest.json")
    smoke = read_json(ROOT / "smoke" / "run_summary.json")
    main_run = read_json(ROOT / "retrieval" / "run_summary.json")
    files = ["PREREG.md", "common.py", "prepare.py", "retrieve.py", "score.py", "controls.py", "run_phase.py", "run_full.sh", "preflight_controls.json", "data/prepare_manifest.json", "data/scifact/corpus.jsonl", "data/scifact/queries.jsonl", "data/nfcorpus/corpus.jsonl", "data/nfcorpus/queries.jsonl", "data/musique/corpus.jsonl", "data/musique/queries.jsonl", "data/smoke_scifact_train.jsonl", "data/raw/scifact.zip", "data/raw/nfcorpus.zip", "cache/scifact_tasb.npy", "cache/scifact_tasb.json", "cache/nfcorpus_tasb.npy", "cache/nfcorpus_tasb.json", "cache/musique_tasb.npy", "cache/musique_tasb.json", "smoke/retrieval.jsonl", "smoke/decompositions.jsonl", "smoke/run_summary.json", "retrieval/retrieval.jsonl", "retrieval/decompositions.jsonl", "retrieval/run_summary.json", "per_query.csv", "results.json", "controls.json", "decompositions.jsonl", "REPORT.md", "DEVIATIONS.md", "hidden_labels/qrels.json", "hidden_labels/supports.json", "stage1_musique/corpus.jsonl", "stage1_musique/data/musique_ans_v1.0_dev.jsonl"]
    hashes = {name: sha256_file(ROOT / name) for name in files if (ROOT / name).is_file()}
    checkpoint_files = {p.name: sha256_file(p) for p in sorted(__import__("common").COLBERT_PATH.iterdir()) if p.is_file()}
    hf_snapshot = ROOT / "hf_cache" / "hub" / "models--sentence-transformers--msmarco-distilbert-base-tas-b" / "snapshots" / __import__("common").MODEL_REVISION
    dense_model_files = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(hf_snapshot.rglob("*")) if p.is_file()} if hf_snapshot.exists() else {}
    versions = {}
    for name in ["torch", "transformers", "sentence-transformers", "colbert-ai", "numpy", "huggingface-hub"]:
        try: versions[name] = importlib_metadata.version(name)
        except Exception: versions[name] = None
    with __import__("common").STAGE1_SAMPLE.open(encoding="utf-8") as f:
        stage1_ids = sorted(str(x["id"]) for x in json.load(f))
    input_hashes = {
        "stage1_corpus": sha256_file(__import__("common").MUSIQUE_CORPUS),
        "stage1_sample_ids_only": hashlib.sha256("\n".join(stage1_ids).encode()).hexdigest(),
        "musique_dev": sha256_file(__import__("common").MUSIQUE_DEV),
    }
    manifest = {
        "task": "IRE P5 OOD_COVDECOMP_20261115", "started_utc": prep["started_utc"], "completed_utc": now(),
        "parent_pid": os.getppid(), "score_pid": os.getpid(), "smoke": smoke, "main_run": main_run,
        "datasets_scored": datasets, "musique_blocked_reason": blocked_reason,
        "model": {"dense": "sentence-transformers/msmarco-distilbert-base-tas-b", "revision": __import__("common").MODEL_REVISION, "dense_model_files_sha256": dense_model_files, "colbert_path": str(__import__("common").COLBERT_PATH), "colbert_files_sha256": checkpoint_files, "decomposition": main_run["ollama_model"], "ollama_digest": main_run["ollama_digest"], "ollama_version": main_run["ollama_version"]},
        "llm_configuration": {"temperature": 0, "seed": 0, "think": False, "system_prompt_sha256": next(iter(read_jsonl(ROOT / "decompositions.jsonl")))["prompt_sha256"]},
        "versions": versions, "python": sys.version, "platform": __import__("platform").platform(),
        "authorized_input_sha256": input_hashes,
        "seeds": {"sampling_salt": __import__("common").SALT, "llm": 0, "bootstrap": 0, "random_embeddings": "sha256(p5-random:qid)"},
        "wall_seconds_score_phase": time.perf_counter() - started, "artifacts_sha256": hashes,
        "privacy_boundary": "gold_HIDDEN.json was never opened, read, statted, or enumerated; labels are under hidden_labels and are produced by score.py after retrieval.",
        "decision_rules": results["decision_rules"],
    }
    write_json_new(ROOT / "MANIFEST.json", manifest)


if __name__ == "__main__":
    main()

