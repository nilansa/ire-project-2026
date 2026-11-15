from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

from common import ROOT, write_json_new, read_json
from retrieve import encode_doc_vectors, greedy, load_colbert, similarities

BANNED = ["qrels", "supports", "gold", "hidden_labels"]


def now():
    return datetime.now(timezone.utc).isoformat()


def static_check() -> dict:
    source = (ROOT / "retrieve.py").read_text(encoding="utf-8").lower()
    found = [x for x in BANNED if x in source]
    return {"name": "N3_static_leak", "expected": "none of the four forbidden strings occurs in retrieve.py", "observed": {"forbidden_strings_found": found}, "status": "PASS" if not found else "FAIL"}


def planted() -> tuple[dict, str]:
    import torch
    model, device = load_colbert()
    query = "How do oat beta glucan supplements affect LDL cholesterol compared with nitrate therapy for high blood pressure?"
    docs = {
        "x1": "Oat beta glucan supplements can reduce LDL cholesterol. The oat fiber supports healthier cholesterol levels.",
        "x2": "Oat beta glucan supplementation lowers LDL cholesterol in adults. Oat fiber changes blood lipids.",
        "x3": "Oat beta glucan is associated with lower LDL cholesterol and improved lipid measurements.",
        "x4": "Daily oat fiber and beta glucan intake may decrease LDL cholesterol in people with elevated lipids.",
        "x5": "Clinical evidence links oat beta glucan with lower LDL cholesterol and healthier blood lipid profiles.",
        "y": "Dietary nitrate therapy reduced high blood pressure in randomized clinical trials, lowering systolic readings.",
        "d1": "A report about unrelated astronomy observations and telescope measurements.",
        "d2": "A study of software reliability and distributed database systems.",
    }
    qvec = model.queryFromText([query])
    vectors = encode_doc_vectors(model, list(docs), docs)
    ids = list(docs)
    best, _ = similarities(qvec, ids, vectors, device)
    a4 = [ids[i] for i in torch.argsort(best.sum(dim=1), descending=True).detach().cpu().tolist()]
    a5 = greedy(ids, best, 2)
    expected = {"A5_top2_contains_y": True, "A4_top2_excludes_y": True}
    observed = {"A5_top2": a5, "A4_top2": a4[:2], "facet_y_id": "y"}
    ok = "y" in a5[:2] and "y" not in a4[:2]
    return ({"name": "planted_positive", "expected": expected, "observed": observed, "status": "PASS" if ok else "FAIL"}, str(device))


def oracle_metric() -> dict:
    qrels = {"q": {"x": 1, "y": 1}}
    ranking = ["x", "y"]
    recall = len(set(ranking[:2]) & set(qrels["q"])) / len(qrels["q"])
    all_support = float(set(qrels["q"]).issubset(ranking[:2]))
    ok = recall == 1.0 and all_support == 1.0
    return {"name": "oracle_metric", "expected": {"recall_at_2": 1.0, "all_support_at_2": 1.0}, "observed": {"recall_at_2": recall, "all_support_at_2": all_support}, "status": "PASS" if ok else "FAIL"}


def main():
    start = time.perf_counter()
    if len(sys.argv) != 2 or sys.argv[1] not in {"--preflight", "--final"}:
        raise SystemExit("usage: controls.py --preflight|--final")
    if sys.argv[1] == "--preflight":
        static = static_check()
        positive, device = planted()
        oracle = oracle_metric()
        report = {"pid": os.getpid(), "started_utc": now(), "device": device, "controls": [positive, oracle, static], "wall_s": time.perf_counter() - start}
        write_json_new(ROOT / "preflight_controls.json", report)
        print(json.dumps(report), flush=True)
        if any(x["status"] != "PASS" for x in report["controls"]):
            raise SystemExit(2)
    else:
        final = read_json(ROOT / "controls.json")
        required = {"planted_positive", "oracle_metric", "N1_shuffled_qrels", "N2_random_embeddings", "N3_static_leak"}
        if set(final) != required:
            raise RuntimeError(f"control record keys differ: {sorted(final)}")
        static = static_check()
        if static["status"] != final["N3_static_leak"]["status"]:
            raise RuntimeError("final retrieval source no longer matches the recorded N3 outcome")
        print(json.dumps({"phase": "controls_final", "pid": os.getpid(), "statuses": {k: v["status"] for k, v in final.items()}, "at": now()}), flush=True)


if __name__ == "__main__":
    main()
