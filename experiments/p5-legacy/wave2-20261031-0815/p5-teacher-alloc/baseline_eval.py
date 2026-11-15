"""Offline MiniLM baseline on frozen DEV2; never reads SEQ_HOLDOUT metrics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


OUT = Path(__file__).resolve().parent
PILOT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot")
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
VALIDATION_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"


def norm(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def eligible(row: dict) -> bool:
    try:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        gold = {norm(title) for title in row["supporting_facts"]["title"]}
        return (
            row["type"] in {"bridge", "comparison"}
            and len(titles) == 10
            and len(sentences) == 10
            and len(gold) == 2
            and sum(norm(title) in gold for title in titles) == 2
        )
    except (KeyError, TypeError):
        return False


def encode(model, tokenizer, texts: list[str], device, batch_size=128) -> np.ndarray:
    vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start : start + batch_size], padding=True,
                              truncation=True, max_length=256, return_tensors="pt").to(device)
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(F.normalize(pooled, p=2, dim=1).numpy())
    return np.concatenate(vectors)


def main() -> None:
    started = time.monotonic()
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.set_num_threads(16)
    device = torch.device("cpu")
    pilot = json.loads((PILOT / "results.json").read_text())
    split = json.loads((OUT / "split_manifest.json").read_text())
    pilot_ids = {str(row["qid"]) for row in pilot["query_info"]}
    holdout_ids = set((OUT / "seq_holdout_ids.txt").read_text().splitlines())
    if len(pilot_ids) != 600 or len(holdout_ids) != 2000 or pilot_ids & holdout_ids:
        raise ValueError("Frozen exclusion lists are invalid")

    validation = PILOT / "data" / "hotpot_validation.parquet"
    validation_hash = hashlib.sha256(validation.read_bytes()).hexdigest()
    if validation_hash != VALIDATION_SHA256:
        raise ValueError("Validation parquet hash mismatch")
    rows = pq.read_table(validation).to_pylist()
    eligible_rows = [row for row in rows if eligible(row)]
    dev_rows = [row for row in eligible_rows
                if str(row["id"]) not in pilot_ids and str(row["id"]) not in holdout_ids]
    dev_rows.sort(key=lambda row: (hashlib.sha256(str(row["id"]).encode()).hexdigest(), str(row["id"])))
    dev_ids = [str(row["id"]) for row in dev_rows]
    dev_hash = hashlib.sha256(("\n".join(dev_ids) + "\n").encode()).hexdigest()
    if len(eligible_rows) != 7342 or len(dev_rows) != 4742 or dev_hash != split["dev2_ids_sha256"]:
        raise ValueError("DEV2 does not match the frozen 4742-question split")
    items = []
    for row in dev_rows:
        qid = str(row["id"])
        titles = [str(title) for title in row["context"]["title"]]
        paragraphs = [
            (title.strip() + " " + " ".join(str(sentence).strip() for sentence in sentences)).strip()
            for title, sentences in zip(titles, row["context"]["sentences"])
        ]
        gold_titles = {norm(title) for title in row["supporting_facts"]["title"]}
        gold = {i for i, title in enumerate(titles) if norm(title) in gold_titles}
        if len(gold) != 2:
            raise ValueError(f"Expected two supporting paragraphs for {qid}")
        items.append({"qid": qid, "type": row["type"], "question": row["question"],
                      "paragraphs": paragraphs, "gold": gold})

    meta = pilot["model"]
    if meta["id"] != MODEL_ID or meta["revision"] != MODEL_REVISION:
        raise ValueError("Pilot cache model identity mismatch")
    for entry in meta["files"]:
        path = PILOT / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or changed cached model file: {entry['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Cached model hash mismatch: {entry['path']}")
    weight = next(f for f in meta["files"] if f["path"].endswith("model.safetensors"))
    model_path = (PILOT / weight["path"]).parent
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)

    metrics_path = OUT / "metrics.jsonl"
    summary_path = OUT / "run_summary.json"
    for path in (metrics_path, summary_path, OUT / "RESULT.md"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    outcomes = []
    for start in range(0, len(items), 64):
        batch_items = items[start : start + 64]
        passages = [p for item in batch_items for p in item["paragraphs"]]
        pvec = encode(model, tokenizer, passages, device).reshape(len(batch_items), 10, -1)
        qvec = encode(model, tokenizer, [item["question"] for item in batch_items], device)
        first = np.einsum("bpd,bd->bp", pvec, qvec).argmax(axis=1)
        conditional = [f"Question: {item['question']}\nEvidence so far: {item['paragraphs'][int(i)]}\nQuestion: {item['question']}"
                       for item, i in zip(batch_items, first)]
        q2vec = encode(model, tokenizer, conditional, device, batch_size=64)
        second_scores = np.einsum("bpd,bd->bp", pvec, q2vec)
        for i, first_choice in enumerate(first):
            second_scores[i, int(first_choice)] = -np.inf
        second = second_scores.argmax(axis=1)
        for item, one, two in zip(batch_items, first, second):
            outcomes.append({"qid": item["qid"], "question_type": item["type"],
                             "full_support_at_2": {int(one), int(two)} == item["gold"]})

    bridge = [row for row in outcomes if row["question_type"] == "bridge"]
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(bridge), size=(1000, len(bridge)))
    bridge_values = np.asarray([row["full_support_at_2"] for row in bridge], dtype=np.float32)
    bridge_ci = [float(x) for x in np.quantile(bridge_values[draws].mean(axis=1), [0.025, 0.975])]
    summary = {
        "status": "baseline_only",
        "compute": {"device": "CPU only", "threads": torch.get_num_threads(),
                    "wall_seconds": time.monotonic() - started, "cost_usd": 0.0},
        "data": {"validation_sha256": validation_hash, "eligible_n": len(eligible_rows),
                 "pilot_excluded_n": len(pilot_ids), "seq_holdout_n": len(holdout_ids),
                 "seq_holdout_sha256": split["seq_holdout_ids_file_sha256"],
                 "dev2_n": len(items), "dev2_ids_sha256": dev_hash,
                 "bridge_n": len(bridge)},
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "files": meta["files"]},
        "retrieval": "untrained dense first hop then conditional dense second hop; exact two-paragraph support",
        "metrics": {"overall_fs2": float(np.mean([row["full_support_at_2"] for row in outcomes])),
                    "bridge_fs2": float(bridge_values.mean()), "bridge_bootstrap_ci95": bridge_ci,
                    "comparison_fs2": float(np.mean([row["full_support_at_2"] for row in outcomes
                                                       if row["question_type"] == "comparison"]))},
        "teacher_allocation": "not run; HotpotQA train data was unavailable after network DNS failure",
        "experiment_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "libraries": {name: importlib.metadata.version(name)
                      for name in ["torch", "transformers", "pyarrow", "numpy"]},
    }
    with metrics_path.open("x") as stream:
        for row in outcomes:
            stream.write(json.dumps({"kind": "baseline_query", **row}, separators=(",", ":")) + "\n")
        stream.write(json.dumps({"kind": "baseline_summary", **summary}, separators=(",", ":")) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
