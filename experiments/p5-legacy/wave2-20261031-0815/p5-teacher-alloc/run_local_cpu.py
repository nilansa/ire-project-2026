"""Reduced-scale CPU fallback; keeps the frozen full DEV2 evaluation split."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


OUT = Path(__file__).resolve().parent
PILOT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot")
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
TRAIN_TOTAL_ROWS = 90447
VALIDATION_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
STUDENT_ID = "sentence-transformers/all-MiniLM-L6-v2"
STUDENT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TEACHER_ID = "local cross-encoder initialized from the cached MiniLM checkpoint"
TEACHER_REVISION = "HotpotQA train-only fine-tune, seed 20261031"
SEED = 20261031
BUDGETS = [("5", 0.05), ("20", 0.20), ("100", 1.00)]
POLICIES = ["random", "student_uncertainty", "bridge_focused"]
MAX_LEN = 192
TRAIN_QUESTIONS = 1000
TRAIN_STEPS = 60
BATCH_SIZE = 16
TEACHER_STEPS = 120
TEACHER_BATCH_SIZE = 32
MAX_SECONDS = 1300
DEVICE = torch.device("cpu")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_title(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def eligible(row: dict) -> bool:
    try:
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        gold = {normalize_title(title) for title in row["supporting_facts"]["title"]}
        return (
            row["type"] in {"bridge", "comparison"}
            and len(titles) == 10
            and len(sentences) == 10
            and len(gold) == 2
            and sum(normalize_title(title) in gold for title in titles) == 2
        )
    except (KeyError, TypeError):
        return False


def prepare(row: dict) -> dict:
    qid = str(row["id"])
    titles = [str(title) for title in row["context"]["title"]]
    paragraphs = [
        (title.strip() + " " + " ".join(str(sentence).strip() for sentence in sentences)).strip()
        for title, sentences in zip(titles, row["context"]["sentences"])
    ]
    gold_titles = list(dict.fromkeys(normalize_title(t) for t in row["supporting_facts"]["title"]))
    if len(gold_titles) != 2:
        raise ValueError(f"Expected two support titles for {qid}")
    gold_order = []
    for title in gold_titles:
        matches = [i for i, candidate in enumerate(titles) if normalize_title(candidate) == title]
        if len(matches) != 1:
            raise ValueError(f"Gold title does not map to one paragraph for {qid}")
        gold_order.append(matches[0])
    return {"qid": qid, "type": str(row["type"]), "question": str(row["question"]),
            "paragraphs": paragraphs, "gold_order": gold_order, "gold_set": set(gold_order)}


def fetch_train_sample() -> tuple[list[dict], dict]:
    from concurrent.futures import ThreadPoolExecutor

    offsets = [int(round(i * (TRAIN_TOTAL_ROWS - 100) / 20)) for i in range(21)]

    def fetch(offset: int) -> tuple[int, list[dict], dict]:
        params = urllib.parse.urlencode({
            "dataset": "hotpotqa/hotpot_qa", "config": "distractor",
            "split": "train", "offset": offset, "length": 100,
        })
        request = urllib.request.Request(
            ROWS_ENDPOINT + "?" + params, headers={"User-Agent": "ire-p5-experiment/1"}
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    body = response.read()
                data = json.loads(body)
                if data.get("num_rows_total") != TRAIN_TOTAL_ROWS:
                    raise ValueError("HotpotQA train row count changed")
                page = data.get("rows", [])
                complete = [entry["row"] for entry in page if not entry.get("truncated_cells")]
                return offset, complete, {
                    "offset": offset, "rows": len(page), "complete_rows": len(complete),
                    "truncated_rows": len(page) - len(complete), "response_bytes": len(body),
                    "response_sha256": sha(body),
                }
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched = sorted(pool.map(fetch, offsets), key=lambda result: result[0])
    rows = [row for _, page, _ in fetched for row in page]
    unique = {str(row["id"]): row for row in rows}
    eligible_rows = [row for row in unique.values() if eligible(row)]
    eligible_rows.sort(key=lambda row: (hashlib.sha256(str(row["id"]).encode()).hexdigest(), str(row["id"])))
    if len(eligible_rows) < TRAIN_QUESTIONS:
        raise ValueError(f"Only {len(eligible_rows)} eligible rows in the train sample")
    info = {
        "source": "HotpotQA distractor train via Hugging Face dataset viewer rows API",
        "endpoint": ROWS_ENDPOINT,
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "train",
        "split_rows_total": TRAIN_TOTAL_ROWS,
        "offset_sampling": "21 evenly spaced 100-row slices; rows with viewer-truncated cells are excluded",
        "pages": [meta for _, _, meta in fetched],
        "rows_retrieved": sum(meta["rows"] for _, _, meta in fetched),
        "complete_rows_retrieved": len(rows),
        "unique_rows_retrieved": len(unique),
        "eligible_rows_retrieved": len(eligible_rows),
        "selected_questions": TRAIN_QUESTIONS,
        "selected_ids_sha256": sha(("\n".join(str(row["id"]) for row in eligible_rows[:TRAIN_QUESTIONS]) + "\n").encode()),
    }
    return eligible_rows[:TRAIN_QUESTIONS], info


def query_text(question: str, evidence: str | None = None) -> str:
    raw = question if evidence is None else f"Question: {question}\nEvidence so far: {evidence}\nQuestion: {question}"
    return raw


def encode_texts(model, tokenizer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    vectors = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start : start + batch_size], padding=True, truncation=True,
                              max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
            hidden = model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            vectors.append(F.normalize(pooled, p=2, dim=1).numpy())
    return np.concatenate(vectors)


def main() -> None:
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    torch.set_num_threads(16)
    torch.set_num_interop_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if DEVICE.type != "cpu" or torch.cuda.is_initialized():
        raise RuntimeError("Fallback must run on CPU only; MPS and CUDA are not allowed")
    for name in ("metrics.jsonl", "run_summary.json", "RESULT.md"):
        if (OUT / name).exists():
            raise FileExistsError(f"Refusing to overwrite {OUT / name}")

    pilot_ids_text = (OUT / "pilot_excluded_ids.txt").read_text()
    holdout_text = (OUT / "seq_holdout_ids.txt").read_text()
    split = json.loads((OUT / "split_manifest.json").read_text())
    pilot_ids, holdout_ids = set(pilot_ids_text.splitlines()), set(holdout_text.splitlines())
    if len(pilot_ids) != 600 or len(holdout_ids) != 2000 or pilot_ids & holdout_ids:
        raise ValueError("Frozen exclusion lists are invalid")
    if sha(holdout_text.encode()) != split["seq_holdout_ids_file_sha256"]:
        raise ValueError("Holdout file hash differs from split manifest")

    validation_path = PILOT / "data" / "hotpot_validation.parquet"
    validation_bytes_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    if validation_bytes_hash != VALIDATION_SHA256:
        raise ValueError("Validation parquet hash differs from the frozen pilot source")
    validation_rows = pq.read_table(validation_path).to_pylist()
    validation_eligible = [row for row in validation_rows if eligible(row)]
    dev_rows = [row for row in validation_eligible
                if str(row["id"]) not in pilot_ids and str(row["id"]) not in holdout_ids]
    dev_rows.sort(key=lambda row: (hashlib.sha256(str(row["id"]).encode()).hexdigest(), str(row["id"])))
    dev_ids = [str(row["id"]) for row in dev_rows]
    if len(validation_eligible) != 7342 or len(dev_rows) != 4742:
        raise ValueError("Validation or DEV2 split count mismatch")
    if sha(("\n".join(dev_ids) + "\n").encode()) != split["dev2_ids_sha256"]:
        raise ValueError("DEV2 IDs differ from the frozen split hash")
    dev_items = [prepare(row) for row in dev_rows]
    del validation_rows, validation_eligible, dev_rows

    with tempfile.TemporaryDirectory(prefix="ire-p5-cpu-", dir=OUT) as temp:
        work = Path(temp)
        print("Fetching stratified HotpotQA train rows", flush=True)
        sampled_train_rows, train_source = fetch_train_sample()
        train_items = [prepare(row) for row in sampled_train_rows]
        if len(train_items) != TRAIN_QUESTIONS:
            raise ValueError(f"Expected {TRAIN_QUESTIONS} train questions, got {len(train_items)}")
        train_eligible_n = train_source["eligible_rows_retrieved"]
        print(f"Loaded {len(train_items)} train questions; pair pool will contain {len(train_items) * 20}", flush=True)

        pilot_results = json.loads((PILOT / "results.json").read_text())
        student_meta = pilot_results["model"]
        if student_meta["id"] != STUDENT_ID or student_meta["revision"] != STUDENT_REVISION:
            raise ValueError("Pilot cache model identity differs from the pinned MiniLM checkpoint")
        student_files = student_meta["files"]
        for entry in student_files:
            path = PILOT / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                raise ValueError(f"Cached student file is missing or changed: {entry['path']}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Cached student file hash mismatch: {entry['path']}")
        weight = next(entry for entry in student_files if entry["path"].endswith("model.safetensors"))
        student_path = (PILOT / weight["path"]).parent
        student_tokenizer = AutoTokenizer.from_pretrained(str(student_path), local_files_only=True)
        base_student = AutoModel.from_pretrained(str(student_path), local_files_only=True).to(DEVICE)
        groups, pairs, pairs_by_group = [], [], []
        for item_index, item in enumerate(train_items):
            first = item["paragraphs"][item["gold_order"][0]]
            for hop, query in enumerate((query_text(item["question"]), query_text(item["question"], first))):
                group_index = len(groups)
                groups.append({"item": item_index, "hop": hop, "query": query, "type": item["type"]})
                group_ids = []
                for candidate, passage in enumerate(item["paragraphs"]):
                    pair_index = len(pairs)
                    pairs.append({"group": group_index, "item": item_index, "candidate": candidate,
                                  "label": int(candidate == item["gold_order"][hop]),
                                  "bridge_second": item["type"] == "bridge" and hop == 1,
                                  "pair_id": f"{item['qid']}|h{hop + 1}|p{candidate}"})
                    group_ids.append(pair_index)
                pairs_by_group.append(group_ids)

        q_vec = encode_texts(base_student, student_tokenizer, [g["query"] for g in groups], batch_size=64)
        p_texts = [p for item in train_items for p in item["paragraphs"]]
        p_vec = encode_texts(base_student, student_tokenizer, p_texts, batch_size=64)
        del base_student

        uncertainty = np.empty(len(pairs), dtype=np.float32)
        for group_index, pair_ids in enumerate(pairs_by_group):
            item_index = groups[group_index]["item"]
            scores = p_vec[item_index * 10 : item_index * 10 + 10] @ q_vec[group_index]
            boundary = float((np.partition(scores, -2)[-2:].sum()) / 2)
            for candidate, pair_index in enumerate(pair_ids):
                uncertainty[pair_index] = -abs(float(scores[candidate]) - boundary)

        total_pairs = len(pairs)
        targets = {budget: max(1, round(total_pairs * fraction)) for budget, fraction in BUDGETS}
        random_order = np.random.default_rng(SEED).permutation(total_pairs).tolist()
        uncertain_order = np.argsort(-uncertainty, kind="stable").tolist()
        bridge_pool = [i for i, pair in enumerate(pairs) if pair["bridge_second"]]
        bridge_order = np.random.default_rng(SEED + 1).permutation(bridge_pool).tolist()
        selected = {}
        for budget, _ in BUDGETS:
            count = targets[budget]
            selected[(budget, "random")] = random_order[:count]
            selected[(budget, "student_uncertainty")] = uncertain_order[:count]
            selected[(budget, "bridge_focused")] = bridge_order[:min(count, len(bridge_order))]

        positives = [i for i, pair in enumerate(pairs) if pair["label"] == 1]
        base_training_ids = list(positives)
        for ids in pairs_by_group:
            negatives = [i for i in ids if pairs[i]["label"] == 0]
            negatives.sort(key=lambda i: hashlib.sha256(
                f"{SEED}|negative|{pairs[i]['pair_id']}".encode()
            ).digest())
            base_training_ids.extend(negatives[:2])

        teacher = AutoModelForSequenceClassification.from_pretrained(
            str(student_path), num_labels=1, ignore_mismatched_sizes=True, local_files_only=True
        ).to(DEVICE)
        teacher.train()
        teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-5)
        teacher_train_ids = np.random.default_rng(SEED + 2).choice(
            np.asarray(base_training_ids), size=TEACHER_STEPS * TEACHER_BATCH_SIZE, replace=True
        )
        teacher_train_started = time.monotonic()
        for step in range(TEACHER_STEPS):
            if time.monotonic() > deadline - 120:
                raise TimeoutError("CPU teacher training reached the runtime safety guard")
            ids = teacher_train_ids[
                step * TEACHER_BATCH_SIZE : (step + 1) * TEACHER_BATCH_SIZE
            ].tolist()
            queries = [groups[pairs[i]["group"]]["query"] for i in ids]
            passages = [train_items[pairs[i]["item"]]["paragraphs"][pairs[i]["candidate"]] for i in ids]
            batch = student_tokenizer(queries, passages, padding=True, truncation=True,
                                     max_length=MAX_LEN, return_tensors="pt")
            teacher_optimizer.zero_grad(set_to_none=True)
            logits = teacher(**batch).logits.reshape(-1).float()
            targets_batch = torch.tensor([pairs[i]["label"] for i in ids], dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(logits, targets_batch)
            loss.backward()
            teacher_optimizer.step()
        teacher_train_seconds = time.monotonic() - teacher_train_started
        print(f"Trained conditional CE teacher for {TEACHER_STEPS} steps", flush=True)
        teacher.eval()

        teacher_needed = sorted({i for ids in selected.values() for i in ids})
        teacher_scores = np.full(total_pairs, np.nan, dtype=np.float32)
        teacher_started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(teacher_needed), 32):
                if time.monotonic() > deadline - 120:
                    raise TimeoutError("CPU teacher scoring reached the runtime safety guard")
                ids = teacher_needed[start : start + 32]
                queries = [groups[pairs[i]["group"]]["query"] for i in ids]
                passages = [train_items[pairs[i]["item"]]["paragraphs"][pairs[i]["candidate"]] for i in ids]
                batch = student_tokenizer(queries, passages, padding=True, truncation=True,
                                          max_length=MAX_LEN, return_tensors="pt")
                teacher_scores[ids] = teacher(**batch).logits.reshape(-1).float().numpy()
        teacher_seconds = time.monotonic() - teacher_started
        print(f"Teacher-scored {len(teacher_needed)} selected candidate pairs", flush=True)
        del teacher, teacher_optimizer

        def evaluate(model):
            outcomes = []
            model.eval()
            for start in range(0, len(dev_items), 64):
                if time.monotonic() > deadline - 120:
                    raise TimeoutError("CPU DEV2 evaluation reached the runtime safety guard")
                batch_items = dev_items[start : start + 64]
                passages = [p for item in batch_items for p in item["paragraphs"]]
                pv = encode_texts(model, student_tokenizer, passages, batch_size=128).reshape(len(batch_items), 10, -1)
                q1 = [query_text(item["question"]) for item in batch_items]
                qv = encode_texts(model, student_tokenizer, q1, batch_size=64)
                first_scores = np.einsum("bpd,bd->bp", pv, qv)
                first = first_scores.argmax(axis=1)
                q2 = [query_text(item["question"], item["paragraphs"][int(chosen)])
                      for item, chosen in zip(batch_items, first)]
                q2v = encode_texts(model, student_tokenizer, q2, batch_size=64)
                second_scores = np.einsum("bpd,bd->bp", pv, q2v)
                for i, chosen in enumerate(first):
                    second_scores[i, int(chosen)] = -np.inf
                second = second_scores.argmax(axis=1)
                for item, one, two in zip(batch_items, first, second):
                    outcomes.append({"qid": item["qid"], "question_type": item["type"],
                                     "full_support_at_2": {int(one), int(two)} == item["gold_set"]})
            return outcomes

        metrics_path = OUT / "metrics.jsonl"
        config_results, config_summaries = [], []
        for budget, _ in BUDGETS:
            for policy in POLICIES:
                if time.monotonic() > deadline - 120:
                    raise TimeoutError("CPU fine-tuning reached the runtime safety guard")
                selected_ids = selected[(budget, policy)]
                labels = np.asarray([pair["label"] for pair in pairs], dtype=np.float32)
                for index in selected_ids:
                    score = float(teacher_scores[index])
                    labels[index] = 1 / (1 + math.exp(-max(-30, min(30, score / 2))))
                pool_ids = np.asarray(sorted(set(base_training_ids).union(selected_ids)), dtype=np.int64)
                sampled = np.random.default_rng(SEED).choice(pool_ids, size=TRAIN_STEPS * BATCH_SIZE, replace=True)
                torch.manual_seed(SEED)
                model = AutoModel.from_pretrained(str(student_path), local_files_only=True).to(DEVICE)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
                train_started = time.monotonic()
                for step in range(TRAIN_STEPS):
                    if time.monotonic() > deadline - 120:
                        raise TimeoutError("CPU fine-tuning reached the runtime safety guard")
                    ids = sampled[step * BATCH_SIZE : (step + 1) * BATCH_SIZE].tolist()
                    queries = [groups[pairs[i]["group"]]["query"] for i in ids]
                    passages = [train_items[pairs[i]["item"]]["paragraphs"][pairs[i]["candidate"]] for i in ids]
                    q_batch = student_tokenizer(queries, padding=True, truncation=True, max_length=MAX_LEN,
                                                return_tensors="pt")
                    p_batch = student_tokenizer(passages, padding=True, truncation=True, max_length=MAX_LEN,
                                                return_tensors="pt")
                    optimizer.zero_grad(set_to_none=True)
                    q_hidden = model(**q_batch).last_hidden_state.float()
                    p_hidden = model(**p_batch).last_hidden_state.float()
                    q_mask = q_batch["attention_mask"].unsqueeze(-1).float()
                    p_mask = p_batch["attention_mask"].unsqueeze(-1).float()
                    qv = F.normalize((q_hidden * q_mask).sum(1) / q_mask.sum(1).clamp(min=1), dim=1)
                    pv = F.normalize((p_hidden * p_mask).sum(1) / p_mask.sum(1).clamp(min=1), dim=1)
                    logits = (qv * pv).sum(1) * 20
                    target = torch.tensor(labels[ids], dtype=torch.float32)
                    loss = F.binary_cross_entropy_with_logits(logits, target)
                    loss.backward()
                    optimizer.step()
                train_seconds = time.monotonic() - train_started
                outcomes = evaluate(model)
                for row in outcomes:
                    config_results.append({"budget_pct": int(budget), "policy": policy, **row})
                config_summaries.append({"budget_pct": int(budget), "policy": policy,
                                         "target_teacher_pairs": targets[budget],
                                         "actual_teacher_pairs": len(selected_ids),
                                         "candidate_pair_pool": total_pairs,
                                         "training_pool_pairs": len(pool_ids),
                                         "optimizer_steps": TRAIN_STEPS, "batch_size": BATCH_SIZE,
                                         "train_seconds": train_seconds})
                with metrics_path.open("a" if metrics_path.exists() else "x") as stream:
                    for row in outcomes:
                        stream.write(json.dumps({"kind": "query", "budget_pct": int(budget),
                                                 "policy": policy, **row}, separators=(",", ":")) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(f"Finished {len(config_summaries)}/9: {policy} {budget}%", flush=True)
                del model, optimizer, outcomes

        maps = {}
        for row in config_results:
            maps.setdefault((row["budget_pct"], row["policy"]), {})[row["qid"]] = row
        bridge_ids = [item["qid"] for item in dev_items if item["type"] == "bridge"]
        all_ids = [item["qid"] for item in dev_items]
        rng = np.random.default_rng(0)
        bridge_samples = rng.integers(0, len(bridge_ids), size=(1000, len(bridge_ids)))
        all_samples = rng.integers(0, len(all_ids), size=(1000, len(all_ids)))
        contrasts, config_metrics = [], []
        for budget, _ in BUDGETS:
            random_map = maps[(int(budget), "random")]
            for policy in POLICIES:
                current = maps[(int(budget), policy)]
                config_metrics.append({"budget_pct": int(budget), "policy": policy,
                                       "overall_fs2": float(np.mean([v["full_support_at_2"] for v in current.values()])),
                                       "bridge_fs2": float(np.mean([v["full_support_at_2"] for v in current.values()
                                                                     if v["question_type"] == "bridge"])),
                                       "comparison_fs2": float(np.mean([v["full_support_at_2"] for v in current.values()
                                                                         if v["question_type"] == "comparison"]))})
                for ids, draws, metric in [(bridge_ids, bridge_samples, "bridge_fs2"),
                                           (all_ids, all_samples, "overall_fs2")]:
                    deltas = np.asarray([int(current[q]["full_support_at_2"])
                                         - int(random_map[q]["full_support_at_2"]) for q in ids], dtype=np.float32)
                    bounds = np.quantile(deltas[draws].mean(axis=1), [0.025, 0.975])
                    contrasts.append({"kind": "contrast", "budget_pct": int(budget), "policy": policy,
                                      "baseline": "random", "metric": metric, "n_queries": len(ids),
                                      "delta_policy_minus_random": float(deltas.mean()),
                                      "paired_bootstrap_ci95": [float(bounds[0]), float(bounds[1])],
                                      "bootstrap_resamples": 1000, "bootstrap_seed": 0})
        with metrics_path.open("a") as stream:
            for row in contrasts:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")

        elapsed = time.monotonic() - started
        summary = {
            "compute": {"provider": "Local Mac CPU fallback", "gpu": "none", "device": "CPU only",
                        "threads": 16, "wall_seconds": elapsed, "cost_usd": 0.0,
                        "modal_status": "not launched; budget and enforced spend limit could not be verified"},
            "data": {"validation_sha256": validation_bytes_hash, "train_source": train_source,
                     "train_eligible_n": train_eligible_n, "train_questions_used": len(train_items),
                     "candidate_pair_pool": total_pairs, "pilot_excluded_n": len(pilot_ids),
                     "seq_holdout_n": len(holdout_ids), "seq_holdout_sha256": sha(holdout_text.encode()),
                     "dev2_n": len(dev_items), "dev2_ids_sha256": split["dev2_ids_sha256"],
                     "dev2_bridge_n": len(bridge_ids)},
            "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION,
                        "cached_file_hashes": student_files},
            "teacher": {"id": TEACHER_ID, "initializer": STUDENT_ID,
                        "train_steps": TEACHER_STEPS, "train_batch_size": TEACHER_BATCH_SIZE,
                        "train_seconds": teacher_train_seconds,
                        "unique_pairs_scored_for_all_configs": len(teacher_needed),
                        "scoring_seconds": teacher_seconds},
            "protocol": {"seed": SEED, "budgets_pct": [5, 20, 100], "policies": POLICIES,
                         "steps_per_config": TRAIN_STEPS, "batch_size": BATCH_SIZE,
                         "max_length": MAX_LEN, "bootstrap_resamples": 1000,
                         "student_uncertainty": "pretrained student's candidate cosine scores nearest the top-1/top-2 decision boundary",
                         "bridge_focused": "conditional second-hop candidate pairs for bridge questions only",
                         "hop2_training_context": "gold first-hop paragraph; evaluation uses the model's predicted first paragraph",
                         "training_loss": "binary cross-entropy on gold labels with selected-pair sigmoid teacher targets",
                         "evaluation": "greedy first-hop then conditional second-hop; exact two-paragraph full support"},
            "configs": config_summaries, "config_metrics": config_metrics,
            "bridge_contrasts": [row for row in contrasts if row["metric"] == "bridge_fs2"],
            "libraries": {name: importlib.metadata.version(name)
                          for name in ["torch", "transformers", "pyarrow", "numpy"]},
        }
        summary["experiment_source_sha256"] = sha(Path(__file__).read_bytes())
        (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"status": "complete", "wall_seconds": elapsed,
                          "dev2": len(dev_items), "train_questions": len(train_items),
                          "configs": len(config_summaries)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
