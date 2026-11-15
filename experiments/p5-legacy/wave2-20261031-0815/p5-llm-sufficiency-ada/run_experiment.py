from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import string
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/wave2-20261031-0815/p5-llm-sufficiency-ada")
DATA_PATH = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot/data/hotpot_validation.parquet")
DATA_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
DATA_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
SEED = 20261031
PILOT_N = 600
HOLDOUT_N = 2000
SAMPLE_N = 1500
DENSE_ID = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CE_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CE_REV = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
READER_ID = "Qwen/Qwen2.5-7B-Instruct"
SCREEN_COLUMNS = ["id", "type", "context.title", "context.sentences", "supporting_facts.title"]
FULL_COLUMNS = ["id", "question", "answer", "type", "context", "supporting_facts"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_title(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def ids_digest(ids: list[str]) -> str:
    return sha256_bytes("\n".join(ids).encode("utf-8"))


def write_once_or_verify(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing artifact differs; refusing to overwrite: {path}")
        return
    with path.open("xb") as f:
        f.write(data)


def eligible(row: dict) -> bool:
    try:
        if row["type"] not in {"bridge", "comparison"}:
            return False
        titles = row["context"]["title"]
        sentence_groups = row["context"]["sentences"]
        if len(titles) != 10 or len(sentence_groups) != 10:
            return False
        gold = {normalized_title(x) for x in row["supporting_facts"]["title"]}
        return len(gold) == 2 and sum(normalized_title(x) in gold for x in titles) == 2
    except (KeyError, TypeError):
        return False


def get_split(data_path: Path, np):
    import pyarrow.parquet as pq

    source_hash = sha256_file(data_path)
    if source_hash != DATA_SHA256:
        raise ValueError(f"Unexpected HotpotQA validation SHA256: {source_hash}")
    eligible_rows = []
    source_rows = 0
    for batch in pq.ParquetFile(data_path).iter_batches(batch_size=128, columns=SCREEN_COLUMNS):
        for row in batch.to_pylist():
            if eligible(row):
                eligible_rows.append({"id": str(row["id"]), "type": str(row["type"])})
            source_rows += 1
    if len(eligible_rows) != 7342 or source_rows != 7405:
        raise ValueError(f"Eligibility drift: source={source_rows}, eligible={len(eligible_rows)}")
    if len({x["id"] for x in eligible_rows}) != len(eligible_rows):
        raise ValueError("Question IDs are not unique")

    sampled_ix = np.random.default_rng(SEED).choice(len(eligible_rows), PILOT_N, replace=False)
    pilot = [eligible_rows[int(i)] for i in sampled_ix]
    pilot.sort(key=lambda x: (hashlib.sha256(x["id"].encode()).hexdigest(), x["id"]))
    pilot_ids = {x["id"] for x in pilot}
    remaining = [x for x in eligible_rows if x["id"] not in pilot_ids]
    remaining.sort(key=lambda x: (hashlib.sha256(x["id"].encode()).hexdigest(), x["id"]))
    holdout = remaining[:HOLDOUT_N]
    dev2 = remaining[HOLDOUT_N:]
    if len(holdout) != HOLDOUT_N or len(dev2) != 4742:
        raise ValueError(f"Unexpected split sizes: holdout={len(holdout)}, DEV2={len(dev2)}")
    sample = dev2[:SAMPLE_N]
    return {
        "source_sha256": source_hash,
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "pilot_ids": [x["id"] for x in pilot],
        "holdout_ids": [x["id"] for x in holdout],
        "dev2_ids": [x["id"] for x in dev2],
        "sample_ids": [x["id"] for x in sample],
        "type_counts": dict(Counter(x["type"] for x in sample)),
    }


def prepare_split():
    import numpy as np

    if not OUT.is_dir():
        raise FileNotFoundError(f"OUT must exist before split artifacts are written: {OUT}")
    split = get_split(DATA_PATH, np)
    files = {
        "SEQ_HOLDOUT_IDS.txt": split["holdout_ids"],
        "DEV2_SAMPLE_IDS.txt": split["sample_ids"],
    }
    manifest = {
        "task_id": "ire-w2-p5-llm-sufficiency-ada-20261031",
        "dataset_url": DATA_URL,
        "dataset_sha256": split["source_sha256"],
        "source_rows": split["source_rows"],
        "eligible_rows": len(split["eligible_rows"]),
        "pilot_excluded_n": len(split["pilot_ids"]),
        "pilot_excluded_ordered_ids_sha256": ids_digest(split["pilot_ids"]),
        "seq_holdout_n": len(split["holdout_ids"]),
        "seq_holdout_ordered_ids_sha256": ids_digest(split["holdout_ids"]),
        "dev2_n": len(split["dev2_ids"]),
        "dev2_ordered_ids_sha256": ids_digest(split["dev2_ids"]),
        "dev2_sample_n": len(split["sample_ids"]),
        "dev2_sample_ordered_ids_sha256": ids_digest(split["sample_ids"]),
        "dev2_sample_type_counts": split["type_counts"],
        "ordering": "eligible IDs; exclude exact pilot seed-20261031 sample; sort remaining by (sha256(UTF-8 qid), qid); first 2000 sequestered; DEV2 remainder; first 1500 DEV2 IDs evaluated",
        "holdout_evaluation": "never loaded with question, answer, or paragraph text; ID list only",
    }
    for name, ids in files.items():
        data = ("\n".join(ids) + "\n").encode("utf-8")
        write_once_or_verify(OUT / name, data)
        write_once_or_verify(OUT / f"{name}.sha256", (sha256_bytes(data) + "  " + name + "\n").encode())
    write_once_or_verify(OUT / "split_manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(manifest, sort_keys=True))


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_scores(pred: str, gold: str) -> tuple[int, float]:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    em = int(p == g)
    pt, gt = p.split(), g.split()
    if not pt or not gt:
        return em, float(pt == gt)
    common = sum((Counter(pt) & Counter(gt)).values())
    if common == 0:
        return em, 0.0
    precision, recall = common / len(pt), common / len(gt)
    return em, 2 * precision * recall / (precision + recall)


def percentile_ci(values, np, seed: int = SEED, reps: int = 2000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(reps, len(values)), dtype=np.int32)
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def remote_experiment(selected_ids, manifest):
    started = time.monotonic()
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import gc
    import numpy as np
    import pyarrow.dataset as ds
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not provide a CUDA GPU")
    data_path = Path("/tmp/ire-p5-hotpot-validation.parquet")
    if not data_path.exists():
        urllib.request.urlretrieve(DATA_URL, data_path)
    if manifest["dataset_sha256"] != DATA_SHA256 or ids_digest(selected_ids) != manifest["dev2_sample_ordered_ids_sha256"]:
        raise ValueError("Local split manifest and DEV2 sample IDs do not match")
    if len(selected_ids) != SAMPLE_N or manifest["seq_holdout_n"] != HOLDOUT_N:
        raise ValueError("Declared sample or sequestered holdout size is invalid")
    table = ds.dataset(str(data_path), format="parquet").to_table(
        columns=FULL_COLUMNS, filter=ds.field("id").isin(selected_ids)
    )
    rows = table.to_pylist()
    by_id = {str(r["id"]): r for r in rows}
    if len(by_id) != SAMPLE_N or set(by_id) != set(selected_ids):
        raise ValueError("Filtered DEV2 sample does not match the declared 1500 IDs")

    items = []
    for qid in selected_ids:
        row = by_id[qid]
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        paragraphs = [
            (str(t).strip() + " " + " ".join(str(s).strip() for s in group)).strip()
            for t, group in zip(titles, sentences)
        ]
        gold_titles = {normalized_title(x) for x in row["supporting_facts"]["title"]}
        gold = {i for i, title in enumerate(titles) if normalized_title(title) in gold_titles}
        if len(paragraphs) != 10 or len(gold) != 2:
            raise ValueError(f"Malformed selected DEV2 row: {qid}")
        items.append({
            "qid": qid, "type": str(row["type"]), "question": str(row["question"]),
            "answer": str(row["answer"]), "paragraphs": paragraphs, "gold": gold,
        })

    api = HfApi(token=False)
    dense_sha = api.model_info(DENSE_ID, revision=DENSE_REV).sha
    ce_sha = api.model_info(CE_ID, revision=CE_REV).sha
    reader_sha = api.model_info(READER_ID).sha
    device = "cuda"
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True

    def get_revision(model_id, revision):
        return revision

    dense_tok = AutoTokenizer.from_pretrained(DENSE_ID, revision=get_revision(DENSE_ID, dense_sha), token=False)
    dense_model = AutoModel.from_pretrained(
        DENSE_ID, revision=dense_sha, token=False, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    def encode(texts, batch_size=128):
        vectors = []
        for start_i in range(0, len(texts), batch_size):
            batch = dense_tok(
                texts[start_i:start_i + batch_size], padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = dense_model(**batch).last_hidden_state.float()
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.concatenate(vectors, axis=0)

    questions = [x["question"] for x in items]
    passages = [p for x in items for p in x["paragraphs"]]
    qvec = encode(questions)
    pvec = encode(passages).reshape(SAMPLE_N, 10, -1)
    dense_orders = []
    for i in range(SAMPLE_N):
        dense_orders.append(np.argsort(-(pvec[i] @ qvec[i]), kind="stable").astype(int).tolist())

    del dense_model, dense_tok
    gc.collect()
    torch.cuda.empty_cache()

    ce_tok = AutoTokenizer.from_pretrained(CE_ID, revision=ce_sha, token=False)
    ce_model = AutoModelForSequenceClassification.from_pretrained(
        CE_ID, revision=ce_sha, token=False, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    def score_pairs(pairs, batch_size=256):
        scores = []
        for start_i in range(0, len(pairs), batch_size):
            chunk = pairs[start_i:start_i + batch_size]
            batch = ce_tok(
                [x[0] for x in chunk], [x[1] for x in chunk],
                padding=True, truncation="longest_first", max_length=256,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                scores.extend(ce_model(**batch).logits[:, 0].float().cpu().tolist())
        return scores

    ce_orders = [[] for _ in items]
    remaining = [set(range(10)) for _ in items]
    for depth in range(5):
        mapping = []
        for qi, item in enumerate(items):
            prefix = " ".join(item["paragraphs"][j] for j in ce_orders[qi])
            condition = item["question"] if depth == 0 else item["question"] + " " + prefix
            for pi in range(10):
                if pi in remaining[qi]:
                    mapping.append((qi, pi, condition, item["paragraphs"][pi]))
        scores = score_pairs([(x[2], x[3]) for x in mapping])
        per_q = defaultdict(list)
        for (qi, pi, _, _), score in zip(mapping, scores):
            per_q[qi].append((float(score), pi))
        for qi in range(SAMPLE_N):
            chosen = sorted(per_q[qi], key=lambda x: (-x[0], x[1]))[0][1]
            ce_orders[qi].append(chosen)
            remaining[qi].remove(chosen)
        print(f"conditional CE depth={depth + 1} scored={len(mapping)}", flush=True)

    del ce_model, ce_tok
    gc.collect()
    torch.cuda.empty_cache()

    reader_tok = AutoTokenizer.from_pretrained(READER_ID, revision=reader_sha, token=False)
    llm = LLM(
        model=READER_ID, revision=reader_sha, dtype="bfloat16", max_model_len=4096,
        gpu_memory_utilization=0.85, max_num_batched_tokens=16384, max_num_seqs=64,
        trust_remote_code=False,
    )

    def render_evidence(item, indices):
        return "\n\n".join(item["paragraphs"][i] for i in indices)

    def chat_prompt(system, user):
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def generate(messages, max_new_tokens, label, batch_size=2048):
        outputs = []
        sampling = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, stop=["\n"],
        )
        for start_i in range(0, len(messages), batch_size):
            response = llm.chat(
                messages[start_i:start_i + batch_size],
                sampling_params=sampling,
                use_tqdm=True,
            )
            outputs.extend(x.outputs[0].text.strip() for x in response)
            print(f"{label} generated={len(outputs)}/{len(messages)}", flush=True)
        return outputs

    answer_specs = [("gold_support", None), ("dense", 1), ("dense", 2), ("dense", 3),
                    ("dense", 4), ("dense", 5), ("ce_conditional", 1), ("ce_conditional", 2),
                    ("ce_conditional", 3), ("ce_conditional", 4), ("ce_conditional", 5), ("all_10", None)]
    answer_prompts, answer_meta = [], []
    answer_system = "Answer using only the evidence. Return the shortest answer phrase. If the evidence does not answer the question, reply unknown."
    for qi, item in enumerate(items):
        for method, k in answer_specs:
            order = list(range(10)) if method == "all_10" else (
                sorted(item["gold"]) if method == "gold_support" else
                (dense_orders[qi] if method == "dense" else ce_orders[qi])[:k]
            )
            evidence = render_evidence(item, order)
            user = f"Question: {item['question']}\n\nEvidence:\n{evidence}\n\nAnswer:"
            answer_prompts.append(chat_prompt(answer_system, user))
            answer_meta.append((qi, method, k, order, len(evidence.split()), item["gold"].issubset(order)))
    answer_texts = generate(answer_prompts, 32, "answers")
    answer_rows = []
    for meta, pred in zip(answer_meta, answer_texts):
        qi, method, k, order, evidence_words, full_support = meta
        em, f1 = answer_scores(pred, items[qi]["answer"])
        answer_rows.append({
            "qid": items[qi]["qid"], "type": items[qi]["type"], "method": method, "k": k,
            "paragraphs": len(order), "evidence_words": evidence_words,
            "full_support": bool(full_support), "em": em, "f1": f1,
        })
    del answer_prompts, answer_texts, answer_meta
    gc.collect()
    torch.cuda.empty_cache()

    judge_prompts, judge_meta = [], []
    judge_system = "Judge only whether the evidence contains enough information to answer the question. Reply YES or NO only."
    for qi, item in enumerate(items):
        for method, order in (("dense", dense_orders[qi]), ("ce_conditional", ce_orders[qi])):
            for k in range(1, 6):
                selected = order[:k]
                user = (
                    f"Question: {item['question']}\n\nEvidence:\n{render_evidence(item, selected)}"
                    "\n\nIs the evidence sufficient to answer the question? YES or NO:"
                )
                judge_prompts.append(chat_prompt(judge_system, user))
                judge_meta.append((qi, method, k, item["gold"].issubset(selected)))
    judge_texts = generate(judge_prompts, 4, "judges")
    judge_rows = []
    for meta, text in zip(judge_meta, judge_texts):
        qi, method, k, gold_full_support = meta
        first = normalize_answer(text).split()
        judge_rows.append({
            "qid": items[qi]["qid"], "method": method, "k": k,
            "gold_full_support": bool(gold_full_support),
            "judged_sufficient": bool(first and first[0] == "yes"),
        })
    del llm, reader_tok, judge_prompts, judge_texts, judge_meta
    gc.collect()
    torch.cuda.empty_cache()

    records = [{
        "record": "run_metadata",
        "task_id": "ire-w2-p5-llm-sufficiency-ada-20261031",
        "status": "complete",
        "dataset_sha256": DATA_SHA256,
        "eligible_n": manifest["eligible_rows"],
        "pilot_excluded_n": manifest["pilot_excluded_n"],
        "seq_holdout_n": manifest["seq_holdout_n"],
        "seq_holdout_ids_sha256": manifest["seq_holdout_ordered_ids_sha256"],
        "dev2_n": manifest["dev2_n"],
        "dev2_sample_n": len(items),
        "sample_ids_sha256": ids_digest(selected_ids),
        "sample_type_counts": manifest["dev2_sample_type_counts"],
        "reader_model": READER_ID,
        "reader_revision": reader_sha,
        "dense_model": DENSE_ID,
        "dense_revision": dense_sha,
        "conditional_ce_model": CE_ID,
        "conditional_ce_revision": ce_sha,
        "answer_query_n": len(answer_rows),
        "judge_query_n": len(judge_rows),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "cuda_device": torch.cuda.get_device_name(0),
        "libraries": {x: importlib.metadata.version(x) for x in ("numpy", "torch", "transformers", "pyarrow", "huggingface-hub", "vllm")},
    }]

    condition_groups = defaultdict(list)
    for row in answer_rows:
        condition_groups[(row["method"], row["k"])].append(row)
    for (method, k), group in sorted(condition_groups.items(), key=lambda x: (str(x[0][0]), str(x[0][1]))):
        records.append({
            "record": "answer_metrics", "method": method, "k": k, "query_n": len(group),
            "em": float(np.mean([x["em"] for x in group])),
            "f1": float(np.mean([x["f1"] for x in group])),
            "mean_paragraphs": float(np.mean([x["paragraphs"] for x in group])),
            "mean_evidence_words": float(np.mean([x["evidence_words"] for x in group])),
            "full_support_rate": float(np.mean([x["full_support"] for x in group])),
        })

    for method in ("dense", "ce_conditional"):
        by_q = {k: {x["qid"]: x for x in condition_groups[(method, k)]} for k in range(1, 6)}
        mins = []
        for qid in selected_ids:
            first_k = next((k for k in range(1, 6) if by_q[k][qid]["em"] == 1), None)
            if first_k is not None:
                mins.append(first_k)
        records.append({
            "record": "answer_reachability", "method": method, "query_n": len(selected_ids),
            "em_reachable_n": len(mins), "em_reachable_rate": len(mins) / len(selected_ids),
            "mean_min_k": float(np.mean(mins)) if mins else None,
            "median_min_k": float(np.median(mins)) if mins else None,
            "criterion": "first k in 1..5 with exact match",
        })

    for k in range(1, 6):
        d = {x["qid"]: x for x in condition_groups[("dense", k)]}
        c = {x["qid"]: x for x in condition_groups[("ce_conditional", k)]}
        for metric in ("em", "f1"):
            diff = [c[qid][metric] - d[qid][metric] for qid in selected_ids]
            records.append({
                "record": "paired_contrast", "contrast": f"ce_conditional@{k}-dense@{k}",
                "metric": metric, "mean_difference": float(np.mean(diff)),
                "ci95": percentile_ci(diff, np), "bootstrap_reps": 2000,
                "bootstrap_seed": SEED, "unit": "question",
            })

    judge_groups = defaultdict(list)
    for row in judge_rows:
        judge_groups[(row["method"], row["k"])].append(row)
    for (method, k), group in sorted(judge_groups.items()):
        tp = sum(x["judged_sufficient"] and x["gold_full_support"] for x in group)
        fp = sum(x["judged_sufficient"] and not x["gold_full_support"] for x in group)
        fn = sum((not x["judged_sufficient"]) and x["gold_full_support"] for x in group)
        tn = sum((not x["judged_sufficient"]) and (not x["gold_full_support"]) for x in group)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        records.append({
            "record": "judge_metrics", "method": method, "k": k, "query_n": len(group),
            "precision_vs_full_gold_support": precision, "recall_vs_full_gold_support": recall,
            "f1_vs_full_gold_support": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # Query-clustered contrast: answer EM when both gold paragraphs are present vs absent.
    em_by_q = defaultdict(lambda: {"supported": [], "incomplete": []})
    for x in answer_rows:
        if x["method"] in {"dense", "ce_conditional"}:
            em_by_q[x["qid"]]["supported" if x["full_support"] else "incomplete"].append(x["em"])
    aligned = [
        float(np.mean(v["supported"]) - np.mean(v["incomplete"]))
        for v in em_by_q.values() if v["supported"] and v["incomplete"]
    ]
    if aligned:
        records.append({
            "record": "support_alignment", "contrast": "answer_em_full_support-minus-incomplete",
            "query_n": len(aligned), "mean_difference": float(np.mean(aligned)),
            "ci95": percentile_ci(aligned, np), "bootstrap_reps": 2000,
            "bootstrap_seed": SEED, "unit": "question",
        })

    candidates = [r for r in records if r.get("record") == "answer_metrics" and r["method"] in {"dense", "ce_conditional"}]
    best = sorted(candidates, key=lambda r: (-r["f1"], r["mean_paragraphs"], r["method"], str(r["k"])))[0]
    records.append({
        "record": "best_config", "selection_rule": "highest DEV2-sample answer F1; tie-break fewer paragraphs",
        "method": best["method"], "k": best["k"], "f1": best["f1"], "em": best["em"],
        "mean_paragraphs": best["mean_paragraphs"],
    })
    records.append({
        "record": "runtime", "elapsed_seconds": round(time.monotonic() - started, 2),
        "modal_gpu_rate_usd_per_hour": 2.10,
        "max_runtime_seconds": 1500,
    })
    return records


try:
    import modal
except ImportError:
    modal = None

if modal is not None:
    app = modal.App("ire-exp-p5-sufficiency-20261031")
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.13"
        )
        .entrypoint([])
        .uv_pip_install(
            "vllm==0.13.0", "huggingface-hub==0.36.0",
            "pyarrow==19.0.1", "numpy==2.1.3", "transformers",
        )
        .env({"HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_XET": "1"})
    )

    @app.function(
        image=image, gpu="A100", cpu=8, memory=32768,
        timeout=1500, startup_timeout=240, retries=0, max_containers=1,
    )
    def run_remote(selected_ids, manifest):
        return remote_experiment(selected_ids, manifest)

    @app.local_entrypoint()
    def main():
        started = time.monotonic()
        try:
            manifest = json.loads((OUT / "split_manifest.json").read_text())
            selected_ids = (OUT / "DEV2_SAMPLE_IDS.txt").read_text().splitlines()
            records = run_remote.remote(selected_ids, manifest)
        except Exception as exc:
            records = [{
                "record": "run_metadata", "task_id": "ire-w2-p5-llm-sufficiency-ada-20261031",
                "status": "failed", "error_type": type(exc).__name__,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }]
        for record in records:
            if record.get("record") == "run_metadata":
                record.update({
                    "compute": "Modal A100",
                    "modal_profile": "nilansarkar",
                    "modal_app_name": "ire-exp-p5-sufficiency-20261031",
                })
        data = b"".join((json.dumps(x, sort_keys=True) + "\n").encode() for x in records)
        metrics_path = OUT / "metrics.jsonl"
        if metrics_path.exists():
            existing = metrics_path.read_bytes()
            prior = [json.loads(line) for line in existing.splitlines() if line.strip()]
            if prior and all(x.get("status") == "failed" for x in prior):
                with metrics_path.open("ab") as f:
                    f.write(data)
            elif not existing.endswith(data):
                raise FileExistsError(f"Existing metrics differ; refusing to overwrite: {metrics_path}")
        else:
            write_once_or_verify(metrics_path, data)
        print(json.dumps(records, sort_keys=True))


if __name__ == "__main__" and modal is None:
    if sys.argv[1:] != ["--prepare-only"]:
        raise SystemExit("Run with --prepare-only locally, or invoke through modal run.")
    prepare_split()
