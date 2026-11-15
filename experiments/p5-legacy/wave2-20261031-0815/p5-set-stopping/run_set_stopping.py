from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parent
PILOT = Path("/Users/nilan/Documents/ire-p5-minimal-evidence/pilot")
DATA = PILOT / "data" / "hotpot_validation.parquet"
DENSE_DIR = PILOT / ".hf-cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
CE_DIR = PILOT / ".hf-cache/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots/233902d25c440f23af6f7d6e94d2946bac0bee0a"
SEED = 20261031
PILOT_N = 600
SEQ_HOLDOUT_N = 2000
MAX_LEN = 256
TORCH_THREADS = 8
QUESTION_BATCH = 8
PAIR_BATCH = 32
REDUNDANCY_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
POLICIES = (
    "score_gap",
    "ratio_0.50",
    "ratio_0.75",
    "ratio_0.90",
    "softmax_mass_T0.05",
    "softmax_mass_T0.10",
    "softmax_mass_T0.20",
    "knee",
)
CONFIGS = [
    (scorer, objective, float(weight))
    for scorer in ("dense", "ce")
    for objective in ("mmr", "facility_location", "query_token_coverage")
    for weight in REDUNDANCY_WEIGHTS
]
DATA_URL = "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_title(value: Any) -> str:
    return " ".join(str(value).split()).casefold()


def is_eligible(row: dict[str, Any]) -> bool:
    if row.get("type") not in {"bridge", "comparison"}:
        return False
    context = row.get("context") or {}
    facts = row.get("supporting_facts") or {}
    titles = context.get("title") or []
    groups = context.get("sentences") or []
    gold = {norm_title(x) for x in facts.get("title", [])}
    if len(titles) != 10 or len(groups) != 10 or len(gold) != 2:
        return False
    matched = [norm_title(title) for title in titles if norm_title(title) in gold]
    return len(matched) == 2 and len(set(matched)) == 2


def prepare(row: dict[str, Any]) -> dict[str, Any]:
    qid = str(row["id"])
    context = row["context"]
    titles, sentence_groups = context["title"], context["sentences"]
    paragraphs = [
        str(title) + " " + " ".join(str(s).strip() for s in group)
        for title, group in zip(titles, sentence_groups)
    ]
    gold_titles = {norm_title(x) for x in row["supporting_facts"]["title"]}
    gold = [i for i, title in enumerate(titles) if norm_title(title) in gold_titles]
    if len(gold) != 2 or len(set(gold)) != 2:
        raise ValueError(f"Integrity filter mismatch for {qid}")
    return {
        "qid": qid,
        "question": str(row["question"]),
        "type": str(row["type"]),
        "paragraphs": paragraphs,
        "gold": np.asarray(gold, dtype=np.int8),
    }


def write_new_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Existing artifact differs; refusing to overwrite: {path}")
        return
    with path.open("xb") as f:
        f.write(content)


def split_data() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as pq

    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    data_hash = sha256_file(DATA)
    rows = pq.read_table(DATA).to_pylist()
    eligible = [r for r in rows if is_eligible(r)]
    if len({str(r["id"]) for r in eligible}) != len(eligible):
        raise ValueError("Eligible validation rows contain duplicate question IDs")
    if len(eligible) < PILOT_N + SEQ_HOLDOUT_N:
        raise ValueError(f"Not enough eligible rows: {len(eligible)}")

    chosen = np.random.default_rng(SEED).choice(len(eligible), PILOT_N, replace=False)
    pilot_ids = {str(eligible[int(i)]["id"]) for i in chosen}
    remaining = [r for r in eligible if str(r["id"]) not in pilot_ids]
    remaining.sort(
        key=lambda r: (
            hashlib.sha256(str(r["id"]).encode("utf-8")).hexdigest(),
            str(r["id"]),
        )
    )
    seq_ids = [str(r["id"]) for r in remaining[:SEQ_HOLDOUT_N]]
    seq_bytes = ("".join(qid + "\n" for qid in seq_ids)).encode("utf-8")
    seq_path = OUT / "SEQ_HOLDOUT_IDS.txt"
    write_new_or_verify(seq_path, seq_bytes)
    seq_digest = hashlib.sha256(seq_bytes).hexdigest()
    write_new_or_verify(
        OUT / "SEQ_HOLDOUT_IDS.sha256",
        f"{seq_digest}  SEQ_HOLDOUT_IDS.txt\n".encode("ascii"),
    )

    dev_rows = remaining[SEQ_HOLDOUT_N:]
    info = {
        "source_url": DATA_URL,
        "source_sha256": data_hash,
        "source_bytes": DATA.stat().st_size,
        "raw_rows": len(rows),
        "eligible_rows": len(eligible),
        "schema_excluded_rows": len(rows) - len(eligible),
        "pilot_excluded_n": len(pilot_ids),
        "pilot_seed": SEED,
        "seq_holdout_n": len(seq_ids),
        "seq_holdout_ids_sha256": seq_digest,
        "dev2_n": len(dev_rows),
        "dev2_type_counts": dict(Counter(str(r["type"]) for r in dev_rows)),
        "split": "filter with the pilot schema rule; recover pilot seed-20261031 sample; sort remaining by (sha256(UTF-8 id), id); first 2000 sealed; evaluate only the rest",
    }
    del rows, eligible, remaining
    return dev_rows, info


def model_files(root: Path) -> list[dict[str, Any]]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return [
        {"name": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in files
    ]


class DenseEncoder:
    def __init__(self, tokenizer: Any, model: Any, torch: Any):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch
        self.truncated = Counter()
        self.inputs = Counter()

    def encode(self, texts: list[str], role: str) -> tuple[np.ndarray, list[np.ndarray]]:
        raw = self.tokenizer(
            texts, add_special_tokens=True, truncation=False, padding=False, verbose=False
        )["input_ids"]
        self.truncated[role] += sum(len(x) > MAX_LEN for x in raw)
        self.inputs[role] += len(texts)
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
            return_special_tokens_mask=True,
            verbose=False,
        )
        special = batch.pop("special_tokens_mask")
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors = vectors.cpu().numpy().astype(np.float32, copy=False)
            valid = batch["attention_mask"].bool() & ~special.bool()
            tokens: list[np.ndarray] = []
            for i in range(hidden.shape[0]):
                values = hidden[i][valid[i]]
                values = self.torch.nn.functional.normalize(values, p=2, dim=1)
                array = values.cpu().numpy().astype(np.float32, copy=False)
                if not len(array):
                    array = np.zeros((1, hidden.shape[-1]), dtype=np.float32)
                tokens.append(array)
        return vectors, tokens


class CrossEncoder:
    def __init__(self, tokenizer: Any, model: Any, torch: Any):
        self.tokenizer, self.model, self.torch = tokenizer, model, torch
        self.truncated = 0
        self.inputs = 0

    def score_pairs(self, questions: list[str], paragraphs: list[str]) -> np.ndarray:
        if len(questions) != len(paragraphs):
            raise ValueError("Cross-encoder question/paragraph pair counts differ")
        scores = []
        for start in range(0, len(paragraphs), PAIR_BATCH):
            qs = questions[start : start + PAIR_BATCH]
            docs = paragraphs[start : start + PAIR_BATCH]
            raw = self.tokenizer(
                qs, docs, add_special_tokens=True, truncation=False, padding=False, verbose=False
            )["input_ids"]
            self.truncated += sum(len(x) > MAX_LEN for x in raw)
            self.inputs += len(docs)
            batch = self.tokenizer(
                qs,
                docs,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
                verbose=False,
            )
            with self.torch.inference_mode():
                logits = self.model(**batch).logits.reshape(-1)
            scores.extend(logits.cpu().numpy().astype(np.float32, copy=False).tolist())
        return np.asarray(scores, dtype=np.float32)


def load_models() -> tuple[DenseEncoder, CrossEncoder, Any, dict[str, Any]]:
    if not DENSE_DIR.is_dir() or not CE_DIR.is_dir():
        raise FileNotFoundError("Pinned local dense and cross-encoder snapshots must both exist")
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(TORCH_THREADS)
    try:
        torch.set_num_interop_threads(4)
    except RuntimeError:
        pass
    dense_tok = AutoTokenizer.from_pretrained(DENSE_DIR, local_files_only=True)
    dense_model = AutoModel.from_pretrained(
        DENSE_DIR, local_files_only=True, use_safetensors=True
    ).to("cpu").eval()
    ce_tok = AutoTokenizer.from_pretrained(CE_DIR, local_files_only=True)
    ce_model = AutoModelForSequenceClassification.from_pretrained(
        CE_DIR, local_files_only=True, use_safetensors=True
    ).to("cpu").eval()
    if ce_model.config.num_labels != 1:
        raise ValueError(f"Expected scalar CE logits; got {ce_model.config.num_labels} labels")
    info = {
        "device": "cpu",
        "threads": TORCH_THREADS,
        "max_length": MAX_LEN,
        "dense_model": {
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": DENSE_DIR.name,
            "files": model_files(DENSE_DIR),
        },
        "cross_encoder": {
            "id": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "revision": CE_DIR.name,
            "files": model_files(CE_DIR),
        },
    }
    return DenseEncoder(dense_tok, dense_model, torch), CrossEncoder(ce_tok, ce_model, torch), torch, info


def minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def greedy_order(
    scorer: str,
    objective: str,
    rho: float,
    dense_scores: np.ndarray,
    ce_scores: np.ndarray,
    doc_similarity: np.ndarray,
    token_coverage: np.ndarray,
) -> tuple[list[int], list[float]]:
    relevance = minmax(dense_scores if scorer == "dense" else ce_scores)
    n = len(relevance)
    selected: list[int] = []
    gains: list[float] = []
    current_facility = np.zeros(n, dtype=np.float32)
    current_tokens = np.zeros(token_coverage.shape[0], dtype=np.float32)

    for _ in range(n):
        best_i, best_value = -1, -np.inf
        for i in range(n):
            if i in selected:
                continue
            if objective == "mmr":
                red = float(np.max(doc_similarity[selected, i])) if selected else 0.0
                value = float(relevance[i] - rho * red)
            elif objective == "facility_location":
                new = np.maximum(current_facility, doc_similarity[:, i])
                coverage_gain = float(np.sum(new - current_facility)) / n
                value = float((1.0 - rho) * relevance[i] + rho * coverage_gain)
            elif objective == "query_token_coverage":
                new = np.maximum(current_tokens, token_coverage[:, i])
                coverage_gain = float(np.mean(new - current_tokens))
                value = float((1.0 - rho) * relevance[i] + rho * coverage_gain)
            else:
                raise ValueError(objective)
            if value > best_value:
                best_i, best_value = i, value
        if best_i < 0:
            raise RuntimeError("Greedy selection failed")
        selected.append(best_i)
        gains.append(float(best_value))
        if objective == "facility_location":
            current_facility = np.maximum(current_facility, doc_similarity[:, best_i])
        elif objective == "query_token_coverage":
            current_tokens = np.maximum(current_tokens, token_coverage[:, best_i])
    return selected, gains


def normalized_curve(values: list[float]) -> np.ndarray:
    return minmax(np.asarray(values, dtype=np.float32))


def stop_sizes(gains: list[float]) -> dict[str, int]:
    raw = np.asarray(gains, dtype=np.float64)
    scaled = minmax(raw.astype(np.float32)).astype(np.float64)
    n = len(raw)
    # Stop immediately before the largest positive adjacent drop; choose 10 if none.
    drops = scaled[:-1] - scaled[1:]
    gap_k = int(np.argmax(drops)) + 1 if len(drops) and np.max(drops) > 0 else n

    result = {"score_gap": gap_k}
    for ratio in (0.50, 0.75, 0.90):
        below = np.flatnonzero(scaled < ratio * scaled[0])
        result[f"ratio_{ratio:.2f}"] = int(below[0]) if len(below) else n
        # below index is zero-based; if rank one itself is below, retain one item.
        result[f"ratio_{ratio:.2f}"] = max(1, result[f"ratio_{ratio:.2f}"])

    shifted = raw - float(np.max(raw))
    for temp in (0.05, 0.10, 0.20):
        logits = shifted / temp
        weights = np.exp(np.clip(logits, -80, 0))
        probs = weights / max(float(np.sum(weights)), 1e-12)
        k = int(np.searchsorted(np.cumsum(probs), 0.90) + 1)
        result[f"softmax_mass_T{temp:.2f}"] = max(1, min(n, k))

    # Knee: maximum distance of the normalized gain curve from its endpoint chord.
    x = np.linspace(0.0, 1.0, n)
    chord = scaled[0] + x * (scaled[-1] - scaled[0])
    result["knee"] = (
        int(np.argmax(np.abs(scaled - chord))) + 1
        if float(np.max(scaled) - np.min(scaled)) > 1e-8
        else n
    )
    return result


def curve_area_gain(
    fixed_support: np.ndarray,
    adaptive_points: list[tuple[float, float, str]],
) -> tuple[float, list[dict[str, Any]]]:
    # Signed normalized area difference on the adaptive policies' shared size span.
    if not adaptive_points:
        return 0.0, []
    grouped: dict[float, tuple[float, str]] = {}
    for size, support, name in adaptive_points:
        x = float(size)
        prior = grouped.get(x)
        if prior is None or support > prior[0]:
            grouped[x] = (float(support), name)
    sorted_points = sorted((x, y, name) for x, (y, name) in grouped.items())
    frontier: list[tuple[float, float, str]] = []
    best = -1.0
    for x, y, name in sorted_points:
        if y > best + 1e-12:
            frontier.append((x, y, name))
            best = y
    if not frontier:
        return 0.0, []
    xs = np.asarray([p[0] for p in frontier], dtype=np.float64)
    ys = np.asarray([p[1] for p in frontier], dtype=np.float64)
    domain_lo = min(float(p[0]) for p in adaptive_points)
    domain_hi = max(float(p[0]) for p in adaptive_points)
    if domain_hi - domain_lo < 1e-10:
        baseline = float(np.interp(domain_lo, np.arange(1, len(fixed_support) + 1), fixed_support))
        return float(ys[-1] - baseline), [
            {"mean_size": float(x), "full_support_rate": float(y), "policy": name}
            for x, y, name in frontier
        ]
    integer_grid = np.arange(math.ceil(domain_lo), math.floor(domain_hi) + 1)
    grid = np.unique(np.concatenate([[domain_lo, domain_hi], xs, integer_grid]))
    adaptive_y = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    fixed_y = np.interp(grid, np.arange(1, len(fixed_support) + 1), fixed_support)
    gain = float(np.trapz(adaptive_y - fixed_y, grid) / (grid[-1] - grid[0]))
    curve = [
        {"mean_size": float(x), "full_support_rate": float(y), "policy": name}
        for x, y, name in frontier
    ]
    return gain, curve


def group_summary(
    indices: np.ndarray,
    needed: np.ndarray,
    gold_ranks: np.ndarray,
    stop_matrix: np.ndarray,
) -> dict[str, Any]:
    n = len(indices)
    output = {}
    for ci, (scorer, objective, rho) in enumerate(CONFIGS):
        req = needed[indices, ci].astype(np.int16)
        ranks = gold_ranks[indices, ci, :].astype(np.int16)
        stops = stop_matrix[indices, ci, :].astype(np.int16)
        fixed = []
        fixed_support = []
        for k in range(1, 11):
            fs = float(np.mean(req <= k))
            rec = float(np.mean((ranks <= k).sum(axis=1) / 2.0))
            fixed_support.append(fs)
            fixed.append({
                "k": k,
                "mean_size": float(k),
                "full_support_rate": fs,
                "paragraph_recall": rec,
            })
        adaptive = {}
        points = []
        for pi, name in enumerate(POLICIES):
            ks = stops[:, pi]
            fs = float(np.mean(ks >= req))
            rec = float(np.mean((ranks <= ks[:, None]).sum(axis=1) / 2.0))
            mean_size = float(np.mean(ks))
            adaptive[name] = {
                "mean_size": mean_size,
                "full_support_rate": fs,
                "paragraph_recall": rec,
            }
            points.append((mean_size, fs, name))
        gain, adaptive_frontier = curve_area_gain(np.asarray(fixed_support), points)
        union_points = [
            (float(x["mean_size"]), float(x["full_support_rate"]), f"fixed_k_{x['k']}")
            for x in fixed
        ] + points
        _, combined_frontier = curve_area_gain(np.asarray(fixed_support), union_points)
        output[f"{scorer}|{objective}|rho={rho:.2f}"] = {
            "score_source": scorer,
            "objective": objective,
            "redundancy_weight": rho,
            "n_queries": n,
            "fixed_k_curve": fixed,
            "adaptive_policies": adaptive,
            "adaptive_pareto_frontier": adaptive_frontier,
            "fixed_k_area_gain": 0.0,
            "adaptive_auc_gain_over_fixed_k": gain,
        "adaptive_auc_domain": (
                [min(p[0] for p in points), max(p[0] for p in points)] if points else None
            ),
            "combined_fixed_and_adaptive_frontier": combined_frontier,
        }
    return output


def bootstrap_grid_gain(
    indices: np.ndarray,
    needed: np.ndarray,
    stop_matrix: np.ndarray,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[int, float, list[float], list[float]]:
    n = len(indices)
    if n < 3:
        raise ValueError("Need at least three bridge queries for bootstrap")
    req_all = needed[indices]
    stop_all = stop_matrix[indices]
    rng = np.random.default_rng(seed)
    grid_means = []
    best_gains = []
    point_gains = []
    # Point estimates for all fixed-grid configurations.
    for c in range(len(CONFIGS)):
        fixed = np.asarray([np.mean(req_all[:, c] <= k) for k in range(1, 11)])
        pts = [
            (float(np.mean(stop_all[:, c, p])), float(np.mean(stop_all[:, c, p] >= req_all[:, c])), POLICIES[p])
            for p in range(len(POLICIES))
        ]
        point_gains.append(curve_area_gain(fixed, pts)[0])
    best_c = int(np.argmax(point_gains))
    point_grid_mean = float(np.mean(point_gains))
    for b in range(resamples):
        sample = rng.integers(0, n, size=n)
        gains = []
        req = req_all[sample]
        stops = stop_all[sample]
        for c in range(len(CONFIGS)):
            fixed = np.asarray([np.mean(req[:, c] <= k) for k in range(1, 11)])
            pts = [
                (float(np.mean(stops[:, c, p])), float(np.mean(stops[:, c, p] >= req[:, c])), POLICIES[p])
                for p in range(len(POLICIES))
            ]
            gains.append(curve_area_gain(fixed, pts)[0])
        grid_means.append(float(np.mean(gains)))
        best_gains.append(float(gains[best_c]))
        if (b + 1) % 250 == 0:
            print(f"Bootstrap {b + 1}/{resamples}", flush=True)
    ci_grid = np.percentile(np.asarray(grid_means), [2.5, 97.5]).tolist()
    ci_best = np.percentile(np.asarray(best_gains), [2.5, 97.5]).tolist()
    return best_c, point_grid_mean, ci_grid, ci_best


def build_metrics(
    dev: list[dict[str, Any]],
    info: dict[str, Any],
    dense_encoder: DenseEncoder,
    cross_encoder: CrossEncoder,
    run_started: float,
    model_info: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n = len(dev)
    needed = np.zeros((n, len(CONFIGS)), dtype=np.int8)
    gold_ranks = np.zeros((n, len(CONFIGS), 2), dtype=np.int8)
    stop_matrix = np.zeros((n, len(CONFIGS), len(POLICIES)), dtype=np.int8)
    query_hashes = []
    for start in range(0, n, QUESTION_BATCH):
        group = dev[start : start + QUESTION_BATCH]
        flat_paras = [para for item in group for para in item["paragraphs"]]
        flat_questions = [item["question"] for item in group for _ in item["paragraphs"]]
        dense_paras, para_tokens = dense_encoder.encode(flat_paras, "candidate_paragraphs")
        questions = [item["question"] for item in group]
        dense_queries, query_tokens = dense_encoder.encode(questions, "queries")
        group_ce_scores = cross_encoder.score_pairs(flat_questions, flat_paras)
        for j, item in enumerate(group):
            lo = j * 10
            hi = lo + 10
            para_vectors = dense_paras[lo:hi]
            tokens = para_tokens[lo:hi]
            qvec = dense_queries[j]
            qtokens = query_tokens[j]
            dense_scores = para_vectors @ qvec
            ce_scores = group_ce_scores[lo:hi]
            doc_similarity = np.clip((para_vectors @ para_vectors.T + 1.0) / 2.0, 0.0, 1.0)
            token_coverage = np.zeros((len(qtokens), 10), dtype=np.float32)
            for pi, ptokens in enumerate(tokens):
                token_coverage[:, pi] = np.clip(np.max(qtokens @ ptokens.T, axis=1), 0.0, 1.0)

            qid = item["qid"]
            query_hashes.append(hashlib.sha256(qid.encode("utf-8")).hexdigest())
            ranks_config = []
            for ci, (scorer, objective, rho) in enumerate(CONFIGS):
                order, gains = greedy_order(
                    scorer, objective, rho, dense_scores, ce_scores, doc_similarity, token_coverage
                )
                ranks = np.empty(10, dtype=np.int8)
                for rank, candidate in enumerate(order, 1):
                    ranks[candidate] = rank
                gr = np.sort(ranks[item["gold"]])
                gold_ranks[start + j, ci, :] = gr
                needed[start + j, ci] = int(np.max(gr))
                decisions = stop_sizes(gains)
                stop_matrix[start + j, ci, :] = [decisions[name] for name in POLICIES]
                ranks_config.append(order)
            if (start + j + 1) % 100 == 0 or start + j + 1 == n:
                print(f"Scored DEV2 {start + j + 1}/{n}", flush=True)

    if len(query_hashes) != n or np.any(needed < 1) or np.any(needed > 10):
        raise AssertionError("Scored DEV2 cardinality or gold ranks invalid")
    if np.any(stop_matrix < 1) or np.any(stop_matrix > 10):
        raise AssertionError("Adaptive stopping returned an invalid set size")

    groups = {
        "overall": np.arange(n, dtype=np.int32),
        "bridge": np.asarray([i for i, x in enumerate(dev) if x["type"] == "bridge"], dtype=np.int32),
        "comparison": np.asarray([i for i, x in enumerate(dev) if x["type"] == "comparison"], dtype=np.int32),
    }
    records = []
    for group_name, indices in groups.items():
        summary = group_summary(indices, needed, gold_ranks, stop_matrix)
        for key, values in summary.items():
            records.append({
                "record_type": "aggregate_metrics",
                "split": "DEV2",
                "group": group_name,
                "configuration": key,
                **values,
            })

    bridge = groups["bridge"]
    best_c, point_grid_mean, ci_grid, ci_best = bootstrap_grid_gain(
        bridge, needed, stop_matrix, resamples=1000, seed=0
    )
    best_scorer, best_objective, best_rho = CONFIGS[best_c]
    # Add a compact provenance and paired-bootstrap record.
    records.append({
        "record_type": "paired_query_bootstrap",
        "split": "DEV2",
        "group": "bridge",
        "resamples": 1000,
        "seed": 0,
        "unit": "question ID, paired across fixed-k and adaptive curves",
        "predefined_grid_contrast": "mean signed adaptive-frontier AUC gain over fixed-k across all 30 scorer/objective/redundancy configurations",
        "mean_grid_gain_ci95": ci_grid,
        "best_configuration_descriptive": {
            "score_source": best_scorer,
            "objective": best_objective,
            "redundancy_weight": best_rho,
            "point_gain": float(max(
                records[i]["adaptive_auc_gain_over_fixed_k"]
                for i, r in enumerate(records)
                if r.get("record_type") == "aggregate_metrics"
                and r.get("group") == "bridge"
                and r.get("score_source") == best_scorer
                and r.get("objective") == best_objective
                and r.get("redundancy_weight") == best_rho
            )),
            "gain_ci95_descriptive_post_sweep": ci_best,
        },
    })
    summary = {
        "hypothesis": "Training-free score-aware set objectives plus adaptive stopping improve the full-support versus mean-set-size frontier over fixed-k selection on DEV2.",
        "compute": {
            "device": "Local Mac CPU only",
            "wall_seconds": time.perf_counter() - run_started,
            "cost_usd": 0,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": TORCH_THREADS,
            "question_batch": QUESTION_BATCH,
            "pair_batch": PAIR_BATCH,
        },
        "data": info,
        "models": model_info,
        "grid": {
            "score_sources": ["dense", "ce"],
            "objectives": ["mmr", "facility_location", "query_token_coverage"],
            "redundancy_weights": list(REDUNDANCY_WEIGHTS),
            "adaptive_stopping_policies": list(POLICIES),
            "fixed_k": list(range(1, 11)),
            "softmax_mass_target": 0.90,
            "config_count": len(CONFIGS),
        },
        "encoder_usage": {
            "dense_inputs": dict(dense_encoder.inputs),
            "dense_truncated": dict(dense_encoder.truncated),
            "cross_encoder_pairs": cross_encoder.inputs,
            "cross_encoder_truncated_pairs": cross_encoder.truncated,
            "training": "none; this branch is training-free",
        },
        "best_configuration": {
            "score_source": best_scorer,
            "objective": best_objective,
            "redundancy_weight": best_rho,
        },
        "bootstrap": {
            "bridge_queries": len(bridge),
            "resamples": 1000,
            "seed": 0,
            "mean_grid_area_gain": point_grid_mean,
            "mean_grid_area_gain_ci95": ci_grid,
            "best_config_descriptive_ci95": ci_best,
            "paired": True,
        },
    }
    return records, summary


def render_result(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    ci = summary["bootstrap"]["mean_grid_area_gain_ci95"]
    lo, hi = ci
    verdict = "SUPPORT" if lo > 0 else "REFUTE" if hi <= 0 else "INCONCLUSIVE"
    best = summary["best_configuration"]
    bstat = next(
        r for r in records
        if r.get("record_type") == "paired_query_bootstrap"
    )
    best_desc = bstat["best_configuration_descriptive"]
    bridge_rows = [
        r for r in records
        if r.get("record_type") == "aggregate_metrics" and r.get("group") == "bridge"
        and r.get("score_source") == best["score_source"]
        and r.get("objective") == best["objective"]
        and r.get("redundancy_weight") == best["redundancy_weight"]
    ][0]
    adaptive = bridge_rows["adaptive_policies"]
    fixed_support = np.asarray(
        [x["full_support_rate"] for x in bridge_rows["fixed_k_curve"]], dtype=np.float64
    )
    def matched_gain(policy: str) -> float:
        item = adaptive[policy]
        baseline = float(np.interp(
            item["mean_size"], np.arange(1, 11, dtype=np.float64), fixed_support
        ))
        return float(item["full_support_rate"] - baseline)
    top_policy = max(POLICIES, key=matched_gain)
    pts = bridge_rows["combined_fixed_and_adaptive_frontier"]
    pareto_text = "; ".join(f"{p['mean_size']:.2f}:{p['full_support_rate']:.3f}" for p in pts)
    dev_n = summary["data"]["dev2_n"]
    tcounts = summary["data"]["dev2_type_counts"]
    lines = [
        "# RESULT — IRE P5 C1/C3 set selection and stopping",
        "",
        f"- Hypothesis: {summary['hypothesis']}",
        f"- Compute: Local Mac CPU, {summary['compute']['torch_threads']} threads; {summary['compute']['wall_seconds']/60:.1f} min; $0.",
        f"- Scale: DEV2 n={dev_n} ({tcounts.get('bridge',0)} bridge, {tcounts.get('comparison',0)} comparison); no fine-tuning, consistent with training-free branch.",
        f"- Split: 6,742 eligible after pilot input-integrity filter; exact 600 pilot IDs excluded; 2,000 sealed SEQ_HOLDOUT IDs; DEV2 only scored.",
        f"- SEQ_HOLDOUT_IDS.txt SHA256: {summary['data']['seq_holdout_ids_sha256']}.",
        f"- Grid: dense/CE × MMR/facility-location/query-token coverage × redundancy weight {list(REDUNDANCY_WEIGHTS)}; 8 stop rules (softmax target 0.90 at T=.05/.10/.20); fixed k=1…10.",
        f"- Best descriptive config on bridge DEV2: {best['score_source']} + {best['objective']}, redundancy={best['redundancy_weight']:.2f}.",
        f"- Best config CI (post-sweep descriptive): area gain {best_desc['point_gain']:+.4f}, paired query-bootstrap 95% CI [{best_desc['gain_ci95_descriptive_post_sweep'][0]:+.4f}, {best_desc['gain_ci95_descriptive_post_sweep'][1]:+.4f}].",
        f"- Best matched-size rule: {top_policy}, mean |S|={adaptive[top_policy]['mean_size']:.2f}, full-support={adaptive[top_policy]['full_support_rate']:.3f}, gain vs fixed-k={matched_gain(top_policy):+.3f}.",
        f"- Bridge Pareto (mean |S|: full-support): {pareto_text}.",
        f"- Key contrast: mean adaptive-frontier AUC gain over fixed-k across 30 configs={summary['bootstrap']['mean_grid_area_gain']:+.4f}, 95% paired query-bootstrap CI [{lo:+.4f}, {hi:+.4f}], n={summary['bootstrap']['bridge_queries']}, B=1000.",
        f"- Verdict: {verdict}.",
        "- Meaning: " + (
            "adaptive stopping improves set-size/support trade-offs on average across the prespecified grid; confirm on SEQ_HOLDOUT before paper claims."
            if verdict == "SUPPORT" else
            "adaptive stopping does not show a reliable mean grid improvement over fixed-k on DEV2; prioritize fixed-k or redesign stopping before the final experiment."
            if verdict == "REFUTE" else
            "the DEV2 bootstrap does not resolve whether adaptive stopping improves the fixed-k trade-off; treat this branch as exploratory and validate on SEQ_HOLDOUT."
        ),
        f"- Data SHA256: {summary['data']['source_sha256']}; encoder: all-MiniLM-L6-v2 dense + cross-encoder/ms-marco-MiniLM-L6-v2; max length={MAX_LEN}.",
        "- Area definition: signed AUC difference of the adaptive-only Pareto frontier and fixed-k support curve, normalized by the adaptive rules’ shared mean-size span.",
        "- Limit: best-config CI is post-sweep descriptive; the primary CI averages all 30 prespecified configurations and resamples bridge question IDs in pairs.",
    ]
    if len(lines) > 25:
        raise AssertionError(f"RESULT.md exceeds 25 lines: {len(lines)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    for name in ("metrics.jsonl", "RESULT.md"):
        if (OUT / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {OUT / name}")
    run_started = time.perf_counter()
    dev_rows, data_info = split_data()
    if len(dev_rows) != 4742:
        raise ValueError(f"Expected full DEV2 of 4742 questions, got {len(dev_rows)}")
    dev = [prepare(row) for row in dev_rows]
    del dev_rows
    dense_encoder, cross_encoder, _, model_info = load_models()
    records, summary = build_metrics(
        dev, data_info, dense_encoder, cross_encoder, run_started, model_info
    )
    summary["compute"]["wall_seconds"] = time.perf_counter() - run_started
    summary["encoder_usage"] = {
        "dense_inputs": dict(dense_encoder.inputs),
        "dense_truncated": dict(dense_encoder.truncated),
        "cross_encoder_pairs": cross_encoder.inputs,
        "cross_encoder_truncated_pairs": cross_encoder.truncated,
        "training": "none; this branch is training-free",
    }
    metrics_records = [dict(record, provenance=summary) for record in records]
    metrics_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in metrics_records
    )
    result_text = render_result(summary, records)
    with (OUT / "metrics.jsonl").open("x", encoding="utf-8") as f:
        f.write(metrics_text)
    with (OUT / "RESULT.md").open("x", encoding="utf-8") as f:
        f.write(result_text)
    print(result_text, end="", flush=True)
    print(f"Wrote {OUT / 'metrics.jsonl'}", flush=True)


if __name__ == "__main__":
    main()

