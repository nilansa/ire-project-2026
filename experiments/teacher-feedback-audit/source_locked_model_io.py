#!/usr/bin/env python3
"""Source-locked TEXT and re-encoding interface for the exact positive control.

CPU-only metadata/text preparation. This module NEVER loads a model, tokenizes,
runs a forward pass, scores a teacher pair, encodes vectors, or ranks passages.
A future authorized implementation must separately attest that it re-encoded
all frozen passages and held-out queries from the same trained checkpoint.
"""
import argparse
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from exact_control_objectives import ARCHIVE, EXPECTED_ZIP_SHA256


ORIGINAL_RUNTIME = (Path.home() / ".codex/scratch/cross-surface/IRE/"
                    "ann-micro-pilot-20260907/shared/runtime")
SLICE = ORIGINAL_RUNTIME / "cache/slice_50000_20260907.json"
TRAIN_QMAX = 32
PASSAGE_MAX = 128
VECTOR_DIM = 768


@dataclass(frozen=True)
class ModelInputContract:
    """Immutable original-text corpus/evaluation mapping, without embeddings."""

    corpus_ids: tuple[int, ...]
    corpus_texts: tuple[str, ...]
    query_texts: dict[int, str]
    training_query_ids: tuple[int, ...]
    eval_query_ids: tuple[int, ...]
    slate_passage_ids: dict[int, tuple[int, ...]]
    source_slice_sha256: str

    def slate_texts(self, slate):
        """Return original query text and 21 passage texts, positive first."""
        if self.slate_passage_ids.get(slate.query_id) != slate.passage_ids:
            raise ValueError("Slate/order is not part of the frozen training plan")
        by_pid = self._passage_lookup
        return self.query_texts[slate.query_id], tuple(
            self.corpus_texts[by_pid[p]] for p in slate.passage_ids
        )

    def evaluation_inputs(self):
        """Full 50k ordered passages, all 500 held-out original query texts."""
        return (tuple(self.query_texts[q] for q in self.eval_query_ids),
                self.corpus_texts)

    @cached_property
    def _passage_lookup(self):
        return {p: row for row, p in enumerate(self.corpus_ids)}

    @staticmethod
    def source_encoding_contract():
        """Historical training and final-evaluation semantics, not an encoder."""
        return {
            "training_forward": {
                "query_max_length": TRAIN_QMAX,
                "passage_max_length": PASSAGE_MAX,
                "tokenizer_padding": "max_length",
                "tokenizer_truncation": True,
                "model_output": "sentence_embedding",
                "scoring": "raw_inner_product",
            },
            "final_evaluation": {
                "encoder": "SentenceTransformer.encode",
                "passage_max_seq_length": PASSAGE_MAX,
                "query_max_seq_length": TRAIN_QMAX,
                "normalize_embeddings": False,
                "numpy_dtype": "float32",
                "corpus_shape": [50000, VECTOR_DIM],
                "query_shape": [500, VECTOR_DIM],
                "ranking": "exhaustive numpy inner product top1000",
                "tie_order": "within returned top-k, descending score then row index",
                "metrics": "existing qrel-locked held-out evaluator",
            },
            "execution_authorization": "NO MODEL INFERENCE OR TRAINING AUTHORIZED",
            "provenance_limit": (
                "A static contract cannot prove future embeddings were freshly "
                "computed from the intended trained checkpoint."
            ),
        }

    def reencode_or_rank(self, *args, **kwargs):
        raise PermissionError("ENCODING/RANKING DISABLED: explicit authorization required")


def load_model_inputs(plan=None):
    """Reconcile real historical texts with the SHA-frozen plan. No model work."""
    if plan is None:
        from guarded_exact_control_runner import build_plan
        plan = build_plan()
    archive_bytes = Path(ARCHIVE).read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != EXPECTED_ZIP_SHA256:
        raise ValueError("Unexpected October 12 archive")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
        nested = outer.read("inputs/identity-and-search-evidence.zip")
        source = outer.read("inputs/ire-audit-evidence.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as identity_zip, (
         zipfile.ZipFile(io.BytesIO(source))) as pilot:
        identity = json.loads(identity_zip.read("marco/identity.json"))
        frozen_split = json.loads(pilot.read("marco/runtime/split.json"))
        if identity["split"] != frozen_split:
            raise ValueError("Original split mismatch")
        # Source-locked model semantics: fixed-pad training, unnormalized final
        # embeddings and exact inner-product ranking; no code execution.
        student_source = pilot.read("marco/teacher_sanity.py").decode("utf8")
        core_source = pilot.read("marco/run_pilot.py").decode("utf8")
        for needle in (
            "padding='max_length'", "truncation=True", "max_length=length",
            "['sentence_embedding']",
        ):
            if needle not in student_source:
                raise ValueError("Archived training model-input contract changed")
        for needle in (
            "model.max_seq_length=maxlen", "normalize_embeddings=False",
            "np.asarray(x,dtype=np.float32)", "def exact_topk(", "np.lexsort",
        ):
            if needle not in core_source:
                raise ValueError("Archived final encoding/ranking contract changed")

    raw = SLICE.read_bytes()
    expected_hash = identity["source_sha256"].get(str(SLICE))
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is None or digest != expected_hash:
        raise ValueError("Original complete 50k passage-text slice is not verified")
    source = json.loads(raw)
    pids = tuple(map(int, source["pids"]))
    texts = tuple(source["texts"])
    if pids != plan.corpus_ids or len(texts) != 50000 or len(set(pids)) != 50000:
        raise ValueError("Text rows do not align with frozen passage IDs/order")
    if source["split"] != frozen_split:
        raise ValueError("Local texts use a different original split")
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Missing/empty original passage text")
    queries = {int(q): v for q, v in identity["queries"].items()}
    if len(queries) != 1700 or any(not isinstance(v, str) or not v.strip()
                                  for v in queries.values()):
        raise ValueError("Historical original query-text mapping is incomplete")
    consumed = tuple(s.query_id for s in plan.slates)
    eval_ids = tuple(plan.eval_qids)
    if len(set(consumed)) != 800 or len(set(eval_ids)) != 500:
        raise ValueError("Frozen schedule or held-out query count changed")
    if (set(consumed) & set(eval_ids) or
            not set(consumed + eval_ids).issubset(queries)):
        raise ValueError("Train/eval query-text leakage or missing mappings")
    # Disjoint IDs do not rule out identical query text in another partition.
    # Group by normalized text for the guard, but retain original text for the
    # historical tokenizer and all subsequent encoding.
    groups = {}
    for partition in ("train_qids", "cal_qids", "eval_qids"):
        qids = tuple(map(int, frozen_split[partition]))
        if len(qids) != len(set(qids)) or not set(qids).issubset(queries):
            raise ValueError("Duplicate or unknown query IDs in original split")
        normalized = [re.sub(
            r"\s+", " ", unicodedata.normalize("NFKC", queries[q]).casefold()
        ).strip() for q in qids]
        if len(normalized) != len(set(normalized)) or not all(normalized):
            raise ValueError("Duplicate or empty normalized query text within split")
        groups[partition] = set(normalized)
    if any(groups[a] & groups[b] for a, b in (
            ("train_qids", "cal_qids"),
            ("train_qids", "eval_qids"),
            ("cal_qids", "eval_qids"))):
        raise ValueError("Normalized query-text leakage across held-out partitions")
    id_set = set(pids)
    slates = {s.query_id: s.passage_ids for s in plan.slates}
    if any(len(ids) != 21 or len(set(ids)) != 21 or not set(ids).issubset(id_set)
           for ids in slates.values()):
        raise ValueError("Frozen 21-way passage-text inputs are incomplete")
    return ModelInputContract(pids, texts, queries, consumed, eval_ids, slates, digest)


def check():
    from guarded_exact_control_runner import build_plan
    plan = build_plan()
    view = load_model_inputs(plan)
    q_eval, p_eval = view.evaluation_inputs()
    if len(q_eval) != 500 or len(p_eval) != 50000:
        raise ValueError("Full-corpus source input contract incomplete")
    by_pid = view._passage_lookup
    pair_count = 0
    for slate in plan.slates:
        qtxt, dtxt = view.slate_texts(slate)
        if qtxt != view.query_texts[slate.query_id] or len(dtxt) != 21:
            raise ValueError("Training-source text alignment failed")
        if any(dtxt[j] != view.corpus_texts[by_pid[pid]]
               for j, pid in enumerate(slate.passage_ids)):
            raise ValueError("Training passage-ID-to-text alignment failed")
        pair_count += len(dtxt)
    if pair_count != 16800:
        raise ValueError("Scored-pair slate count changed")
    assert view.source_encoding_contract()["final_evaluation"]["corpus_shape"] == [50000, 768]
    try:
        view.reencode_or_rank()
    except PermissionError:
        pass
    else:
        raise AssertionError("Unauthorized model encoding unexpectedly enabled")
    return {
        "status": "SOURCE_LOCKED_TEXT_AND_REENCODING_CONTRACT_PASS_NO_MODEL_RUN",
        "training_slates": len(plan.slates),
        "ordered_training_pair_texts": pair_count,
        "eval_query_texts": len(q_eval),
        "corpus_passage_texts": len(p_eval),
        "source_slice_sha256": view.source_slice_sha256,
        "normalized_query_text_partition_overlap": 0,
        "model_forward_calls": 0, "reencoded_embeddings": 0,
        "teacher_scored_pairs": 0, "optimizer_steps": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    result = check()
    print("SOURCE_LOCKED_MODEL_IO_PREFLIGHT PASS")
    print(json.dumps(result, sort_keys=True))
