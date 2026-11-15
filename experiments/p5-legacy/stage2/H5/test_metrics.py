import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from metrics import evaluate_dataset, paired_bootstrap_ci
from run_h5 import verify_hotpotqa_hashes


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.questions = [
            {"qid": "q1", "type": "bridge", "gold_pids": ["p1", "p2"]},
            {"qid": "q2", "type": "comparison", "gold_pids": ["p3"]},
        ]
        self.corpus = {
            "p1": {"text": "one two"},
            "p2": {"text": "three"},
            "p3": {"text": "four five six"},
        }

    def test_full_support_recall_tokens_and_paired_delta(self):
        result = evaluate_dataset(
            "hotpotqa",
            self.questions,
            self.corpus,
            {
                "disco": {"q1": ["p1", "p2"], "q2": ["p3"]},
                "dense_minilm": {"q1": ["p1", "p3"], "q2": ["p1"]},
            },
            {
                "disco": {"q1": 0.2, "q2": 0.4},
                "dense_minilm": {"q1": 0.1, "q2": 0.3},
            },
            k=2,
        )
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["methods"]["disco"]["full_support"], 1.0)
        self.assertEqual(result["methods"]["dense_minilm"]["full_support"], 0.0)
        self.assertEqual(result["methods"]["dense_minilm"]["sp_recall"], 0.25)
        self.assertEqual(result["methods"]["disco"]["evidence_tokens"], 3.0)
        delta = result["paired_differences"]["disco_minus_dense_minilm"]
        self.assertEqual(delta["full_support"]["mean_difference"], 1.0)
        self.assertEqual(delta["sp_recall"]["n"], 2)

    def test_bootstrap_is_reproducible(self):
        left = paired_bootstrap_ci([1.0, 0.0, 1.0], seed=0)
        right = paired_bootstrap_ci([1.0, 0.0, 1.0], seed=0)
        self.assertEqual(left, right)
        self.assertEqual(left["mean_difference"], 2 / 3)

    def test_ready_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "hotpotqa"
            dataset.mkdir()
            hashes = {}
            for name in ("corpus.jsonl", "questions.jsonl", "emb_minilm.f16.npy"):
                path = dataset / name
                path.write_bytes(name.encode())
                hashes[f"data/hotpotqa/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
            ready = root / "READY.json"
            ready.write_text(json.dumps({"files": hashes}), encoding="utf-8")
            result = verify_hotpotqa_hashes(ready, root)
            self.assertEqual(set(result), set(hashes_key.split("/")[-1] for hashes_key in hashes))


if __name__ == "__main__":
    unittest.main()
