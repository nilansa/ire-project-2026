import unittest

import numpy as np

import h3


class H3Tests(unittest.TestCase):
    def test_support_and_recall(self):
        self.assertEqual(h3.support_metrics({"a", "b"}, ["a", "b", "c"]), (1, 1.0))
        self.assertEqual(h3.support_metrics({"a", "b"}, ["a"]), (0, 0.5))

    def test_empty_gold_is_rejected(self):
        with self.assertRaises(ValueError):
            h3.support_metrics([], ["a"])

    def test_threshold_targets_precision_and_maximizes_recall(self):
        result = h3.select_threshold(np.array([1, 1, 0, 0]), np.array([.9, .8, .7, .1]))
        self.assertTrue(result["reached"])
        self.assertAlmostEqual(result["threshold"], .8)
        self.assertEqual(result["tune_calls"], 2)

    def test_threshold_unavailable_returns_no_threshold(self):
        result = h3.select_threshold(np.array([1, 0, 0]), np.array([.2, .9, .8]))
        self.assertFalse(result["reached"])
        self.assertIsNone(result["threshold"])

    def test_fsr_denominator_is_truly_insufficient_sets(self):
        rate = h3.false_sufficient_rate(np.array([1, 0, 0, 1]), np.array([1, 1, 0, 0]))
        self.assertAlmostEqual(rate, .5)
        self.assertIsNone(h3.false_sufficient_rate(np.array([1, 1]), np.array([1, 0])))

    def test_disco_greedy_selects_collective_coverage(self):
        candidates = np.array([0, 1, 2])
        dense_scores = np.array([.9, .8, .7])
        docs = np.array([[.8, .1], [.1, .8], [.7, .7]], dtype=np.float32)
        tokens = np.eye(2, dtype=np.float32)
        order, gains = h3.disco_greedy(candidates, dense_scores, tokens, docs, k=2)
        self.assertEqual(order.tolist(), [2, 0])
        self.assertGreater(float(gains[0]), float(gains[1]))

    def test_ready_hashes_reads_nested_entries(self):
        digest = "a" * 64
        result = h3.ready_hashes({"files": {"hotpotqa": {"corpus.jsonl": digest}}})
        self.assertEqual(result["files/hotpotqa/corpus.jsonl"], digest)

    def test_whitespace_token_count_includes_title(self):
        self.assertEqual(h3.whitespace_tokens("New York", "has 3 boroughs"), 5)

    def test_sampling_is_deterministic_and_capped(self):
        rows = [{"qid": str(i), "type": "bridge" if i % 2 else "comparison"} for i in range(40)]
        self.assertEqual(h3.sample_split(rows, 12), h3.sample_split(rows, 12))
        selected = h3.sample_split(rows, 12)
        self.assertEqual(len(selected), 12)
        self.assertEqual({r["type"] for r in selected}, {"bridge", "comparison"})


if __name__ == "__main__":
    unittest.main()
