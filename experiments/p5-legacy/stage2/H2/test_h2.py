import unittest

import numpy as np
import run_h2
import torch


class RetrievalHelpersTest(unittest.TestCase):
    def test_stable_topk_breaks_ties_by_corpus_order(self):
        self.assertEqual(run_h2.stable_topk(np.array([0.2, 0.7, 0.7, 0.1]), 2), [1, 2])
        self.assertEqual(run_h2.stable_topk(np.array([1.0, 1.0, 1.0]), 2), [0, 1])

    def test_metric_values_use_question_level_gold_set(self):
        result = run_h2.metric_values([0, 2, 1], {1, 2}, ["a b", "c", "d e f"], 2)
        self.assertEqual(result["full_support"], 0.0)
        self.assertEqual(result["sp_recall"], 0.5)
        self.assertEqual(result["evidence_tokens"], 5.0)

    def test_paired_bootstrap_is_seeded_and_keeps_pairing(self):
        left = [1.0, 0.0, 1.0, 0.0]
        right = [0.0, 0.0, 1.0, 0.0]
        result = run_h2.paired_bootstrap(left, right, resamples=1000, seed=0)
        self.assertEqual(result, run_h2.paired_bootstrap(left, right, resamples=1000, seed=0))
        self.assertEqual(result["n"], 4)
        self.assertEqual(result["difference"], 0.25)
        self.assertLessEqual(result["ci95_low"], result["difference"])
        self.assertGreaterEqual(result["ci95_high"], result["difference"])

    def test_bm25_zero_score_ties_retain_corpus_order(self):
        index = run_h2.BM25Index(["apple fruit", "banana fruit", "pear fruit"])
        self.assertEqual(index.rank("unmatchedterm", 2), [0, 1])

    def test_cost_normalized_facility_coverage_can_change_first_choice(self):
        query_tokens = np.eye(2, dtype=np.float32)
        candidates = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
            np.eye(2, dtype=np.float32),
        ]
        self.assertEqual(run_h2.facility_order(query_tokens, candidates, [1, 1, 4], 1, True), [0])
        self.assertEqual(run_h2.facility_order(query_tokens, candidates, [1, 1, 4], 1, False), [2])

    def test_ready_parser_handles_nested_dataset_hashes(self):
        digest = "a" * 64
        found = run_h2._ready_entries({"files": {"hotpotqa": {"corpus.jsonl": {"sha256": digest}}}})
        self.assertEqual(found["files/hotpotqa/corpus.jsonl"], digest)

    def test_infonce_loss_backpropagates_with_multiple_positives_and_negatives(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        passages = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]], requires_grad=True
        )
        loss = run_h2.contrastive_loss(
            torch,
            query,
            passages,
            ["a", "a2", "b", "hard"],
            [{"a", "a2"}, {"b"}],
            [{"a", "a2"}, {"b"}],
        )
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(passages.grad)


if __name__ == "__main__":
    unittest.main()
