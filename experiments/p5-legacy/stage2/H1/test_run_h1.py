import unittest

import numpy as np

from run_h1 import (
    facility_coverages,
    greedy_coverage_order,
    mmr_order,
    paired_bootstrap_ci,
    selection_metrics,
)


class RetrievalUnitTests(unittest.TestCase):
    def test_selection_metrics_full_support_and_whitespace_cost(self):
        result = selection_metrics([0, 2, 1], {0, 2}, [5, 7, 11], 2)
        self.assertTrue(result["full_support"])
        self.assertEqual(result["sp_recall"], 1.0)
        self.assertEqual(result["evidence_tokens"], 16)

    def test_cost_normalized_coverage_can_choose_short_passage(self):
        coverages = np.array([[4.0, 0.0], [3.0, 3.0]], dtype=np.float32)
        order = greedy_coverage_order(coverages, [8, 2], True, max_k=1)
        self.assertEqual(order, [1])

    def test_mmr_starts_at_dense_best_and_returns_unique_rows(self):
        query = np.array([1.0, 0.0], dtype=np.float32)
        vectors = np.array([[0.95, 0.31], [0.8, 0.6], [0.6, 0.8]], dtype=np.float32)
        order = mmr_order(query, vectors, np.array([8, 9, 10]), max_k=3)
        self.assertEqual(order[0], 8)
        self.assertEqual(len(order), 3)
        self.assertEqual(len(set(order)), 3)

    def test_facility_coverage_and_bootstrap_are_deterministic(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        passages = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0]], dtype=np.float32),
        ]
        coverage, costs = facility_coverages(query, passages)
        self.assertEqual(coverage.shape, (2, 1))
        self.assertEqual(costs, [1, 1])
        first = paired_bootstrap_ci([1, 0, 1], [0, 0, 0])
        second = paired_bootstrap_ci([1, 0, 1], [0, 0, 0])
        self.assertEqual(first, second)
        self.assertEqual(first["delta"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
