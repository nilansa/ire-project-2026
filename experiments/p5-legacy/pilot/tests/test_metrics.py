import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pilot import greedy_facility_order, score_selection


class MetricTests(unittest.TestCase):
    def test_support_recall_and_whitespace_tokens(self):
        result = score_selection(
            selected=[0, 2],
            gold_indices={0, 1},
            paragraphs=["Gold  paragraph", "Other gold", "A distractor here"],
        )
        self.assertEqual(result["sp_paragraph_recall"], 0.5)
        self.assertFalse(result["full_support"])
        self.assertEqual(result["evidence_tokens"], 5)

    def test_facility_cost_normalization_changes_greedy_choice(self):
        query = np.array([[1.0, 0.0]], dtype=np.float32)
        paragraphs = [
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[0.8, 0.6]], dtype=np.float32),
        ]
        cost_order, _ = greedy_facility_order(query, paragraphs, True, max_k=1)
        raw_order, _ = greedy_facility_order(query, paragraphs, False, max_k=1)
        self.assertEqual(cost_order, [1])
        self.assertEqual(raw_order, [0])


if __name__ == "__main__":
    unittest.main()
