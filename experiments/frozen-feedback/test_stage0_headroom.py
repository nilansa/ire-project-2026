import unittest
from stage0_headroom import compute, observed_ndcg, oracle_reorder


class OfflineHeadroomTest(unittest.TestCase):
    def test_permutation_only_does_not_admit(self):
        docs = ['irrelevant', 'relevant']
        qrels = {'relevant': 2}
        self.assertEqual(set(oracle_reorder(docs, qrels, 2)), set(docs))
        self.assertGreater(observed_ndcg(oracle_reorder(docs, qrels, 2), qrels, 2),
                           observed_ndcg(docs, qrels, 2))

    def test_rank_and_admission_headroom(self):
        rows = [
            {'query_id': 'q1', 'ranked_doc_ids': ['d0', 'd1', 'd2', 'd3']},
            {'query_id': 'q2', 'ranked_doc_ids': ['e0', 'e1', 'e2', 'e3']},
        ]
        qrels = [
            {'query_id': 'q1', 'doc_id': 'd2', 'relevance': 3},
            {'query_id': 'q1', 'doc_id': 'd3', 'relevance': 2},
            {'query_id': 'q2', 'doc_id': 'e0', 'relevance': 1},
            {'query_id': 'q2', 'doc_id': 'e1', 'relevance': 3},
        ]
        result = compute(rows, qrels, k=2, b=4, bootstrap=100)
        self.assertEqual(result['counts']['queries_with_promotable_judged_positive'], 1)
        self.assertGreater(result['mean_over_queries_with_positive_qrels']['order_gain'], 0)
        self.assertGreater(result['mean_over_queries_with_positive_qrels']['admission_gain_over_order'], 0)
        self.assertEqual(result['counts']['queries'], 2)

    def test_duplicate_candidate_rejected(self):
        with self.assertRaises(ValueError):
            compute([{'query_id': 'q1', 'ranked_doc_ids': ['d0', 'd0']}], [], k=1, b=2,
                    bootstrap=100)


if __name__ == '__main__':
    unittest.main()
