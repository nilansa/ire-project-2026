import unittest
from metrics import metrics, paired

class MetricTests(unittest.TestCase):
    def test_full_qrel_ideal_and_recall(self):
        gold={'a': 2, 'b': 1, 'c': 1}
        m=metrics(['a','b'], gold)
        self.assertAlmostEqual(m['recall10'], 2/3)
        self.assertAlmostEqual(m['ndcg10'], .878961873034099)
        self.assertEqual(m['mrr10'], 1.)
    def test_known_bootstrap_and_ties(self):
        out=paired([.2,.3,.4], [.1,.2,.3], n=10000, seed=20261116)
        self.assertAlmostEqual(out['delta'],.1)
        self.assertAlmostEqual(out['ci95'][0],.1)
        self.assertEqual(out['win_tie_loss'],[3,0,0])
    def test_mapping_corruption_is_detected(self):
        with self.assertRaises(ValueError): metrics(['a','a'],{'a':1})
        with self.assertRaises(ValueError): metrics(['a'],{'a':-1})
        with self.assertRaises(ValueError): metrics(['a'],{'a':float('nan')})

if __name__=='__main__': unittest.main()
