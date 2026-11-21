import unittest
from evaluate_run import evaluate_one, validate_rankings, verify_digest
import tempfile, hashlib
from pathlib import Path

class EvaluationTests(unittest.TestCase):
    def test_planted_truth_and_corrupt_query_map(self):
        rankings={'dense':['b','c'],'coverage':['a','c'],'dense10':['b','c'],
                  'bert_maxsim':['b','c'],'rankmatched_0':['b','c'],'rankmatched_1':['b','c'],
                  'rankmatched_2':['b','c'],'uniform':['b','c'],'rerank_dense10':['b','c'],
                  'rerank_coverage':['a','c']}
        result=evaluate_one('q1',rankings,{'a':1},['a'],['a','b','c'])
        self.assertGreater(result['metrics']['coverage']['ndcg10'],result['metrics']['dense10']['ndcg10'])
        self.assertEqual(result['gained']['selected'],1)
        with self.assertRaises(ValueError): evaluate_one('q1',rankings,{},['a'],['a','b','c'])
        with self.assertRaises(ValueError): evaluate_one('q1',rankings,{'a':1},['a'],['b','c'])

    def test_input_identity_rejects_nonempty_swapped_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'qrels.json'; path.write_text('{"q1":{"a":1},"q2":{"b":1}}')
            digest=hashlib.sha256(path.read_bytes()).hexdigest(); verify_digest(path,digest)
            path.write_text('{"q1":{"b":1},"q2":{"a":1}}')
            with self.assertRaises(ValueError): verify_digest(path,digest)

    def test_incomplete_or_unknown_rankings_fail(self):
        with self.assertRaises(ValueError): validate_rankings({'coverage':['a']},{'a'})

if __name__=='__main__': unittest.main()
