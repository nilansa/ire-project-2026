import unittest
from benchmark import read_qrels, score, validate_slate, parse_qrels

class BenchmarkTests(unittest.TestCase):
    def test_recall_denominator_all_judged_positives(self):
        qrels = {'q1': {'a': 1, 'b': 1, 'c': 0}, 'q2': {'d': 1}}
        prediction = {'q1': ['a', 'c'], 'q2': ['d']}
        r = score(prediction, qrels, 100)
        self.assertEqual(r['per_query'], {'q1': 0.5, 'q2': 1.0})
        self.assertEqual(r['mean_recall'], 0.75)

    def test_missing_queries_and_duplicate_documents_fail(self):
        with self.assertRaises(ValueError): score({'q1':['a']}, {'q1':{'a':1}, 'q2':{'b':1}}, 100)
        with self.assertRaises(ValueError): score({'q1':['a','a']}, {'q1':{'a':1}}, 100)
        with self.assertRaises(ValueError): validate_slate(['a','b'], ['b','a'], [0.1,0.2])

    def test_qrel_parser_rejects_conflict_and_tracks_zeros(self):
        self.assertEqual(parse_qrels(['query-id\tcorpus-id\tscore\n','q\td\t1\n','q\te\t0\n']), {'q': {'d':1, 'e':0}})
        with self.assertRaises(ValueError): parse_qrels(['query-id\tcorpus-id\tscore\n','q\td\t1\n','q\td\t0\n'])

if __name__ == '__main__': unittest.main()

class ShardingTests(unittest.TestCase):
    def test_shards_are_disjoint_complete_and_balanced(self):
        from benchmark import shard_ids
        ids=[str(i) for i in range(1001)]
        parts=[shard_ids(ids,i,4) for i in range(4)]
        self.assertEqual(set().union(*map(set,parts)),set(ids))
        self.assertEqual(sum(map(len,parts)),len(ids))
        self.assertLessEqual(max(map(len,parts))-min(map(len,parts)),1)
        with self.assertRaises(ValueError): shard_ids(ids,4,4)

    def test_teacher_microbatch_limits_and_order(self):
        from benchmark import batched_teacher
        seen=[]
        def f(q,docs):
            seen.append(len(docs)); return [float(x) for x in docs]
        self.assertEqual(batched_teacher('q',list(range(125)),f),list(map(float,range(125))))
        self.assertLessEqual(max(seen),8)
        with self.assertRaises(ValueError): batched_teacher('q',[1],lambda q,d:[])
