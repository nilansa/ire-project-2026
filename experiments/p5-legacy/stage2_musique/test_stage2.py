import json
import tempfile
import unittest
from pathlib import Path

from stage2_adapter import assert_qrel_blind, evidence_word_count, map_internal_ids, write_beir_dataset


class AdapterTests(unittest.TestCase):
    def test_beir_ids_and_qrel_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'dataset'
            doc_map = root / 'doc_map.json'
            qids = root / 'qids.json'
            corpus = [{'doc_id': 'd-01', 'title': 'One', 'text': 'alpha beta'}, {'doc_id': 'd-02', 'title': 'Two', 'text': 'gamma'}]
            questions = [{'id': 'q-01', 'question': 'first?'}, {'id': 'q-02', 'question': 'second?'}]
            write_beir_dataset(corpus, questions, dataset, doc_map, qids)
            self.assertEqual(assert_qrel_blind(dataset / 'qrels' / 'test.tsv'), 0)
            self.assertEqual([x['_id'] for x in map(json.loads, (dataset / 'corpus.jsonl').read_text().splitlines())], ['d-01', 'd-02'])
            self.assertEqual(json.loads(doc_map.read_text()), ['d-01', 'd-02'])
            self.assertEqual(json.loads(qids.read_text()), ['q-01', 'q-02'])

    def test_qrel_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            qrels = Path(tmp) / 'test.tsv'
            qrels.write_text('query-id\tcorpus-id\tscore\nq1\td1\t1\n')
            with self.assertRaises(AssertionError):
                assert_qrel_blind(qrels)

    def test_internal_ids_preserve_original_ids(self):
        self.assertEqual(map_internal_ids([1, 0], ['abc', 'xyz']), ['xyz', 'abc'])
        with self.assertRaises(IndexError):
            map_internal_ids([2], ['abc', 'xyz'])

    def test_evidence_words_use_whitespace_split(self):
        self.assertEqual(evidence_word_count('Two words', 'and three more.'), 5)


class BootstrapTests(unittest.TestCase):
    def test_fixed_bootstrap_contract(self):
        from stage2_metrics import bootstrap_indices, paired_difference_ci
        a = bootstrap_indices(4)
        b = bootstrap_indices(4)
        self.assertEqual(a.shape, (1000, 4))
        self.assertTrue((a == b).all())
        result = paired_difference_ci([1, 0, 1, 0], [0, 0, 1, 0], a)
        self.assertAlmostEqual(result['mean_difference'], 0.25)
        self.assertEqual(len(result['ci95']), 2)

if __name__ == '__main__':
    unittest.main()
