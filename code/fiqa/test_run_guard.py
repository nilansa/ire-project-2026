import tempfile, unittest
from pathlib import Path
from run_guard import begin_once, validate_selection

class GuardTests(unittest.TestCase):
    def test_interrupted_invocation_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'start.json'; begin_once(p,{'elapsed':0})
            with self.assertRaises(RuntimeError): begin_once(p,{'elapsed':0})
    def test_foreign_or_duplicate_selection_is_rejected(self):
        pool=[str(i) for i in range(50)]
        validate_selection(pool,{'a':pool[:10]})
        with self.assertRaises(ValueError): validate_selection(pool,{'a':pool[:9]+['foreign']})
        with self.assertRaises(ValueError): validate_selection(pool,{'a':pool[:9]+[pool[0]]})

if __name__=='__main__': unittest.main()
