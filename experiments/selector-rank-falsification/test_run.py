import unittest

from run import (
    perturb_ranking,
    select_random_ids,
    select_rank_positions,
    select_stable_ids,
    shared_fraction,
)


class RankFalsificationTests(unittest.TestCase):
    def setUp(self):
        self.candidates = tuple(f"D{i:03}" for i in range(200))

    def test_identity_baseline_is_one(self):
        selected = select_rank_positions(self.candidates, 20, 73)
        self.assertEqual(shared_fraction(selected, selected, 20), 1.0)

    def test_rank_selector_is_seeded_and_rank_sensitive(self):
        first = select_rank_positions(self.candidates, 20, 73)
        self.assertEqual(first, select_rank_positions(self.candidates, 20, 73))
        perturbed = perturb_ranking(self.candidates, 1.0, 91)
        selected_after = select_rank_positions(perturbed, 20, 73)
        self.assertLess(shared_fraction(first, selected_after, 20), 1.0)
        self.assertEqual(set(perturbed), set(self.candidates))

    def test_stable_id_control_ignores_rank_order(self):
        reordered = perturb_ranking(self.candidates, 1.0, 91)
        selected = select_stable_ids(self.candidates, 20, 73)
        self.assertEqual(selected, select_stable_ids(reordered, 20, 73))
        self.assertEqual(shared_fraction(selected, selected, 20), 1.0)

    def test_random_identity_sample_is_seeded_and_unique(self):
        selected = select_random_ids(self.candidates, 20, 42)
        self.assertEqual(selected, select_random_ids(self.candidates, 20, 42))
        self.assertEqual(len(set(selected)), 20)

    def test_invalid_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            select_rank_positions(("D1", "D1"), 1, 1)
        with self.assertRaisesRegex(ValueError, "strength"):
            perturb_ranking(self.candidates, 1.1, 1)


if __name__ == "__main__":
    unittest.main()
