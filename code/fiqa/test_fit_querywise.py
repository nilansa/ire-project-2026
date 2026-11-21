import unittest
import torch
import torch.nn.functional as F
from fit_querywise import fit, objective

class QuerywiseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.x = F.normalize(torch.tensor([[1., 0., 0.], [0., 1., 0.]]), dim=-1)
        self.z = F.normalize(torch.tensor([[[1., 0., 0.], [0., 1., 0.]],
                                            [[0., 1., 0.], [0., 0., 1.]]]), dim=-1)
        self.r = torch.tensor([[0., 1.], [1., 0.]])

    def test_loss_is_sum_over_documents_then_mean_over_queries(self):
        v = torch.zeros_like(self.x, requires_grad=True)
        p = (self.r / .02).softmax(-1)
        logp = (torch.einsum('bd,bkd->bk', self.x, self.z) / .05).log_softmax(-1)
        expected = (p * (p.clamp_min(1e-30).log() - logp)).sum(-1).mean()
        self.assertTrue(torch.allclose(objective(self.x, self.z, self.r, v), expected, atol=1e-6))

    def test_batched_fit_matches_fresh_single_query_adam(self):
        batched = fit(self.x, self.z, self.r, steps=40)
        separate = torch.cat([fit(self.x[i:i+1], self.z[i:i+1], self.r[i:i+1], steps=40) for i in range(2)])
        self.assertTrue(torch.allclose(batched, separate, atol=2e-5), (batched, separate))
        changed = self.r.clone(); changed[1] = changed[1].flip(0)
        self.assertTrue(torch.allclose(fit(self.x, self.z, changed, steps=40)[0], batched[0], atol=1e-7))

    def test_zero_teacher_with_identical_documents_does_not_move(self):
        z = self.z[:1, :1].repeat(1, 2, 1)
        self.assertEqual(float(fit(self.x[:1], z, torch.zeros(1,2), steps=30).abs().max()), 0.)

    def test_planted_teacher_moves_toward_selected_document(self):
        v = fit(self.x[:1], self.z[:1], self.r[:1], steps=80)
        before = self.x[0] @ self.z[0,1]
        after = F.normalize(self.x[:1] + .5*v, dim=-1)[0] @ self.z[0,1]
        self.assertGreater(float(after), float(before))

if __name__ == '__main__': unittest.main()
