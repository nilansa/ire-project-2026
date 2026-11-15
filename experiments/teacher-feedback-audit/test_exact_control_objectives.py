#!/usr/bin/env python3
"""CPU-only parity against archived original teacher runner; no model/training."""

import ast
import csv
import io
import json
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exact_control_objectives import ARCHIVE, query_loss, source_locked_execution_spec


class TestRealArchivedTeacherLossParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = source_locked_execution_spec()
        assert spec["status"] == "SOURCE_LOCKED_LOSS_RECIPE_ONLY_NO_MODEL_RUN"
        with zipfile.ZipFile(ARCHIVE) as outer:
            with zipfile.ZipFile(io.BytesIO(outer.read("inputs/identity-and-search-evidence.zip"))) as ids:
                identity = json.loads(ids.read("marco/identity.json"))
            with zipfile.ZipFile(io.BytesIO(outer.read("inputs/ire-audit-evidence.zip"))) as z:
                script = z.read("marco/teacher_sanity.py").decode("utf-8")
                source_tree = ast.parse(script)
                # Execute ONLY the two pure archived reference functions.
                nodes = [n for n in source_tree.body if isinstance(n, ast.FunctionDef)
                         and n.name in {"distribution", "soft_loss"}]
                assert len(nodes) == 2
                code = compile(ast.Module(body=nodes, type_ignores=[]), "original_teacher_functions", "exec")
                ns = {"np": np, "F": F}
                exec(code, ns)
                cls.original_distribution = staticmethod(ns["distribution"])
                cls.original_soft_loss = staticmethod(ns["soft_loss"])
                split = json.loads(z.read("marco/runtime/split.json"))
                consumed = json.loads(z.read("marco/teacher_sanity/train_hnsw.json"))[
                    "training_query_ids"
                ]
                neg = np.load(io.BytesIO(z.read("marco/runtime/neg_hnsw.npy")))
                rows = list(csv.DictReader(io.StringIO(
                    z.read("marco/teacher_sanity/teacher_scores.csv").decode("utf-8"))))
        cls.cached = {(int(row["query_id"]), int(row["passage_id"])):
                      float(row["raw_teacher_logit"]) for row in rows}
        cls.raw_slates = []
        pids = list(map(int, identity["pids"]))
        valid = set(pids)
        qrels = {int(q): set(map(int, ps)) for q, ps in identity["qrels"].items()}
        train = list(map(int, split["train_qids"]))
        train_position = {qid: i for i, qid in enumerate(train)}
        assert len(consumed) == 800 and len(set(consumed)) == 800
        for qid in consumed[:8]:
            i = train_position[int(qid)]
            positive = sorted(qrels[qid] & valid)[0]
            slate = [positive] + [pids[int(j)] for j in neg[i]]
            assert len(slate) == len(set(slate)) == 21
            cls.raw_slates.append(np.asarray([cls.cached[(qid, pid)] for pid in slate],
                                                dtype=np.float64))

    def test_teacher_soft_parity_on_real_saved_logits(self):
        for raw in self.raw_slates:
            # Student test values are deterministic tensors, not model inference.
            student = torch.linspace(-3.0, 2.0, 21, dtype=torch.float32).requires_grad_()
            original_target = torch.tensor(self.original_distribution(raw), dtype=torch.float32)
            old = self.original_soft_loss(student, original_target)
            new = query_loss(student, objective="teacher_soft",
                             teacher_raw_logits=torch.tensor(raw, dtype=torch.float64))
            self.assertTrue(torch.allclose(old, new, rtol=0, atol=2e-6))
            old_grad = torch.autograd.grad(old, student, retain_graph=True)[0]
            new_grad = torch.autograd.grad(new, student)[0]
            self.assertTrue(torch.allclose(old_grad, new_grad, rtol=0, atol=2e-6))
            self.assertTrue(torch.allclose(new_grad,
                                          torch.softmax(student.detach(), 0) - original_target,
                                          rtol=0, atol=2e-6))

    def test_eight_consumed_query_losses_preserve_archived_batch_mean(self):
        student = torch.linspace(-1, 1, 21, dtype=torch.float32)
        old_losses, new_losses = [], []
        for raw in self.raw_slates:
            teacher = torch.tensor(self.original_distribution(raw), dtype=torch.float32)
            old_losses.append(self.original_soft_loss(student, teacher))
            new_losses.append(query_loss(student, objective="teacher_soft",
                                         teacher_raw_logits=torch.tensor(raw)))
        self.assertEqual(len(old_losses), 8)
        self.assertTrue(torch.allclose(torch.stack(old_losses).mean(),
                                       torch.stack(new_losses).mean(), rtol=0, atol=2e-6))

    def test_hard_positive_matches_ce_reference_and_gradient(self):
        student = torch.linspace(0, 2, 21, dtype=torch.float32).requires_grad_()
        actual = query_loss(student, objective="hard_positive")
        expected = F.cross_entropy(student.reshape(1, 21),
                                   torch.tensor([0], dtype=torch.long))
        self.assertTrue(torch.allclose(actual, expected, rtol=0, atol=1e-7))
        gradient = torch.autograd.grad(actual, student)[0]
        one_hot = torch.zeros(21)
        one_hot[0] = 1
        self.assertTrue(torch.allclose(gradient,
                                      torch.softmax(student.detach(), 0) - one_hot,
                                      rtol=0, atol=2e-7))

    def test_guardrails_reject_invalid_slates_and_leakage(self):
        valid = torch.arange(21, dtype=torch.float32)
        with self.assertRaises(ValueError):
            query_loss(valid[:20], objective="hard_positive")
        with self.assertRaises(ValueError):
            query_loss(valid, objective="hard_positive", teacher_raw_logits=valid)
        with self.assertRaises(ValueError):
            query_loss(valid, objective="teacher_soft")
        with self.assertRaises(ValueError):
            query_loss(valid, objective="teacher_soft", teacher_raw_logits=valid[:20])
        with self.assertRaises(ValueError):
            query_loss(valid, objective="teacher_soft",
                       teacher_raw_logits=torch.full((21,), float("nan")))
        with self.assertRaises(ValueError):
            query_loss(valid, objective="unknown")


if __name__ == "__main__":
    unittest.main()
