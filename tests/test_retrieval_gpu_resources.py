"""Preflight uses actual native CUDA paths, including disposable optimizer state."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import retrieval_gpu_resources as study
import torch
from torch.nn import functional as F


class GPUResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        study.configure()

    def test_output_gradient_and_deliberate_errors(self):
        e, q, y = study.fixture(3, 8)
        for arm, cpu in study.parent.pair(3299).items():
            cpu.float().double()
            gpu = study.parent.Reader(arm).float().cuda()
            gpu.load_state_dict(cpu.state_dict())
            ref = study.parent.reference(cpu, e, q)
            actual = gpu(e.cuda(), q.cuda())
            study.check_scores(actual, ref)
            F.cross_entropy(ref, y).backward()
            F.cross_entropy(actual, y.cuda()).backward()
            for a, b in zip(cpu.parameters(), gpu.parameters()):
                torch.testing.assert_close(b.grad.cpu().double(), a.grad, atol=1e-4, rtol=1e-4)
            with self.assertRaisesRegex(AssertionError, 'Numerical tolerance'):
                study.check_scores(actual, study.parent.reference(cpu, e, q, wrong=True))
            with self.assertRaisesRegex(AssertionError, 'Numerical tolerance'):
                study.check_scores(actual + .1, ref)

    def test_payload_alias_fixture_and_exact_cache(self):
        e, q, y = study.fixture(2, 32)
        for arm, model in study.parent.pair(3299).items():
            model.float().cuda()
            with torch.inference_mode():
                state = model.prefill(e.cuda())
                self.assertEqual(study.payload_bytes(state), study.expected_payload(arm, 2, 32))
                self.assertTrue(torch.equal(model(e.cuda(), q.cuda()), model.read(state, q.cuda())))
                with self.assertRaisesRegex(AssertionError, 'Aliased state'):
                    study.payload_bytes((state[0], state[0]))
                self.assertNotEqual(study.payload_bytes(state), study.expected_payload('delta' if arm == 'softmax' else 'softmax', 2, 32))
        for events, query, label in zip(e.tolist(), q.tolist(), y.tolist()):
            self.assertEqual(dict(events)[query], label)

    def test_actual_saved_entry_timing_profile_and_audit(self):
        from audit_retrieval_gpu_resources import audit_cell
        run = study.runs()[0]
        before = study.parent.digest(study.ROOT / run['checkpoint'])
        cell, evidence = study.inference_case(run, 1, 8)
        audit_cell(cell, evidence, run)
        wrong = dict(cell, payload_bytes=cell['payload_bytes'] + 4)
        with self.assertRaisesRegex(AssertionError, 'Payload'):
            audit_cell(wrong, evidence, run)
        profile = study.training_profile(run, 8)
        self.assertEqual(profile['updates'], 1)
        self.assertGreater(profile['optimizer_state_bytes'], 0)
        self.assertGreaterEqual(profile['peak_reserved_bytes'], profile['peak_allocated_bytes'])
        self.assertEqual(study.parent.digest(study.ROOT / run['checkpoint']), before)


if __name__ == '__main__':
    unittest.main()
