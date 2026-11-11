"""Real saved confirmation path, independent audit and lineage/error controls."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import confirm_full_pool_fitting as study
import audit_full_pool_fitting_confirmation as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_saved_confirmation_and_faults(self):
        out = ROOT/'results/test_runs'/('full_fit_confirm_'+uuid.uuid4().hex)
        pools = study.data_source.generate(2026100598, confirmation=True,
                    base_counts=dict(train=2, validation=1, test=1))['pools']
        rng = torch.get_rng_state().clone()
        study.execute(out, (107,), pools, warm_steps=4, budget=4, ratio=4, interval=2, fixture=True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(audit.audit(out, persist=False)['passed'])
        report_path = out/'report.json'
        original = report_path.read_bytes()
        report = json.loads(original)
        faults = (
            lambda r: r['warmups'][0].__setitem__('source_step', 1),
            lambda r: r['runs'][0].__setitem__('source_step', 5),
            lambda r: r['cells'].pop(),
            lambda r: r['counts']['warmup'].__setitem__('answer_targets', 1),
            lambda r: r['decision']['lbfgs'].__setitem__('all_seed_fit', not r['decision']['lbfgs']['all_seed_fit']))
        for fault in faults:
            wrong = copy.deepcopy(report)
            fault(wrong)
            study.replay.write(report_path, wrong)
            with self.assertRaises(AssertionError):
                audit.audit(out, persist=False)
            report_path.write_bytes(original)
        score_path = out/'scores.pt'
        raw = score_path.read_bytes()
        for field in ('labels', 'logits'):
            scores = torch.load(score_path, weights_only=True)
            scores[0][field].flatten()[0] += 1
            torch.save(scores, score_path)
            wrong = copy.deepcopy(report)
            wrong['scores_sha256'] = study.replay.digest(score_path)
            study.replay.write(report_path, wrong)
            with self.assertRaises(AssertionError):
                audit.audit(out, persist=False)
            score_path.write_bytes(raw)
            report_path.write_bytes(original)
        warm_trace = ROOT/report['warmups'][0]['trace']
        raw = warm_trace.read_bytes()
        trace = torch.load(warm_trace, weights_only=True)
        trace[0]['gradients']['write_logit'].add_(1)
        torch.save(trace, warm_trace)
        wrong = copy.deepcopy(report)
        wrong['warmups'][0]['trace_sha256'] = study.replay.digest(warm_trace)
        study.replay.write(report_path, wrong)
        with self.assertRaises(AssertionError):
            audit.audit(out, persist=False)
        warm_trace.write_bytes(raw)
        report_path.write_bytes(original)
        self.assertTrue(audit.audit(out, persist=False)['passed'])

    def test_data_controls_and_supplied_scores(self):
        pools = study.data_source.generate(2026100597, confirmation=True,
                    base_counts=dict(train=2, validation=1, test=1))['pools']
        self.assertEqual(audit.check_data(pools, True), 16)
        wrong = copy.deepcopy(pools)
        wrong['train_32'][0]['label'] = (wrong['train_32'][0]['label']+1) % 6
        with self.assertRaises(AssertionError):
            audit.check_data(wrong, True)
        wrong = copy.deepcopy(pools)
        wrong['validation_32'][0] = copy.deepcopy(wrong['train_32'][0])
        with self.assertRaises(AssertionError):
            audit.check_data(wrong, True)
        wrong = copy.deepcopy(pools)
        wrong.pop('test_8')
        with self.assertRaises(AssertionError):
            audit.check_data(wrong, True)
        z = torch.zeros(1, 2, 6, dtype=torch.float64)
        y = torch.tensor([[0, 1]])
        repeat = torch.zeros_like(y, dtype=torch.bool)
        supplied = audit.numerical.reference.metrics(z, y, repeat)
        self.assertEqual(supplied['groups']['all']['accuracy'], .5)
        self.assertEqual(supplied['complete_maps']['correct'], 0)
        self.assertIsNone(supplied['groups']['repeat']['ce'])


if __name__ == '__main__':
    unittest.main()
