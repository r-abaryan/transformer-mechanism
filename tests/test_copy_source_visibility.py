"""Initialized contracts and real saved evaluator/audit entry, isolated fixtures."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from copy_source_visibility import DIRECTORY, MODES, make_model, forward, metrics, produce, digest, write
from audit_copy_source_visibility import reference, scalar_metrics, close, audit
from encoder_decoder_copy import logits

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


class VisibilityTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(71)
        return make_model('rope').eval()

    def test_equations_gradients_masks_and_marker(self):
        model = self.model()
        parameters = tuple(model.parameters())
        for n in (6, 8, 10):
            rows = [[3+i % 8 for i in range(n)], [4]*n]
            x = torch.tensor(rows)
            for mode in MODES:
                actual = forward(model, x, mode)
                expected = reference(model, rows, mode)
                for k in actual:
                    self.assertLessEqual(float((actual[k]-expected[k]).detach().abs().max()), 1e-9)
                ga = torch.autograd.grad(F.cross_entropy(actual['logits'], x[:, 0]), parameters)
                gb = torch.autograd.grad(F.cross_entropy(expected['logits'], x[:, 0]), parameters)
                self.assertLessEqual(max(float((a-b).abs().max()) for a, b in zip(ga, gb)), 1e-9)
            native = forward(model, x, 'E1M1')['logits']
            self.assertLessEqual(float((native-logits(model, x, torch.ones(2, 1, dtype=torch.long))[:, 0]).detach().abs().max()), 1e-9)
            hidden = forward(model, x, 'E0M0')
            gap = reference(model, rows, 'short', source_positions=list(range(6))+[n])
            self.assertLessEqual(float((hidden['logits']-gap['logits']).detach().abs().max()), 1e-9)
            if n == 6:
                for mode in MODES[:4]:
                    self.assertLessEqual(float((forward(model, x, mode)['logits']-native).detach().abs().max()), 1e-9)
            else:
                for sign in (-20., 20.):
                    perturbed = forward(model, x, 'E0M0', perturb=sign)
                    self.assertLessEqual(float((perturbed['logits']-hidden['logits']).detach().abs().max()), 1e-9)
                    unmasked = forward(model, x, 'E1M0', perturb=sign)
                    self.assertGreater(float((unmasked['logits']-hidden['logits']).detach().abs().max()), 1e-6)
                self.assertEqual(float(hidden['weights'][..., 6:n].detach().abs().max()), 0.)
                exposed = forward(model, x, 'E0M1')
                self.assertGreater(float(exposed['weights'][..., 6:n].detach().max()), 0.)
                self.assertTrue(torch.equal(hidden['k'], exposed['k']))
                self.assertTrue(torch.equal(hidden['v'], exposed['v']))

    def test_supplied_logits_ties_and_missing_group(self):
        rows = [[3]*10, [4]*10]
        z = torch.zeros(2, 16, dtype=torch.float64)
        z[0, 3], z[1, 4] = 10., 10.
        saved = dict(logits=z, weights=torch.ones(2, 2, 1, 11, dtype=torch.float64)/11)
        good = metrics(rows, saved)
        close(good, scalar_metrics(rows, saved))
        self.assertEqual(good['all']['accuracy'], 1.)
        self.assertEqual(good['distinct']['count'], 0)
        self.assertIsNone(good['distinct']['ce'])
        z[0, 0] = 10.  # Tie: first index wins, so this answer is wrong.
        wrong = metrics(rows, saved)
        close(wrong, scalar_metrics(rows, saved))
        self.assertEqual(wrong['all']['accuracy'], .5)
        self.assertNotEqual(wrong, good)

    def test_saved_producer_audit_and_deliberate_corruptions(self):
        folder = ROOT / 'results/test_runs' / ('copy_visibility_' + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        cp = folder / 'initialized.pt'
        torch.save({'model': self.model().state_dict()}, cp)
        pools = {'test_6': [[3,4,5,6,7,8], [3]*6],
                 'test_8': [[3,4,5,6,7,8,9,10], [4]*8],
                 'test_10': [[3,4]*5, [4]*10]}
        write(folder / 'lock.json', dict(runs=[dict(seed=71, checkpoint=cp.relative_to(ROOT).as_posix(), checkpoint_sha256=digest(cp))],
              pools=pools, modes=list(MODES), protected={}, archive=None))
        report = produce(folder)
        result = audit(folder)
        self.assertTrue(result['passed'])
        self.assertEqual(result['cells'], 18)
        self.assertEqual(result['outputs'], 36)
        # Reach numerical/structural audit checks, not merely stale hash rejection.
        for defect in ('missing_cell', 'group', 'decision', 'label', 'logits', 'weights'):
            broken = copy.deepcopy(report)
            original_lock = (folder / 'lock.json').read_bytes()
            evidence_path = folder / 'evidence.pt'
            original_evidence = evidence_path.read_bytes()
            if defect == 'missing_cell':
                broken['cells'].pop()
            elif defect == 'group':
                broken['cells'][0]['metrics']['distinct']['count'] += 1
            elif defect == 'decision':
                broken['decision']['primary_groups'] += 1
            elif defect == 'label':
                lock = json.loads(original_lock)
                lock['pools']['test_8'][0][0] = 11
                write(folder / 'lock.json', lock)
                broken['lock_sha256'] = digest(folder / 'lock.json')
            else:
                saved = torch.load(evidence_path, weights_only=True)
                if defect == 'logits':
                    saved['71/test_8/E0M0']['logits'][0, 0] += 1.
                else:
                    w = saved['71/test_8/E0M0']['weights']
                    w[..., 6] = .5
                torch.save(saved, evidence_path)
                broken['evidence_sha256'] = digest(evidence_path)
            write(folder / 'report.json', broken)
            try:
                with self.assertRaises(AssertionError, msg=defect):
                    audit(folder, save=False)
            finally:
                (folder / 'lock.json').write_bytes(original_lock)
                evidence_path.write_bytes(original_evidence)
                write(folder / 'report.json', report)
        self.assertTrue(audit(folder, save=False)['passed'])


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(VisibilityTests))
    if result.wasSuccessful():
        DIRECTORY.mkdir(exist_ok=True)
        destination = DIRECTORY / 'preflight.json'
        assert not destination.exists()
        write(destination, dict(passed=True, methods=result.testsRun, initialized_gradient_cases=18,
              saved_evaluator_audit=True, intentional_defects=6, excluded_input_perturbation_signs=[-20,20],
              tests_sha256=digest(Path(__file__))))
    sys.exit(not result.wasSuccessful())
