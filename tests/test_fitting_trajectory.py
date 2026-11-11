"""Real saved evaluator/audit, decoder counterexamples and intentional faults."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
import fitting_trajectory as s
from audit_fitting_trajectory import audit, independent_summary
import association_load


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)

    def test_margins_and_decoding_controls(self):
        c = torch.tensor([[[0., 1., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0.]]], dtype=torch.float64)
        z = c[..., [1, 0, 2, 3, 4, 5]]; y = torch.tensor([[0, 0]]); repeat = torch.zeros_like(y, dtype=torch.bool)
        initial = dict(coefficients=c, logits=z)
        result = s.summarize_cell(c, z, y, repeat, initial)
        s.prior.compare(result, independent_summary(c, z, y, repeat, initial))
        self.assertEqual(result['groups']['all']['coefficient_only'], 1)
        self.assertEqual(result['groups']['all']['native_only'], 1)
        self.assertIsNone(result['groups']['repeat']['coefficient_accuracy'])
        _, unit = s.margins(c, y)
        _, scaled = s.margins(c*100+7, y)
        torch.testing.assert_close(unit, scaled)
        zero = torch.zeros_like(c)
        raw, unit = s.margins(zero, y)
        self.assertTrue(torch.equal(raw, zero[..., 0]) and torch.equal(unit, raw))
        tie_y = torch.tensor([[0, 1]])
        tied = s.summarize_cell(zero, zero, tie_y, repeat, dict(coefficients=zero, logits=zero))
        self.assertEqual(tied['native']['groups']['all']['accuracy'], .5)
        # Strict competing signatures, including the deliberately unresolved case.
        for gains, verdict in [([1., 1., 1.], 'broad'), ([1., -1., -1.], 'training_only'), ([1., 1., -1.], 'mixed_or_unresolved')]:
            cells = []
            for split, gain in zip(s.SPLITS, gains):
                for arm, value in zip(s.prior.ARMS, [gain, 0.]):
                    cells.append(dict(seed=1, arm=arm, stage=1, pool=split+'_32', summary=dict(groups=dict(all=dict(
                        coefficient_unit_margin=dict(mean=value), coefficient_accuracy=.5, native_unit_margin=dict(mean=0.))))))
            self.assertEqual(s.decisions(cells)['verdict'], verdict)

    def test_saved_runtime_and_faults(self):
        folder = ROOT/'results/test_runs'/('fitting_trajectory_'+uuid.uuid4().hex); folder.mkdir(parents=True)
        with torch.random.fork_rng(devices=[]): reader = association_load.pair(151)['delta']
        runs = []
        for i, arm in enumerate(s.prior.ARMS):
            initial = folder/(arm+'_0.pt'); final = folder/(arm+'_1.pt')
            torch.save(dict(model=reader.state_dict()), initial)
            state = copy.deepcopy(reader.state_dict()); state['key.weight'][0].add_(.03*(i+1))
            state['output.bias'][0] += .04*(i+1)
            torch.save(dict(model=state), final)
            runs.append(dict(seed=151, arm=arm, snapshots=[dict(step=0, checkpoint=initial.relative_to(ROOT).as_posix()),
                                                         dict(step=1, checkpoint=final.relative_to(ROOT).as_posix())]))
        pools = association_load.generate(151, confirmation=True, base_counts=dict(train=1, validation=1, test=1))['pools']
        out = folder/'study'; rng = torch.get_rng_state().clone()
        s.execute(out, runs, pools, fixture=True)
        self.assertTrue(audit(out, persist=False)['passed'])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        report_path = out/'report.json'; report_bytes = report_path.read_bytes(); report = json.loads(report_bytes)
        path = out/'scores.pt'; original = path.read_bytes()
        for field in ['coefficients', 'logits', 'labels', 'missing_cell']:
            scores = torch.load(path, weights_only=True)
            if field == 'missing_cell': scores.pop()
            else: scores[0][field].flatten()[0] += 1
            torch.save(scores, path); wrong = copy.deepcopy(report)
            wrong['scores_sha256'] = s.prior.replay.digest(path); s.prior.replay.write(report_path, wrong)
            with self.assertRaises(AssertionError, msg=field): audit(out, persist=False)
            path.write_bytes(original); report_path.write_bytes(report_bytes)
        for change in [lambda r: r['cells'][0]['summary']['groups'].pop('repeat'),
                       lambda r: r['cells'][0]['summary']['groups']['all'].__setitem__('coefficient_only', 999),
                       lambda r: r.__setitem__('evaluation_targets', r['evaluation_targets']+1),
                       lambda r: r['decision'].__setitem__('verdict', 'wrong')]:
            wrong = copy.deepcopy(report); change(wrong); s.prior.replay.write(report_path, wrong)
            with self.assertRaises((AssertionError, KeyError)): audit(out, persist=False)
            report_path.write_bytes(report_bytes)
        csv_path = out/'trajectory.csv'; csv_bytes = csv_path.read_bytes()
        csv_path.write_bytes(csv_bytes.replace(b'151', b'152', 1))
        wrong = copy.deepcopy(report); wrong['csv_sha256'] = s.prior.replay.digest(csv_path)
        s.prior.replay.write(report_path, wrong)
        with self.assertRaisesRegex(AssertionError, 'CSV mismatch'): audit(out, persist=False)
        csv_path.write_bytes(csv_bytes); report_path.write_bytes(report_bytes)


if __name__ == '__main__': unittest.main()
