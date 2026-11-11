"""Distinct-value, block-order, actual reload/audit and decision controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import history_repetition as study


class HistoryRepetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture_model(self, arm):
        model = study.study.LoadReader(arm)
        with torch.no_grad():
            model.key.weight.zero_(); model.key.weight[0, 0] = 1
            model.key.weight[1, :2] = torch.tensor([.6, .8])
            model.value.weight[0].fill_(1); model.value.weight[1].zero_()
            model.output.weight.zero_(); model.output.weight[0].fill_(1); model.output.bias.zero_()
            if arm == 'softmax':
                model.log_scale.zero_()
        return model

    def test_softmax_identity_and_incomplete_repetition_error(self):
        model = self.fixture_model('softmax')
        e = torch.tensor([[[0, 0], [1, 1]]]); q = torch.tensor([0])
        with torch.no_grad():
            native = model(e, q)
            for factor in study.FACTORS:
                torch.testing.assert_close(model(study.repeat_events(e, factor), q), native, atol=1e-12, rtol=1e-12)
            # Duplicating only one key changes its relative weighting.
            wrong = model(torch.cat((e, e[:, :1]), 1), q)
            with self.assertRaisesRegex(AssertionError, 'Repetition equations'):
                study.check_scores(wrong, native)

    def test_whole_block_order_and_delta_wrong_tiling(self):
        model = self.fixture_model('delta')
        e = torch.tensor([[[0, 0], [1, 1]]]); q = torch.tensor([0])
        expected = torch.tensor([[[0, 0], [1, 1], [0, 0], [1, 1]]])
        self.assertTrue(torch.equal(study.repeat_events(e, 2), expected))
        with torch.no_grad():
            wrong = model(e.repeat_interleave(2, dim=1), q)
            ref = model(expected, q)
            with self.assertRaisesRegex(AssertionError, 'Repetition equations'):
                study.check_scores(wrong, ref)
        with self.assertRaisesRegex(AssertionError, 'Undeclared'):
            study.repeat_events(e, 3)

    def test_initialized_saved_entry_labels_absent_groups_and_bad_scalar(self):
        rows = study.study.generate(2026100431, base_counts=dict(train=32, validation=32))['pools']['train_2'][:6]
        folder = ROOT / 'results/test_runs' / ('history_repetition_' + uuid.uuid4().hex)
        folder.mkdir(parents=True, exist_ok=False)
        for arm, model in study.study.pair(3323).items():
            cp = folder / f'{arm}.pt'; torch.save(dict(model=model.state_dict(), step=0), cp)
            for factor in study.FACTORS:
                metrics, scores = study.evaluate(model, rows, factor)
                error, _ = study.audit_cell(cp, arm, rows, factor, metrics, scores)
                self.assertLess(error, 1e-9)
                with self.assertRaises(AssertionError):
                    study.audit_cell(cp, arm, rows, factor, dict(metrics, loss=metrics['loss'] + .01), scores)
            wrong = copy.deepcopy(rows); wrong[0]['label'] = (wrong[0]['label'] + 1) % 6
            with self.assertRaisesRegex(AssertionError, 'Changed map or label'):
                study.reference_scores(model, wrong, 2)

    def test_paired_counts_and_complete_decision_boundaries(self):
        rows = [dict(label=0), dict(label=1), dict(label=2), dict(label=3)]
        native = torch.eye(6, dtype=torch.float64)[[1, 1, 2, 0]]
        repeat = torch.eye(6, dtype=torch.float64)[[0, 0, 2, 1]]
        self.assertEqual(study.paired_counts(native, repeat, rows), dict(count=4, rescued=1, harmed=1, both_correct=1, both_wrong=1))
        cells = [dict(seed=s, arm=a, pool=p, factor=f, metrics=dict(accuracy=.5), native_max_logit_change=0.)
                 for s in study.study.CONFIRM_SEEDS for a in study.study.parent.ARMS for p in study.POOLS for f in study.FACTORS]
        self.assertFalse(study.decision(cells)['delta']['2']['all_seed_10_point_gain_at_32'])
        for c in cells:
            if c['arm'] == 'delta' and c['pool'] == 'test_32' and c['factor'] == 2:
                c['metrics']['accuracy'] = .7
        self.assertTrue(study.decision(cells)['delta']['2']['all_seed_10_point_gain_at_32'])
        with self.assertRaisesRegex(AssertionError, 'Missing or duplicate'):
            study.decision(cells[:-1])
        cells[0]['native_max_logit_change'] = .01
        self.assertFalse(study.decision(cells)['softmax_logit_invariance'])


if __name__ == '__main__':
    unittest.main()
