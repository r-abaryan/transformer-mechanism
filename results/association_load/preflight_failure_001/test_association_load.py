"""Reusable correctness and inverse controls for the association-load design."""
import copy
import hashlib
import sys
import unittest
import uuid
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import association_load as study
from audit_association_load import audit_run, independent_selection, independent_decision
from audit_delta_retrieval import reduce_scores, check_metrics


class AssociationLoadTests(unittest.TestCase):
    def test_new_model_equations_gradients_and_cross_write(self):
        torch.set_num_threads(1)
        data = study.generate(2026100431, base_counts=dict(train=32, validation=32))
        for m in study.LOADS:
            rows = data['pools'][f'train_{m}'][:4]
            e, q, y = study.parent.tensors(rows)
            for arm, model in study.pair(3313).items():
                refmodel = copy.deepcopy(model)
                a = model(e, q); b = study.parent.reference(refmodel, e, q)
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
                F.cross_entropy(a, y).backward(); F.cross_entropy(b, y).backward()
                for p, r in zip(model.parameters(), refmodel.parameters()):
                    self.assertIsNotNone(p.grad); torch.testing.assert_close(p.grad, r.grad, atol=1e-10, rtol=1e-10)
                if arm == 'softmax':
                    changed = copy.deepcopy(model); changed.log_recency.fill_(0.)
                    self.assertGreater(float((changed(e, q) - a).detach().abs().max()), 1e-4)
                else:
                    wrong = study.parent.reference(model, e, q, wrong=True)
                    self.assertGreater(float((wrong - a).detach().abs().max()), 1e-4)
                measured, _ = study.probes(model, rows)
                self.assertEqual(measured['count'], 4)

    def test_data_labels_multiplicity_leakage_and_baselines(self):
        data = study.generate(2026100431, base_counts=dict(train=32, validation=32))
        b = study.audit_data(data)
        self.assertEqual(sum(c['count'] for c in b.values()), 1536)
        wrong = copy.deepcopy(data); wrong['pools']['train_2'][0]['label'] = (wrong['pools']['train_2'][0]['label'] + 1) % 6
        with self.assertRaisesRegex(AssertionError, 'Wrong label'):
            study.audit_data(wrong)
        wrong = copy.deepcopy(data); wrong['pools']['validation_2'][0] = copy.deepcopy(wrong['pools']['train_2'][0])
        with self.assertRaisesRegex(AssertionError, 'Family leakage'):
            study.audit_data(wrong)
        wrong = copy.deepcopy(data); row = wrong['pools']['train_2'][0]
        row['events'][0][1] = (row['events'][0][1] + 1) % 6
        with self.assertRaisesRegex(AssertionError, 'Conflicting assignment'):
            study.audit_data(wrong)

    def test_actual_initialized_reload_audit_absent_group_and_bad_scalar(self):
        rows = study.generate(2026100431, base_counts=dict(train=32, validation=32))['pools']['train_2'][:6]
        folder = ROOT / 'results/test_runs' / ('association_load_' + uuid.uuid4().hex)
        folder.mkdir(parents=True, exist_ok=False)
        for arm, model in study.pair(3313).items():
            cp = folder / f'{arm}.pt'
            torch.save(dict(model=model.state_dict(), step=0), cp)
            metrics, scores = study.parent.evaluate(model, rows)
            self.assertNotIn('unique', metrics)
            check_metrics(metrics, reduce_scores(scores, rows))
            run = dict(seed=3313, arm=arm, phase='development', recipe='constant',
                       checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=study.parent.digest(cp),
                       metrics={'train_2': metrics})
            self.assertEqual(audit_run(run, dict(pools={'train_2': rows}), expected_steps=0)[0]['predictions'], 6)
            bad = copy.deepcopy(run); bad['metrics']['train_2']['all']['loss'] += .01
            with self.assertRaises(AssertionError):
                audit_run(bad, dict(pools={'train_2': rows}), expected_steps=0)

    def test_selection_and_per_load_decision_controls(self):
        metrics = {f'{split}_{m}': {'all': dict(count=100, correct=100, accuracy=1., loss=.02)}
                   for split in ('train', 'validation', 'test') for m in study.LOADS}
        runs = [dict(phase='development', seed=s, arm=a, recipe=r, status='complete', metrics=copy.deepcopy(metrics))
                for s in study.DEV_SEEDS for a in study.parent.ARMS for r in study.RECIPES]
        self.assertEqual(independent_selection(runs), {a: c['selected_recipe'] for a, c in study.select(runs).items()})
        self.assertEqual(study.select(runs)['delta']['selected_recipe'], 'constant')
        for r in runs:
            if r['arm'] == 'delta':
                r['metrics']['train_2']['all'].update(correct=98, accuracy=.98)
        self.assertIsNone(study.select(runs)['delta']['selected_recipe'])
        self.assertIsNone(study.select([])['delta']['selected_recipe'])
        confirm = [dict(phase='confirmation', seed=s, arm=a, metrics=copy.deepcopy(metrics))
                   for s in study.CONFIRM_SEEDS for a in study.parent.ARMS]
        self.assertEqual(study.decision(confirm), independent_decision(confirm))
        confirm[-1]['metrics']['train_32']['all'].update(correct=98, accuracy=.98)
        self.assertFalse(study.decision(confirm)['delta']['loads']['32']['all_seed_training_fit'])
        self.assertTrue(study.decision(confirm)['delta']['low_load_control'])


if __name__ == '__main__':
    unittest.main()
