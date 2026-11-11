"""Penalty derivative, actual evaluator/audit and selection/decision controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import overlap_regularization as study


class OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_penalty_independent_value_gradient_and_direction(self):
        model=study.study.pair(3407)['delta']
        raw=model.key.weight;loss=study.penalty(model)
        actual=torch.autograd.grad(loss,raw)[0]
        with torch.no_grad():
            norm=raw.square().sum(-1,keepdim=True).sqrt();u=raw/norm
            gram=u@u.T;gram.fill_diagonal_(0)
            gu=4*(gram@u)/(32*31)
            expected=(gu-(gu*u).sum(-1,keepdim=True)*u)/norm
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
        self.assertAlmostEqual(float(loss.detach()),study.independent_penalty(model),places=12)
        d=torch.randn_like(raw);d=d/d.norm();original=raw.detach().clone();eps=1e-6
        with torch.no_grad():
            raw.copy_(original+eps*d);plus=float(study.penalty(model))
            raw.copy_(original-eps*d);minus=float(study.penalty(model));raw.copy_(original)
        self.assertAlmostEqual((plus-minus)/(2*eps),float((actual*d).sum()),places=8)
        with self.assertRaises(AssertionError):torch.testing.assert_close(actual,torch.zeros_like(actual),atol=1e-12,rtol=1e-10)

    def test_nonzero_floor_and_wrong_diagonal_control(self):
        model=study.study.LoadReader('delta')
        with torch.no_grad():model.key.weight.copy_(torch.cat((torch.eye(16),torch.eye(16)),0))
        self.assertAlmostEqual(float(study.penalty(model).detach()),1/31,places=12)
        k=model.keys(torch.arange(32));wrong=(k@k.T).square().mean()
        self.assertGreater(abs(float(wrong.detach())-1/31),.01)
        with torch.no_grad():model.key.weight[16:].neg_()
        self.assertAlmostEqual(float(study.penalty(model).detach()),1/31,places=12)

    def test_initialized_checkpoint_actual_audit_and_bad_metrics(self):
        rows=[dict(events=[[k,k%6] for k in range(32)],query=7,label=1,load=32,group='unique')]
        folder=ROOT/'results/test_runs'/('overlap_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
        model=study.study.pair(3407)['delta'];cp=folder/'model.pt';torch.save(dict(model=model.state_dict(),step=0),cp)
        for allq in (False,True):
            metrics,scores=study.metrics(model,rows,allq);study.audit_cell(cp,rows,allq,metrics,scores)
            bad=copy.deepcopy(metrics)
            (bad['all_queries'] if allq else bad)['loss']+=.01
            with self.assertRaises(AssertionError):study.audit_cell(cp,rows,allq,bad,scores)
        wrong=copy.deepcopy(rows);wrong[0]['label']=2
        metrics,scores=study.metrics(model,rows)
        with self.assertRaises(AssertionError):study.audit_cell(cp,wrong,False,metrics,scores)

    def test_selection_eligibility_and_declared_order(self):
        runs=[]
        for s in study.DEV_SEEDS:
            for w in study.WEIGHTS:
                metrics={f'{split}_{m}':dict(accuracy=1.) for split in ('train','validation') for m in study.study.LOADS}
                metrics['all_validation_32']=dict(all_queries=dict(accuracy=.5+w*.02,loss=1.))
                runs.append(dict(seed=s,weight=w,metrics=metrics))
        self.assertEqual(study.select(runs)['selected_weight'],10.)
        next(r for r in runs if r['weight']==10.)['metrics']['train_32']['accuracy']=.98
        self.assertEqual(study.select(runs)['selected_weight'],1.)
        next(r for r in runs if r['weight']==1.)['metrics']['validation_2']['accuracy']=.96
        self.assertIsNone(study.select(runs)['selected_weight'])
        with self.assertRaises(AssertionError):study.select(runs[:-1])

    def test_decision_distinguishes_improvement_from_complete_retention(self):
        runs=[]
        for s in study.CONFIRM_SEEDS:
            for arm in ('baseline','penalty'):
                metrics={f'{split}_{m}':dict(accuracy=1.) for split in ('train','test') for m in study.study.LOADS}
                metrics['all_train_32']=dict(all_queries=dict(accuracy=1.))
                metrics['all_test_32']=dict(all_queries=dict(accuracy=.5 if arm=='baseline' else .7),complete_maps=dict(accuracy=.96))
                runs.append(dict(seed=s,arm=arm,metrics=metrics))
        self.assertTrue(study.decision(runs)['simultaneous_retention_supported'])
        runs[0]['metrics']['all_train_32']['all_queries']['accuracy']=.5
        result=study.decision(runs);self.assertTrue(result['bounded_benefit']);self.assertFalse(result['simultaneous_retention_supported'])
        next(r for r in runs if r['arm']=='penalty')['metrics']['all_test_32']['all_queries']['accuracy']=.55
        self.assertFalse(study.decision(runs)['bounded_benefit'])
        with self.assertRaises(AssertionError):study.decision(runs[:-1])


if __name__=='__main__':unittest.main()
