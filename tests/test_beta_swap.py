"""Beta-only changes, first-prefix eligibility, real reload/audit and inverse controls."""
import copy
import math
import sys
import unittest
import uuid
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import beta_swap as study


def fixture(load):
    return [dict(events=[[k,k%6] for _ in range(32//load) for k in range(load)],query=1,label=1,load=load,group='unique' if load==32 else 'repeat')]


class BetaSwapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_beta_only_and_self_copy_controls(self):
        base=study.study.pair(3461)['delta'];other=study.study.pair(3463)['delta']
        with torch.no_grad():other.write_logit.fill_(-1.)
        changed=study.variant_model(base,other,'donor_beta')
        self.assertTrue(torch.equal(changed.write_logit,other.write_logit));study.beta_only(base,changed)
        rows=fixture(32);a=study.evaluate(base,rows);b=study.evaluate(study.variant_model(base,base,'donor_beta'),rows)
        for s in a:torch.testing.assert_close(a[s],b[s],atol=0,rtol=0)
        with torch.no_grad():changed.key.weight[0,0]+=.01
        with self.assertRaises(AssertionError):study.beta_only(base,changed)

    def test_first_write_positions_and_wrong_label(self):
        for load in study.study.LOADS:
            rows=fixture(load);q,y,p=study.positions(rows)
            self.assertEqual(q.tolist(),[list(range(load))]);self.assertEqual(p.tolist(),[list(range(load))])
            self.assertEqual(y.tolist(),[[k%6 for k in range(load)]])
        bad=fixture(8);bad[0]['label']=4
        with self.assertRaises(AssertionError):study.positions(bad)
        bad=fixture(8);bad[0]['events'][-1][1]=4
        with self.assertRaises(AssertionError):study.positions(bad)

    def test_actual_initialized_checkpoint_evaluator_and_audit(self):
        folder=ROOT/'results/test_runs'/('beta_swap_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
        base=study.study.pair(3461)['delta'];other=study.study.pair(3463)['delta']
        with torch.no_grad():other.write_logit.fill_(-1.)
        cp,dp=folder/'base.pt',folder/'donor.pt';torch.save(dict(model=base.state_dict(),step=0),cp);torch.save(dict(model=other.state_dict(),step=0),dp)
        for load in study.study.LOADS:
            rows=fixture(load)
            for variant in study.VARIANTS:
                scores=study.evaluate(study.variant_model(base,other,variant),rows);metrics=study.summarize(scores,rows)
                study.audit_cell(cp,dp,variant,rows,scores,metrics)
                bad=copy.deepcopy(metrics);bad['immediate']['loss']+=.01
                with self.assertRaises(AssertionError):study.audit_cell(cp,dp,variant,rows,scores,bad)
                bad=copy.deepcopy(metrics);bad['complete_maps']['correct']=1-bad['complete_maps']['correct']
                with self.assertRaises(AssertionError):study.audit_cell(cp,dp,variant,rows,scores,bad)
                bad={s:v.clone() for s,v in scores.items()};bad['immediate'][...,0]+=.1
                with self.assertRaises(AssertionError):study.audit_cell(cp,dp,variant,rows,bad,metrics)
                if load==32:torch.testing.assert_close(scores['immediate'][:,31],scores['final'][:,31],atol=1e-12,rtol=1e-12)

    def test_two_write_equation_gradient_and_wrong_zero_control(self):
        model=study.study.LoadReader('delta');c=.8;beta=.3
        with torch.no_grad():
            model.key.weight.zero_();model.key.weight[0,0]=1.;model.key.weight[1,0]=c;model.key.weight[1,1]=math.sqrt(1-c*c)
            model.value.weight.zero_();model.value.weight[0,0]=1.;model.value.weight[1,1]=1.
            model.output.weight.zero_();model.output.weight[0,0]=1.;model.output.weight[1,1]=1.;model.output.bias.zero_()
            model.write_logit.fill_(math.log(beta/(1-beta)))
        e=torch.tensor([[[0,0],[1,1]]]);q=torch.tensor([0]);scores=model(e,q)
        torch.testing.assert_close(scores[0,:2],torch.tensor([beta*(1-beta*c*c),beta*c],dtype=torch.float64),atol=1e-12,rtol=1e-12)
        derivative=torch.autograd.grad(scores[0,0],model.write_logit)[0]
        expected=(1-2*beta*c*c)*beta*(1-beta);self.assertAlmostEqual(float(derivative),expected,places=12)
        raw=float(model.write_logit.detach());eps=1e-6
        with torch.no_grad():
            model.write_logit.fill_(raw+eps);plus=float(model(e,q)[0,0]);model.write_logit.fill_(raw-eps);minus=float(model(e,q)[0,0]);model.write_logit.fill_(raw)
        self.assertAlmostEqual((plus-minus)/(2*eps),expected,places=8)
        with self.assertRaises(AssertionError):torch.testing.assert_close(derivative,torch.zeros_like(derivative),atol=1e-12,rtol=1e-12)

    def test_decision_directions_and_missing_coverage(self):
        cells=[]
        for seed in study.prior.DEV_SEEDS:
            for arm in study.prior.ARMS:
                for pool in study.POOLS:
                    for variant in study.VARIANTS:
                        acc=(.5 if arm=='fixed' else .9) if variant=='native' else (.7 if arm=='fixed' else .6)
                        cells.append(dict(seed=seed,arm=arm,pool=pool,variant=variant,metrics=dict(final=dict(accuracy=acc),complete_maps=dict(accuracy=.0))))
        self.assertTrue(study.decision(cells)['bidirectional_write_strength_evidence'])
        next(c for c in cells if c['arm']=='fixed' and c['variant']=='donor_beta' and c['pool']=='validation_32')['metrics']['final']['accuracy']=.5001
        self.assertFalse(study.decision(cells)['bidirectional_write_strength_evidence'])
        with self.assertRaises(AssertionError):study.decision(cells[:-1])


if __name__=='__main__':unittest.main()
