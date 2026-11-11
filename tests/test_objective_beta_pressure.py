"""Endpoint gradients, numerical inverses and actual initialized audit reload."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import objective_beta_pressure as study
import audit_objective_beta_pressure as audit


def fixture(load):
    events=[[k,k%6] for _ in range(32//load) for k in range(load)]
    return [dict(events=events,query=1,label=1,load=load,group='unique' if load==32 else 'repeat'),
            dict(events=list(reversed(events)),query=0,label=0,load=load,group='unique' if load==32 else 'repeat')]


class ObjectivePressureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_labels_counts_and_inverse(self):
        for load in study.LOADS:
            rows=fixture(load);audit.check_rows(rows)
            for objective in study.OBJECTIVES:
                a=study.prepare(rows,objective);b=audit.inputs(rows,objective)
                for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
                self.assertEqual(a[2].numel(),2*(load if objective=='all_queries' else 1))
            broken=copy.deepcopy(rows);broken[0]['label']=2
            with self.assertRaises(AssertionError):audit.check_rows(broken)
        self.assertEqual(2*2*768*(4+2+8+16+32),190464)

    def test_actual_saved_evaluator_and_audit(self):
        model=study.model_source.pair(1193)['delta'].eval()
        folder=ROOT/'results/test_runs'/('objective_pressure_'+uuid.uuid4().hex);folder.mkdir()
        checkpoint=folder/'initial.pt';torch.save(dict(model=model.state_dict(),step=0),checkpoint)
        for load in study.LOADS:
            for objective in study.OBJECTIVES:
                rows=fixture(load);cell=study.evaluate(model,rows,objective,batch=1)
                checked=audit.audit_checkpoint(checkpoint,rows,objective,cell,batch=2)
                self.assertLess(checked['max_parameter_gradient_error'],1e-10)
                missing='repeat' if load==32 else 'unique'
                self.assertEqual(cell['metrics'][missing]['count'],0)
                self.assertIsNone(cell['metrics'][missing]['ce'])
        broken=copy.deepcopy(cell);broken['gradients']['write_logit']+=.1
        with self.assertRaises(AssertionError):audit.audit_checkpoint(checkpoint,rows,objective,broken)

    def test_supplied_logits_and_bad_metric(self):
        scores=torch.tensor([[[0.,3.,0.,0.,0.,0.]],[[3.,0.,0.,0.,0.,0.]]],dtype=torch.float64)
        labels=torch.tensor([[1],[0]]);repeat=torch.ones(2,1,dtype=torch.bool)
        native=study.reduce_logits(scores,labels,repeat);ref=audit.metrics(scores,labels,repeat)
        self.assertEqual(native['all']['accuracy'],1.)
        self.assertEqual(native['unique']['count'],0)
        audit.close(native['all']['ce'],ref['all']['ce'])
        with self.assertRaises(AssertionError):audit.close(native['all']['ce']+.1,ref['all']['ce'])

    def test_tangent_chain_rule_and_directions(self):
        model=study.model_source.pair(1199)['delta'].eval()
        row=dict(events=[[0,0]],query=0,label=0,load=1,group='unique')
        e,q,y,_=audit.inputs([row],'assigned')
        with torch.no_grad():
            logits,tangent=audit.forward_tangent(model,e,q)
            alpha=model.write_logit.item();h=1e-4
            plus,_=audit.forward_tangent(model,e,q,h);minus,_=audit.forward_tangent(model,e,q,-h)
            beta=float(model.write_logit.sigmoid())
            torch.testing.assert_close((plus-minus)/(2*h),tangent*beta*(1-beta),atol=1e-9,rtol=1e-6)
            # One normalized-key write/read equals beta*v, so tangent equals output(v).
            expected=model.value.weight[0]@model.output.weight.T
            torch.testing.assert_close(tangent[0,0],expected,atol=1e-12,rtol=1e-12)
        self.assertEqual(study.pressure(.001),'lower');self.assertEqual(study.pressure(-.001),'higher')
        self.assertEqual(study.pressure(0.),'unresolved')
        a={'write_logit':torch.tensor(.001)};b={'write_logit':torch.tensor(-.001)}
        self.assertTrue(study.compare(a,b)['opposite_resolved_signs'])
        self.assertFalse(study.compare(a,a)['opposite_resolved_signs'])
        self.assertIsNone(study.compare({'write_logit':torch.tensor(0.)},b)['gradient_cosines']['whole_model'])


if __name__=='__main__':unittest.main()
