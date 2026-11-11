"""Finite-pool variance and actual initialized checkpoint audits."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import query_gradient_variability as study
import audit_query_gradient_variability as audit


def fixture(load):
    events=[[k,k%6] for _ in range(32//load) for k in range(load)]
    return [dict(events=events,query=1,label=1,load=load,group='unique' if load==32 else 'repeat'),
            dict(events=list(reversed(events)),query=0,label=0,load=load,group='unique' if load==32 else 'repeat')]


class VariabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_variance_and_inverse(self):
        g=torch.tensor([[1.,3.],[5.,7.]],dtype=torch.float64);slots=torch.tensor([0,1])
        stats=study.statistics(g,slots)
        self.assertEqual(stats['mean'],4.);self.assertEqual(stats['population_variance'],5.)
        self.assertEqual(stats['history_variance'],4.);self.assertEqual(stats['query_variance'],1.)
        self.assertEqual(stats['assigned_mean'],4.);self.assertEqual(stats['assigned_variance'],9.)
        audit.compare_statistics(stats,audit.reduce_gradients(g.tolist(),slots.tolist()))
        wrong=dict(stats);wrong['query_variance']=0.
        with self.assertRaisesRegex(AssertionError,'Variance reduction'):audit.compare_statistics(wrong,stats)
        zero=study.statistics(torch.zeros(2,2,dtype=torch.float64),slots)
        self.assertIsNone(zero['one_query_snr']);self.assertIsNone(zero['query_variance_fraction'])
        same=study.statistics(torch.tensor([[1.,1.],[3.,3.]],dtype=torch.float64),slots)
        self.assertEqual(same['query_variance'],0.);self.assertEqual(same['all_to_one_se_ratio'],1.)
        self.assertEqual(2*2*7*4,112);self.assertEqual(2*2*7*768*(2+8+16+32),1247232)

    def test_actual_checkpoint_reload_and_target_controls(self):
        model=study.replay.pressure.model_source.pair(1201)['delta'].eval()
        folder=ROOT/'results/test_runs'/('query_variability_'+uuid.uuid4().hex);folder.mkdir()
        cp=folder/'initial.pt';torch.save(dict(model=model.state_dict(),step=0),cp)
        for load in (2,8,16,32):
            rows=fixture(load);cell=study.evaluate(model,rows,batch=1)
            checked=audit.audit_cell(cp,rows,cell,batch=2)
            self.assertLess(checked['full_mean_gradient_error'],1e-10)
            missing='repeat' if load==32 else 'unique'
            self.assertEqual(cell['metrics'][missing]['count'],0)
            self.assertIsNone(cell['metrics'][missing]['ce'])
        wrong=copy.deepcopy(cell);wrong['gradients'][0,0]+=.1
        with self.assertRaisesRegex(AssertionError,'finite-difference'):audit.audit_cell(cp,rows,wrong)
        bad=copy.deepcopy(rows);bad[0]['label']=5
        with self.assertRaisesRegex(AssertionError,'label'):audit.audit_cell(cp,bad,cell)

    def test_supplied_logits_and_bad_metric(self):
        logits=torch.tensor([[[0.,3.,0.,0.,0.,0.]]],dtype=torch.float64)
        labels=torch.tensor([[1]]);repeat=torch.ones(1,1,dtype=torch.bool)
        a=study.replay.pressure.reduce_logits(logits,labels,repeat);b=study.tangent.metrics(logits,labels,repeat)
        self.assertEqual(a['all']['accuracy'],1.);self.assertIsNone(a['unique']['accuracy'])
        study.tangent.close(a['all']['ce'],b['all']['ce'])
        with self.assertRaises(AssertionError):study.tangent.close(a['all']['ce']+.1,b['all']['ce'])


if __name__=='__main__':unittest.main()
