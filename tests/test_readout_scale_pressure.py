"""Pressure signs, weighting and actual saved evaluator/head-gradient audit."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'tests'))
import readout_scale_pressure as study
import audit_readout_scale_pressure as audit
from test_query_feedback_training import payload


class PressureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_supplied_signs_ties_absent_groups_and_corruptions(self):
        # Wrong class: near-top true score can still benefit from expansion.
        z=torch.tensor([[[4,0,0,0,0,0],[0,4,0,0,0,0],[1.9,2,0,0,0,0],[0,0,0,0,0,0]]],dtype=torch.float64)
        y=torch.zeros(1,4,dtype=torch.long);r=torch.zeros_like(y,dtype=torch.bool)
        cell=dict(logits=z,labels=y,repeat=r);g=study.derivative(z,y)
        self.assertLess(float(g[0]),0);self.assertGreater(float(g[1]),0);self.assertLess(float(g[2]),0);self.assertEqual(float(g[3]),0)
        actual=study.summarize(cell);expected=audit.reconstruct(cell);audit.prior.compare(actual,expected)
        self.assertEqual(actual['all']['partitions']['correct']['count'],2);self.assertIsNone(actual['repeat']['mean_g'])
        study.score_checks(cell)
        shifted=dict(cell,logits=z+1000);audit.prior.compare(study.summarize(shifted),expected)
        wrong=copy.deepcopy(actual);wrong['all']['mean_g']*=-1
        with self.assertRaisesRegex(AssertionError,'scalar'):audit.prior.compare(wrong,expected)
        wrong=copy.deepcopy(actual);wrong['all']['partitions']['error']['count']=0
        with self.assertRaisesRegex(AssertionError,'Count'):audit.prior.compare(wrong,expected)

    def test_equal_load_objective(self):
        records=[]
        for load in (2,8,16,32):
            records.append(dict(load=load,groups=dict(all=dict(ce=float(load),mean_g=-float(load),
                partitions=dict(correct=dict(contribution=-load-1.),error=dict(contribution=1.))))))
        self.assertEqual(study.objective(records),audit.objective(records))
        self.assertEqual(study.objective(records)['mean_g'],-14.5)
        with self.assertRaisesRegex(AssertionError,'Missing'):study.objective(records[:-1])

    def test_initialized_saved_model_reload_and_independent_gradients(self):
        rows=payload()['pools']['train_32'];folder=ROOT/'results/test_runs'/('scale_pressure_'+uuid.uuid4().hex);folder.mkdir()
        model=study.training.model_source.pair(1511)['delta'];path=folder/'initialized.pt'
        torch.save(dict(model=model.state_dict(),step=0),path)
        loaded=study.training.previous.load_model(path);source=study.training.evaluate(loaded,rows)
        record=dict(groups=study.summarize(source),**study.score_checks(source),**study.model_gradient(path,rows,source))
        result=audit.audit_cell(path,rows,source,record);self.assertLess(result['max_scalar_error'],1e-9)
        wrong=copy.deepcopy(record);wrong['head_gradient_g']+=.01
        with self.assertRaisesRegex(AssertionError,'gradient'):audit.audit_cell(path,rows,source,wrong)
        bad=copy.deepcopy(rows);bad[0]['label']=5
        with self.assertRaisesRegex(AssertionError,'label'):audit.audit_cell(path,bad,source,record)
        wrong=copy.deepcopy(source);wrong['logits'][0,0,0]+=.1
        with self.assertRaisesRegex(AssertionError,'logits'):study.model_gradient(path,rows,wrong)


if __name__=='__main__':unittest.main()
