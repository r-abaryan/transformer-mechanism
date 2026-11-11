"""Actual saved gradients/audit, fixed-mask probes and inverse error controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'tests'))
import memory_gradient_competition as study
import audit_memory_gradient_competition as audit
from test_query_feedback_training import payload


class MemoryGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_saved_gradients_actual_audit_and_fixed_mask_probes(self):
        folder=ROOT/'results/test_runs'/('memory_gradient_'+uuid.uuid4().hex);folder.mkdir()
        path=folder/'initial.pt';model=study.training.model_source.pair(1721)['delta']
        torch.save(dict(model=model.state_dict()),path);data=payload();cells=[];gradients=[]
        for load in (2,8,16,32):
            rows=data['pools'][f'train_{load}'];model=study.training.previous.load_model(path)
            cell=study.cell(model,rows,batch=2);cell['load']=load
            independent,_=audit.audit_cell(path,rows,cell,cell,batch=3)
            cells.append(cell);gradients.append(independent)
        pooled=study.pooled(cells);independent=audit.combine(cells,gradients)
        audit.prior.compare(study.geometry(pooled),audit.scalar_geometry(independent))
        probes,answers=study.probes(model,data['pools']['train_32'],cells[-1],pooled)
        audit.audit_probes(probes,independent);self.assertGreater(answers,0)
        bad=copy.deepcopy(cells[0]);bad['gradients']['correct']['key.weight'][0,0]+=.01
        with self.assertRaisesRegex(AssertionError,'numerical'):audit.audit_cell(path,data['pools']['train_2'],bad,bad)
        bad=copy.deepcopy(cells[0]);bad['labels'][0,0]=(int(bad['labels'][0,0])+1)%6
        with self.assertRaisesRegex(AssertionError,'Labels'):audit.audit_cell(path,data['pools']['train_2'],bad,bad)
        bad=copy.deepcopy(cells[0]);bad['partitions']['error']['count']+=1
        with self.assertRaises(AssertionError):audit.audit_cell(path,data['pools']['train_2'],bad,bad)
        bad=copy.deepcopy(probes);first=next(p for p in bad if not p['skipped']);first['analytic']+=1
        with self.assertRaisesRegex(AssertionError,'Probe sign'):audit.audit_probes(bad,independent)
        bad=copy.deepcopy(cells)
        for c in bad:
            for p in c['gradients']:
                for n in c['gradients'][p]:c['gradients'][p][n]*=c['labels'].numel()/sum(x['labels'].numel() for x in cells)*4
        with self.assertRaises(AssertionError):audit.prior.compare(study.geometry(study.pooled(bad)),audit.scalar_geometry(independent))
        with self.assertRaisesRegex(AssertionError,'Missing load'):study.pooled(cells[:3])

    def test_supplied_ties_and_absent_partitions(self):
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]])
        actual=study.partitions(z,y);audit.prior.compare(actual,audit.partition_metrics(z,y))
        self.assertEqual(actual['correct']['count'],1);self.assertEqual(actual['error']['count'],1)
        for label,absent in ((0,'error'),(1,'correct')):
            actual=study.partitions(z,torch.full_like(y,label));audit.prior.compare(actual,audit.partition_metrics(z,torch.full_like(y,label)))
            self.assertEqual(actual[absent]['count'],0);self.assertIsNone(actual[absent]['ce']);self.assertEqual(actual[absent]['weighted_loss'],0)

    def test_pairwise_opposition_does_not_determine_total_direction(self):
        for c,e,total_alignment in ((-1.,2.,'aligned'),(-3.,1.,'opposed')):
            gradients={p:{n:torch.full_like(v,0.) for n,v in study.training.model_source.pair(1723)['delta'].state_dict().items() if n in study.NAMES}
                       for p in ('correct','error32','other_errors','total','conditional_error32')}
            for part,value in (('correct',c),('error32',e),('total',c+e),('conditional_error32',e)):
                gradients[part]['write_logit'].fill_(value)
            record=study.geometry(gradients)['alpha'];self.assertEqual(record['correct_alignment'],'opposed')
            self.assertEqual(record['total_alignment'],total_alignment)
            audit.prior.compare(study.geometry(gradients),audit.scalar_geometry(gradients))


if __name__=='__main__':unittest.main()
