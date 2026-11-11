"""Multi-query supervision, actual saved evaluators and inverse controls."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import query_feedback_training as study
import audit_query_feedback_training as audit


def payload():
    pools={}
    for split in ('train','validation'):
        for m in study.LOADS:
            events=[[k,k%6] for _ in range(32//m) for k in range(m)]
            pools[f'{split}_{m}']=[dict(events=events,query=0,label=0,load=m,group='unique' if m==32 else 'repeat'),
                                  dict(events=list(reversed(events)),query=1,label=1,load=m,group='unique' if m==32 else 'repeat')]
    return dict(pools=pools)


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_policy_counts_and_gradient_normalization(self):
        data=payload();model=study.model_source.pair(1207)['delta']
        for m in study.LOADS:
            rows=data['pools'][f'train_{m}'];events,_,_=study.parent.study.parent.tensors(rows)
            q,y=study.parent.query_tables(rows);indices=torch.tensor([0,1]);draws=torch.zeros(2,m,dtype=torch.long)
            for arm in study.ARMS:
                selected,labels,slots=study.choose(q,y,indices,draws,arm)
                self.assertEqual(labels.numel(),2*m)
                self.assertEqual(len(set(slots[0].tolist())),m if arm=='exhaustive' else 1)
                self.assertLess(audit.check_gradients(model,events,selected,labels),1e-9)
            with self.assertRaises(AssertionError):study.choose(q,y,indices,draws+32,'sampled_m')
        self.assertEqual(64*750*(2+8+16+32),2784000)
        self.assertEqual(4*(4*768+192+384)*(2+8+16+32),846336)

    def test_actual_training_reload_audit_and_streams(self):
        data=payload();folder=ROOT/'results/test_runs'/('query_feedback_'+uuid.uuid4().hex);folder.mkdir()
        report=dict(runs=[],cells=[]);saved=[]
        for arm in study.ARMS:
            study.train_run(folder,1207,arm,data,report,saved,steps=8,probes=(0,4,8))
            run=report['runs'][-1]
            counts=torch.load(ROOT/run['exposure_path'],weights_only=True)
            checked=audit.audit_stream(1207,arm,{m:data['pools'][f'train_{m}'] for m in study.LOADS},8,counts)
            for name,value in checked.items():self.assertEqual(run[name],value)
        for c in saved:
            checked=audit.audit_cell(ROOT/c['checkpoint'],data['pools'][c['pool']],c,batch=1)
            self.assertLess(checked['max_logit_error'],1e-9)
            missing='repeat' if c['pool'].endswith('_32') else 'unique'
            self.assertIsNone(c['metrics']['groups'][missing]['ce'])
        a,b=report['runs']
        self.assertEqual(a['initial_sha256'],b['initial_sha256'])
        self.assertEqual(a['history_stream_sha256'],b['history_stream_sha256'])
        self.assertEqual(a['potential_draws_sha256'],b['potential_draws_sha256'])
        wrong=copy.deepcopy(saved[-1]);wrong['logits'][0,0,0]+=.01
        with self.assertRaises(AssertionError):audit.audit_cell(ROOT/wrong['checkpoint'],data['pools'][wrong['pool']],wrong)
        corrupted=copy.deepcopy(counts);corrupted[32][0,0]+=1
        with self.assertRaises(AssertionError):audit.audit_stream(1207,'sampled_m',{m:data['pools'][f'train_{m}'] for m in study.LOADS},8,corrupted)

    def test_supplied_logits_labels_and_complete_maps(self):
        logits=torch.tensor([[[3.,0.,0.,0.,0.,0.],[0.,3.,0.,0.,0.,0.]]],dtype=torch.float64)
        labels=torch.tensor([[0,1]]);repeat=torch.ones_like(labels,dtype=torch.bool)
        native=study.reduce_scores(logits,labels,repeat);ref=audit.metrics(logits,labels,repeat)
        audit.check_metrics(native,ref);self.assertEqual(native['complete_maps']['accuracy'],1.)
        broken=copy.deepcopy(native);broken['complete_maps']['correct']=0
        with self.assertRaises(AssertionError):audit.check_metrics(broken,ref)
        bad=payload()['pools']['train_32'];bad[0]['label']=5
        with self.assertRaisesRegex(AssertionError,'label'):audit.independent.check_rows(bad)


if __name__=='__main__':unittest.main()
