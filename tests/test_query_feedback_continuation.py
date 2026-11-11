"""Exact resume, actual evaluator/audit paths and stopping-rule error controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT/'experiments'))
import query_feedback_continuation as study
import audit_query_feedback_continuation as audit
from test_query_feedback_training import payload


class ContinuationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_exact_uninterrupted_resume_and_inverse_controls(self):
        data=payload();folder=ROOT/'results/test_runs'/('feedback_resume_'+uuid.uuid4().hex);folder.mkdir()
        original=dict(runs=[],cells=[]);old_scores=[]
        study.parent.train_run(folder,1207,'exhaustive',data,original,old_scores,steps=8,probes=(0,4,8))
        prior=original['runs'][0];resume_folder=folder/'resume';resume_folder.mkdir()
        report=dict(runs=[],cells=[]);scores=[]
        item=study.continue_run(resume_folder,1207,ROOT/prior['snapshots'][1]['checkpoint'],data,report,scores,
                                cap=8,interval=4,threshold=1.01)
        baseline=torch.load(ROOT/prior['final_checkpoint'],weights_only=True)
        resumed=torch.load(ROOT/item['final_checkpoint'],weights_only=True)
        for name in ('model','optimizer'):self.assertTrue(study.parent.replay.equal(baseline[name],resumed[name]))
        self.assertEqual(item['stop_reason'],'budget_cap')
        checked=audit.audit_training(item,data,scores,prior['steps'][:4],2.)
        self.assertLess(checked['max_logit_error'],1e-9)
        broken=copy.deepcopy(item);broken['steps'][0]['exp_avg']+=.01
        with self.assertRaisesRegex(AssertionError,'Adam'):audit.audit_training(broken,data,scores,prior['steps'][:4],2.)
        badpath=resume_folder/'wrong_rng.pt';bad=copy.deepcopy(resumed);bad['history_rng']=torch.Generator().manual_seed(9).get_state();torch.save(bad,badpath)
        broken=copy.deepcopy(item);broken['snapshots'][-1].update(checkpoint=badpath.relative_to(ROOT).as_posix(),sha256=study.parent.replay.digest(badpath))
        with self.assertRaisesRegex(AssertionError,'stream'):audit.audit_training(broken,data,scores,prior['steps'][:4],2.)
        wrong=copy.deepcopy(scores[-1]);wrong['logits'][0,0,0]+=.1
        with self.assertRaises(AssertionError):audit.reference.audit_cell(ROOT/wrong['checkpoint'],data['pools'][wrong['pool']],wrong)

    def test_initialized_reload_gradients_and_first_fit_stop(self):
        data=payload();folder=ROOT/'results/test_runs'/('feedback_initial_'+uuid.uuid4().hex);folder.mkdir()
        model=study.parent.model_source.pair(1207)['delta'];opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        cp=folder/'initial.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0),cp)
        report=dict(runs=[],cells=[]);scores=[]
        initialized=study.parent.evaluate(model,data['pools']['train_32'])
        e,q,y,_=audit.reference.independent.inputs(data['pools']['train_32'],'all_queries')
        self.assertLess(audit.reference.check_gradients(model,e,q,y),1e-9)
        initial_accs=[study.parent.evaluate(model,data['pools'][f'train_{m}'])['metrics']['groups']['all']['accuracy'] for m in (2,8,16,32)]
        self.assertLess(min(initial_accs),.126)
        item=study.continue_run(folder,1207,cp,data,report,scores,cap=64,interval=16,threshold=.126)
        self.assertLess(item['stop'],64);self.assertEqual(item['stop_reason'],'training_fit')
        audit.audit_training(item,data,scores,[],2.)
        initial_cell=next(c for c in scores if c['step']==0 and c['pool']=='train_32')
        self.assertTrue(torch.equal(initialized['logits'],initial_cell['logits']))
        self.assertIsNone(initial_cell['metrics']['groups']['repeat']['ce'])
        broken=copy.deepcopy(item);broken['stop_reason']='budget_cap'
        with self.assertRaises(AssertionError):audit.audit_training(broken,data,scores,[],2.)

    def test_metrics_missing_groups_stopping_and_counts(self):
        logits=torch.tensor([[[3.,0.,0.,0.,0.,0.],[0.,3.,0.,0.,0.,0.]]],dtype=torch.float64)
        labels=torch.tensor([[0,1]]);repeat=torch.ones_like(labels,dtype=torch.bool)
        audit.reference.check_metrics(study.parent.reduce_scores(logits,labels,repeat),audit.reference.metrics(logits,labels,repeat))
        training={}
        for m in (2,8,16,32):
            y=torch.tensor([[0]*m]);z=torch.zeros(1,m,6,dtype=torch.float64);z[:,:,0]=2.
            training[str(m)]=study.parent.reduce_scores(z,y,torch.full_like(y,m!=32,dtype=torch.bool))
        self.assertTrue(study.training_fit(training));self.assertTrue(audit.fit(training,.99))
        wrong=copy.deepcopy(training);del wrong['16']
        with self.assertRaises(AssertionError):study.training_fit(wrong)
        wrong=copy.deepcopy(training);wrong['32']['groups']['unique']['count']=0
        with self.assertRaisesRegex(AssertionError,'group'):study.training_fit(wrong)
        with self.assertRaisesRegex(AssertionError,'group'):audit.fit(wrong,.99)
        wrong=copy.deepcopy(training);wrong['32']['groups']['all']['accuracy']=.98;wrong['32']['groups']['unique']['accuracy']=.98
        self.assertFalse(study.training_fit(wrong));self.assertFalse(audit.fit(wrong,.99))
        bad=payload()['pools']['train_32'];bad[0]['label']=5
        with self.assertRaisesRegex(AssertionError,'label'):audit.reference.independent.check_rows(bad)
        self.assertEqual(2*(768*10+192+384)*sum((2,8,16,32)),957696)
        self.assertEqual(64*2250*sum((2,8,16,32)),8352000)


if __name__=='__main__':unittest.main()
