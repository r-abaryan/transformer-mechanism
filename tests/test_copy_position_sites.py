"""Isolated factorial wiring, evaluator and saved-audit controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import copy_position_sites as study
from copy_position_exposure import forward as preceding_forward
from audit_copy_position_sites import audit, audit_cell, own_decision
from audit_copy_position_exposure import own_measures, close

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


class SiteTests(unittest.TestCase):
    def test_equations_gradients_and_site_wiring(self):
        torch.manual_seed(41);model=study.make_model('rope')
        for n in (3,6,10):
            x=torch.tensor([[3+i%8 for i in range(n)],[4]*n]);prefix,y=study.inputs(x)
            self.assertEqual(y.tolist(),[r+[2] for r in x.tolist()])
            self.assertEqual(prefix.tolist(),[[1]+r for r in x.tolist()])
            pos=study.positions(n,torch.Generator().manual_seed(12))
            for arm in study.ARMS:
                a,b=study.SITES[arm];sp=pos if a else None;cp=pos if b else None
                z,s=study.forward(model,x,prefix,sp,cp)
                rz,rs=study.forward(model,x,prefix,sp,cp,True)
                self.assertLess(float((z-rz).detach().abs().max()),1e-9)
                self.assertLess(float((s-rs).detach().abs().max()),1e-9)
                ga=torch.autograd.grad(F.cross_entropy(z.flatten(0,1),y.flatten()),tuple(model.parameters()))
                gb=torch.autograd.grad(F.cross_entropy(rz.flatten(0,1),y.flatten()),tuple(model.parameters()))
                self.assertLess(max(float((aa-bb).abs().max()) for aa,bb in zip(ga,gb)),1e-9)
                if arm in ('ordinary','both'):
                    old=preceding_forward(model,x,prefix,pos if arm=='both' else None)
                    self.assertTrue(torch.equal(z,old[0]));self.assertTrue(torch.equal(s,old[1]))
            # Explicit wiring assertions do not derive expected sites from SITES.
            for arm,expected_self,expected_cross in [('ordinary',False,False),('self_only',True,False),('cross_only',False,True),('both',True,True)]:
                with patch.object(study,'attention',wraps=study.attention) as att, patch.object(study,'rotate',wraps=study.rotate) as rot:
                    met,saved=study.evaluate(model,x.tolist(),arm,pos)
                ep=pos if expected_self else torch.arange(n+1)
                cp=pos if expected_cross else torch.arange(n+1)
                for call in att.call_args_list[:2]:
                    self.assertTrue(torch.equal(call.args[5],ep));self.assertTrue(torch.equal(call.args[6],ep))
                for call in rot.call_args_list[:2]:self.assertTrue(torch.equal(call.args[1],cp))
                audit_cell(model,x.tolist(),pos,arm,saved,met)
        # Intentionally reversed site choice must fail the actual saved evaluator audit.
        rows=[[3,4,5],[4,4,4]];pos=torch.tensor([0,2,6,10])
        met,saved=study.evaluate(model,rows,'cross_only',pos)
        with self.assertRaises(AssertionError):audit_cell(model,rows,pos,'self_only',saved,met)
        audit_cell(model,rows,pos,'cross_only',saved,met)
        with self.assertRaises(AssertionError):study.forward(model,torch.tensor(rows),torch.tensor([[1]+r for r in rows]),torch.zeros(4,dtype=torch.long))

    def test_known_logits_and_decision(self):
        rows=[[3]*10,[4]*10];labels=torch.tensor([r+[2] for r in rows])
        z=torch.full((2,11,16),-10.,dtype=torch.float64);z.scatter_(-1,labels[:,:,None],10.)
        s=torch.zeros(2,2,11,dtype=torch.float64);s[:,:,0]=1;outputs=[r+[2] for r in rows]
        close(study.measures(rows,z,s,outputs),own_measures(rows,z,s,outputs))
        self.assertIsNone(study.measures(rows,z,s,outputs)['distinct'])
        z[0,0,0]=10;s[0,0,7]=2
        met=study.measures(rows,z,s,outputs);close(met,own_measures(rows,z,s,outputs))
        self.assertEqual(met['all']['first_accuracy'],.5);self.assertEqual(met['all']['outranked_fraction'],.5)
        runs=[]
        for arm in study.ARMS:
            metrics={}
            for n in (3,4,5,6,8,10):
                for split in ('train','test'):
                    if split=='train' and n>6:continue
                    good=n<=6 or arm in ('cross_only','both')
                    metrics[f'{split}_{n}']={g:dict(exact_accuracy=1. if good else .2,outranked_fraction=0. if good else .5) for g in ('all','distinct','repeated')}
            for n in range(3,7):
                for j in range(4):metrics[f'layout{j}_{n}']=copy.deepcopy(metrics[f'train_{n}'])
            runs.append(dict(seed=1,arm=arm,metrics=metrics))
        close(study.decision(runs),own_decision(runs));d=study.decision(runs)
        self.assertTrue(d['combined_replicated']);self.assertTrue(d['cross_only_sufficient']);self.assertFalse(d['self_only_sufficient'])
        broken=copy.deepcopy(runs);broken[2]['metrics']['test_10']['repeated']['outranked_fraction']=.5
        self.assertFalse(study.decision(broken)['cross_only_sufficient']);close(study.decision(broken),own_decision(broken))
        broken=copy.deepcopy(runs);broken[0]['metrics']['train_3']['distinct']['exact_accuracy']=.5
        self.assertFalse(study.decision(broken)['combined_replicated'])
        broken=copy.deepcopy(runs);del broken[1]['metrics']['test_8']['distinct']
        with self.assertRaises(KeyError):study.decision(broken)

    def test_actual_initialized_saved_run_and_faults(self):
        folder=ROOT/'results/test_runs'/('position_sites_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        pools={f'{split}_{n}':[list(range(3,3+n)),[4]*n] for n in range(3,7) for split in ('train','validation')}
        data=dict(pools=pools,layouts={str(n):[list(range(n+1)) for _ in range(4)] for n in range(3,7)})
        study.write(folder/'data.json',data);study.write(folder/'lock.json',dict(dev=43,confirm=[47],stages=[2],protected={}))
        report=study.experiment(folder);self.assertEqual(report['status'],'development_fit_failed')
        self.assertTrue(audit(folder,formal=False)['passed'])
        for fault in ('metric','missing','stream','selection','arm'):
            broken=copy.deepcopy(report)
            if fault=='metric':broken['development'][0]['metrics']['train_3']['all']['loss']+=1
            elif fault=='missing':broken['development'][0]['metrics'].pop('layout0_3')
            elif fault=='stream':broken['development'][0]['position_stream_sha256']='bad'
            elif fault=='selection':broken['selected_steps']=2
            else:broken['development'][0]['arm']='both'
            study.write(folder/'report.json',broken)
            try:
                with self.assertRaises(AssertionError):audit(folder,save=False,formal=False)
            finally:study.write(folder/'report.json',report)
        self.assertTrue(audit(folder,save=False,formal=False)['passed'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SiteTests))
    if result.wasSuccessful():
        assert not (study.OUT/'preflight.json').exists()
        study.write(study.OUT/'preflight.json',dict(passed=True,methods=result.testsRun,gradient_cases=12,
            initialized_training_updates=8,actual_saved_audit=True,intentional_artifact_faults=5,
            wrong_site_control=True,source_sha256=study.digest(Path(__file__))))
    sys.exit(not result.wasSuccessful())
