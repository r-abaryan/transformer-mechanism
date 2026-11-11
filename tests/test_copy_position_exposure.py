"""Isolated learning/evaluation/audit fixtures for positional exposure."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from copy_position_exposure import OUT, ARMS, make_model, forward, positions, inputs, measures, decision, experiment, write, digest
from audit_copy_position_exposure import audit, audit_cell, own_measures, own_decision, close
from encoder_decoder_copy import logits

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


class PositionTests(unittest.TestCase):
    def test_positions_equations_gradients_and_reload(self):
        torch.manual_seed(31);model=make_model('rope')
        gen=torch.Generator().manual_seed(22)
        for n in (3,6,10):
            x=torch.tensor([[3+i%8 for i in range(n)],[4]*n]);prefix,y=inputs(x)
            self.assertEqual(y.tolist(),[r+[2] for r in x.tolist()])
            self.assertEqual(prefix.tolist(),[[1]+r for r in x.tolist()])
            for pos in (None,positions(n,gen)):
                z,s=forward(model,x,prefix,pos);rz,rs=forward(model,x,prefix,pos,True)
                self.assertLess(float((z-rz).detach().abs().max()),1e-9)
                self.assertLess(float((s-rs).detach().abs().max()),1e-9)
                a=torch.autograd.grad(F.cross_entropy(z.flatten(0,1),y.flatten()),tuple(model.parameters()))
                b=torch.autograd.grad(F.cross_entropy(rz.flatten(0,1),y.flatten()),tuple(model.parameters()))
                self.assertLess(max(float((aa-bb).abs().max()) for aa,bb in zip(a,b)),1e-9)
            native=forward(model,x,prefix)[0]
            self.assertLess(float((native-logits(model,x,prefix)).detach().abs().max()),1e-9)
            identity=forward(model,x,prefix,torch.arange(n+1))[0]
            self.assertTrue(torch.equal(native,identity))
            translated=forward(model,x,prefix,torch.arange(n+1)+13)[0]
            self.assertLess(float((native-translated).detach().abs().max()),1e-9)
            if n==6:
                changed=forward(model,x,prefix,torch.tensor([0,1,3,5,6,8,10]))[0]
                self.assertGreater(float((native-changed).detach().abs().max()),1e-6)
            with self.assertRaises(AssertionError):forward(model,x,prefix,torch.zeros(n+1,dtype=torch.long))

    def test_metrics_and_hypothesis_control(self):
        rows=[[3]*10,[4]*10];labels=torch.tensor([r+[2] for r in rows])
        z=torch.full((2,11,16),-10.,dtype=torch.float64);z.scatter_(-1,labels[:,:,None],10.)
        s=torch.zeros(2,2,11,dtype=torch.float64);s[:,:,0]=1
        generated=[r+[2] for r in rows]
        m=measures(rows,z,s,generated);close(m,own_measures(rows,z,s,generated))
        self.assertIsNone(m['distinct']);self.assertEqual(m['all']['exact_accuracy'],1.)
        s[0,0,7]=2;self.assertEqual(measures(rows,z,s,generated)['all']['outranked_fraction'],.5)
        z[0,0,0]=10;self.assertEqual(measures(rows,z,s,generated)['all']['first_accuracy'],.5)
        # A failure of the mechanistic prediction must fail the joint criterion.
        runs=[]
        for arm in ARMS:
            metrics={}
            for n in (3,4,5,6,8,10):
                value=1. if n<=6 or arm=='spaced' else .2
                for split in ('train','test'):
                    metrics[f'{split}_{n}']={g:dict(exact_accuracy=value,outranked_fraction=0. if arm=='spaced' else .5) for g in ('all','distinct','repeated')}
            for n in range(3,7):
                for j in range(4):metrics[f'layout{j}_{n}']=metrics[f'train_{n}']
            metrics={k:v for k,v in metrics.items() if k not in ('train_8','train_10')}
            runs.append(dict(seed=1,arm=arm,metrics=metrics))
        close(decision(runs),own_decision(runs));self.assertTrue(decision(runs)['supported'])
        runs[1]['metrics']['test_10']['repeated']['outranked_fraction']=.5
        close(decision(runs),own_decision(runs));self.assertFalse(decision(runs)['supported'])

    def test_actual_initialized_training_saved_audit_corruptions(self):
        folder=ROOT/'results/test_runs'/('position_exposure_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        pools={}
        for n in range(3,7):
            for split in ('train','validation'):
                pools[f'{split}_{n}']=[list(range(3,3+n)),[4]*n]
        data=dict(pools=pools,layouts={str(n):[list(range(n+1)) for _ in range(4)] for n in range(3,7)})
        write(folder/'data.json',data);write(folder/'lock.json',dict(protected={},stages=[2],confirm=[47]))
        report=experiment(folder,dev=43,confirm=(47,),stages=(2,))
        self.assertEqual(report['status'],'development_fit_failed')
        self.assertTrue(audit(folder,formal=False)['passed'])
        for fault in ('metric','missing','stream','selection'):
            broken=copy.deepcopy(report)
            if fault=='metric':broken['development'][0]['metrics']['train_3']['all']['loss']+=1
            elif fault=='missing':broken['development'][0]['metrics'].pop('layout0_3')
            elif fault=='stream':broken['development'][0]['stream_sha256']='wrong'
            else:broken['selected_steps']=2
            write(folder/'report.json',broken)
            try:
                with self.assertRaises(AssertionError):audit(folder,save=False,formal=False)
            finally:write(folder/'report.json',report)
        self.assertTrue(audit(folder,save=False,formal=False)['passed'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PositionTests))
    if result.wasSuccessful():
        assert not (OUT/'preflight.json').exists()
        write(OUT/'preflight.json',dict(passed=True,methods=result.testsRun,gradient_cases=6,
              initialized_training_updates=4,actual_saved_audit=True,intentional_artifact_faults=4,source_sha256=digest(Path(__file__))))
    sys.exit(not result.wasSuccessful())
