"""Reusable pretraining checks, including the actual final evaluation/audit cell."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from copy_rope_model import make_model, reference, rotate, matrix_rotation, rope_reference_features
from copy_rope_training import evaluate, DIRECTORY, SEEDS
from encoder_decoder_copy import inputs, logits
from audit_copy_rope_training import audit_cell
from position_distractors import digest
from recurrent_retrieval import write

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
ERRORS=[]


class CopyRopeTests(unittest.TestCase):
    def test_equations_gradients_and_reload_evaluator(self):
        for seed in SEEDS:
            for n in (3,6,10):
                for arm in ('sinusoidal','rope'):
                    with self.subTest(seed=seed,n=n,arm=arm):
                        torch.manual_seed(seed);model=make_model(arm)
                        self.assertEqual(sum(p.numel() for p in model.parameters()),1888)
                        # Both groups below 8, repeated-only at 10, as in the real pools.
                        rows=[list(range(3,3+n)) if n<=8 else [3,4]*5,[3]*n]
                        source=torch.tensor(rows);prefix,labels=inputs(source)
                        z=logits(model,source,prefix);rz=reference(model,source,prefix,arm)
                        a=F.cross_entropy(z.flatten(0,1),labels.flatten());b=F.cross_entropy(rz.flatten(0,1),labels.flatten())
                        ga=torch.autograd.grad(a,tuple(model.parameters()));gb=torch.autograd.grad(b,tuple(model.parameters()))
                        le=float((z-rz).detach().abs().max());ge=max(float((x-y).abs().max()) for x,y in zip(ga,gb))
                        self.assertLessEqual(le,1e-9);self.assertLessEqual(ge,1e-9)
                        with tempfile.TemporaryDirectory(dir=ROOT/'results') as temp:
                            path=Path(temp)/'initial.pt';torch.save(model.state_dict(),path)
                            loaded=make_model(arm);loaded.load_state_dict(torch.load(path,weights_only=True))
                            metrics,stored=evaluate(loaded,rows)
                            _,cell=audit_cell(loaded,arm,rows,metrics,stored)
                        if n==10:self.assertEqual(set(metrics),{'all','repeated'})
                        ERRORS.append(dict(seed=seed,length=n,arm=arm,initial_logit_error=le,gradient_error=ge,**cell))

    def test_relative_identity_and_error_controls(self):
        torch.manual_seed(5);q=torch.randn(2,2,5,4,dtype=torch.float64);k=torch.randn_like(q)
        p=torch.arange(5)
        a=rotate(q,p)@rotate(k,p).transpose(-1,-2)
        b=rotate(q,p+17)@rotate(k,p+17).transpose(-1,-2)
        self.assertLess(float((a-b).abs().max()),1e-12)
        self.assertLess(float((rotate(q,p)-matrix_rotation(q,p)).abs().max()),1e-12)
        # Omitting the K rotation violates the common-offset invariant.
        wrong=rotate(q,p+17)@k.transpose(-1,-2)
        self.assertGreater(float((a-wrong).abs().max()),1e-3)
        # Using model width 8 rather than head width 4 gives the wrong frequency.
        angle=p.double()*.1;pair=q[...,2:4]
        wrong_pair=torch.stack((pair[...,0]*angle.cos()-pair[...,1]*angle.sin(),
                                pair[...,0]*angle.sin()+pair[...,1]*angle.cos()),-1)
        self.assertGreater(float((rotate(q,p)[...,2:4]-wrong_pair).abs().max()),1e-3)

    def test_causal_padding_and_common_offset(self):
        torch.manual_seed(31);model=make_model('rope');model.train()  # Dropout remains exactly zero.
        s=torch.randn(2,5,8,dtype=torch.float64);t=torch.randn(2,4,8,dtype=torch.float64)
        sp=torch.zeros(2,5,dtype=torch.bool);sp[:,-1]=True
        tp=torch.zeros(2,4,dtype=torch.bool)
        z=model(s,t,sp,tp);rz=rope_reference_features(model,s,t,sp,tp)
        self.assertLess(float((z-rz).detach().abs().max()),1e-9)
        offset=model(s,t,sp,tp,position_offset=19)
        self.assertLess(float((z-offset).detach().abs().max()),1e-9)
        future=t.clone();future[:,2:]+=torch.randn_like(future[:,2:])*7
        altered=model(s,future,sp,tp)
        self.assertLess(float((z[:,:2]-altered[:,:2]).detach().abs().max()),1e-9)
        wrong=model(s,future,sp,tp,causal=False)
        self.assertGreater(float((z[:,:2]-wrong[:,:2]).detach().abs().max()),1e-5)
        padded=s.clone();padded[:,-1]+=torch.randn_like(padded[:,-1])*9
        self.assertLess(float((z-model(padded,t,sp,tp)).detach().abs().max()),1e-9)
        for flag in ('encoder_mask','memory_mask'):
            wrong=model(padded,t,sp,tp,**{flag:False})
            self.assertGreater(float((z-wrong).detach().abs().max()),1e-5)
        tp[:,-1]=True
        changed=t.clone();changed[:,-1]+=torch.randn_like(changed[:,-1])*7
        self.assertLess(float((model(s,t,sp,tp)-model(s,changed,sp,tp))[:,:-1].detach().abs().max()),1e-9)
        with self.assertRaisesRegex(ValueError,'Empty source'):model(s,t,torch.ones_like(sp),tp)

    def test_supplied_logits_absent_group_and_gradient_direction(self):
        rows=[[3,4]*5,[5]*10];z=torch.full((2,11,16),-10.,dtype=torch.float64)
        for i,row in enumerate(rows):
            for j,y in enumerate(row+[2]):z[i,j,y]=10.
        from copy_length_exposure import metrics
        result=metrics(z,rows,[r+[2] for r in rows])
        self.assertEqual(set(result),{'all','repeated'});self.assertEqual(result['repeated']['exact_accuracy'],1.)
        torch.manual_seed(47);model=make_model('rope');source=torch.tensor([[3,4,5],[3,3,4]])
        prefix,labels=inputs(source);loss=F.cross_entropy(logits(model,source,prefix).flatten(0,1),labels.flatten())
        params=tuple(model.parameters());grad=torch.autograd.grad(loss,params)
        direction=[torch.randn_like(p) for p in params];size=sum(float(d.square().sum()) for d in direction)**.5
        direction=[d/size for d in direction];analytical=sum(float((g*d).sum()) for g,d in zip(grad,direction))
        saved=[p.detach().clone() for p in params];values=[];epsilon=1e-5
        with torch.no_grad():
            for sign in (1,-1):
                for p,x,d in zip(params,saved,direction):p.copy_(x+sign*epsilon*d)
                values.append(float(F.cross_entropy(logits(model,source,prefix).flatten(0,1),labels.flatten())))
            for p,x in zip(params,saved):p.copy_(x)
        self.assertAlmostEqual(analytical,(values[0]-values[1])/(2*epsilon),places=7)

    def test_decisions_and_deliberate_fit_error(self):
        from copy_rope_training import decision
        from audit_copy_rope_training import own_decision
        runs=[]
        for seed in SEEDS:
            for arm in ('sinusoidal','rope'):
                metrics={}
                for split in ('train','test'):
                    for n in ((3,4,5,6) if split=='train' else (3,4,5,6,8,10)):
                        value=1. if n<=6 or arm=='rope' else 0.
                        metrics[f'{split}_{n}']={g:dict(exact_accuracy=value) for g in (('all','repeated') if n==10 else ('all','distinct','repeated'))}
                runs.append(dict(seed=seed,arm=arm,status='trained',metrics=metrics))
        self.assertEqual(decision(runs),own_decision(runs));self.assertTrue(decision(runs)['rope_benefit'])
        runs[-1]['metrics']['train_6']['repeated']['exact_accuracy']=.5
        self.assertFalse(decision(runs)['rope_benefit']);self.assertEqual(decision(runs),own_decision(runs))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CopyRopeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful() and '--record' in sys.argv[1:]:
        assert DIRECTORY.exists() and not (DIRECTORY/'preflight.json').exists()
        write(DIRECTORY/'preflight.json',dict(passed=True,unittest_methods=result.testsRun,initialized_cases=ERRORS,
            source_sha256=digest(Path(__file__)),final_evaluator_and_audit_cell_exercised=True,
            absent_group_reload_and_supplied_logits_checked=True,training_performed=False))
    sys.exit(not result.wasSuccessful())
