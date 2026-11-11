"""Source-format contract and actual evaluator/audit preflight before training."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from copy_marker_model import make_model,reference,marked_source
from copy_marker_training import DIRECTORY,SEEDS,evaluate,decision
from audit_copy_marker_training import audit_cell,own_decision
from encoder_decoder_copy import inputs,logits
from copy_length_exposure import metrics
from recurrent_retrieval import write
from position_distractors import digest

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
CASES=[]


class MarkerTests(unittest.TestCase):
    def test_formats_targets_and_error_controls(self):
        content=torch.tensor([[3,4,5],[3,3,3]])
        prefix,labels=inputs(content)
        self.assertEqual(marked_source(content).tolist(),[[3,4,5,2],[3,3,3,2]])
        self.assertEqual(labels.tolist(),[[3,4,5,2],[3,3,3,2]])
        self.assertEqual(prefix.tolist(),[[1,3,4,5],[1,3,3,3]])
        # Wrongly copying the appended marker and then adding another EOS changes targets.
        wrong=torch.cat((marked_source(content),torch.full((2,1),2,dtype=torch.long)),1)
        self.assertNotEqual(wrong.tolist(),labels.tolist())
        for arm in ('sinusoidal','rope'):
            torch.manual_seed(3163);model=make_model(arm)
            s,t=model.features(content,prefix);sp=torch.zeros_like(content,dtype=torch.bool);tp=torch.zeros_like(prefix,dtype=torch.bool)
            self.assertEqual(s.shape[1],4);self.assertEqual(t.shape[1],4)
            correct=model(s,t,sp,tp)
            pad=torch.cat((sp,torch.ones(2,1,dtype=torch.bool)),1)
            masked=model.base(s,t,pad,tp)
            self.assertGreater(float((correct-masked).detach().abs().max()),1e-6)
            with self.assertRaisesRegex(ValueError,'Empty content'):marked_source(content[:,:0])

    def test_equations_gradients_checkpoint_and_actual_audit(self):
        for seed in SEEDS:
            for n in (3,6,10):
                for arm in ('sinusoidal','rope'):
                    with self.subTest(seed=seed,length=n,arm=arm):
                        torch.manual_seed(seed);model=make_model(arm)
                        self.assertEqual(sum(p.numel() for p in model.parameters()),1888)
                        rows=[list(range(3,3+n)) if n<=8 else [3,4]*5,[3]*n]
                        content=torch.tensor(rows);prefix,labels=inputs(content)
                        z=logits(model,content,prefix);rz=reference(model,content,prefix,arm)
                        ga=torch.autograd.grad(F.cross_entropy(z.flatten(0,1),labels.flatten()),tuple(model.parameters()))
                        gb=torch.autograd.grad(F.cross_entropy(rz.flatten(0,1),labels.flatten()),tuple(model.parameters()))
                        le=float((z-rz).detach().abs().max());ge=max(float((a-b).abs().max()) for a,b in zip(ga,gb))
                        self.assertLessEqual(le,1e-9);self.assertLessEqual(ge,1e-9)
                        source_index=[i for i,(name,p) in enumerate(model.named_parameters()) if name=='base.source.weight'][0]
                        self.assertGreater(float(ga[source_index][2].abs().max()),0.)
                        # A regular workspace path avoids the previous Windows temp-directory issue.
                        cp=DIRECTORY/f'preflight_{seed}_{n}_{arm}.pt'
                        if '--record' not in sys.argv[1:]:cp=DIRECTORY/f'recheck_{seed}_{n}_{arm}.pt'
                        torch.save(model.state_dict(),cp)
                        loaded=make_model(arm);loaded.load_state_dict(torch.load(cp,weights_only=True))
                        met,stored=evaluate(loaded,rows);_,cell=audit_cell(loaded,arm,rows,met,stored)
                        if n==10:self.assertEqual(set(met),{'all','repeated'})
                        CASES.append(dict(seed=seed,length=n,arm=arm,initial_logit_error=le,gradient_error=ge,**cell))

    def test_known_logits_and_decisions(self):
        rows=[[3,4]*5,[3]*10];z=torch.full((2,11,16),-10.,dtype=torch.float64)
        for i,row in enumerate(rows):
            for j,y in enumerate(row+[2]):z[i,j,y]=10.
        met=metrics(z,rows,[row+[2] for row in rows])
        self.assertEqual(set(met),{'all','repeated'});self.assertEqual(met['repeated']['exact_accuracy'],1.)
        runs=[]
        for seed in SEEDS:
            for arm in ('sinusoidal','rope'):
                pools={}
                for split in ('train','test','uniform'):
                    for n in ((3,4,5,6) if split=='train' else (3,4,5,6,8,10)):
                        value=1. if split!='test' or n<=6 or arm=='rope' else 0.
                        groups=('all','repeated') if split=='uniform' or n==10 else ('all','distinct','repeated')
                        pools[f'{split}_{n}']={g:dict(exact_accuracy=value) for g in groups}
                runs.append(dict(seed=seed,arm=arm,status='trained',metrics=pools))
        self.assertEqual(decision(runs),own_decision(runs));self.assertTrue(decision(runs)['rope_benefit'])
        runs[-1]['metrics']['train_6']['repeated']['exact_accuracy']=.5
        self.assertEqual(decision(runs),own_decision(runs));self.assertFalse(decision(runs)['rope_benefit'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MarkerTests))
    if result.wasSuccessful() and '--record' in sys.argv[1:]:
        target=DIRECTORY/'preflight.json';assert not target.exists()
        write(target,dict(passed=True,unittest_methods=result.testsRun,initialized_cases=CASES,
            expected_initialized_cases=len(SEEDS)*3*2,actual_initialized_cases=len(CASES),
            source_sha256=digest(Path(__file__)),final_evaluator_and_audit_cell_exercised=True,
            absent_group_reload_and_supplied_logits_checked=True,training_performed=False))
    sys.exit(not result.wasSuccessful())
