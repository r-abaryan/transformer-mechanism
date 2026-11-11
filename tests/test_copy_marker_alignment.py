"""Initialized preflight, selectors and deliberate errors for the saved probe."""
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from copy_marker_alignment import DIRECTORY,SEEDS,MODES,POOLS,route,patched,evaluate,details,decision
from copy_marker_model import make_model
from encoder_decoder_copy import inputs,logits
from audit_copy_marker_alignment import reference,audit_cell,own_details,compare,own_decision
from recurrent_retrieval import write
from position_distractors import digest

torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
CASES=[]


class AlignmentTests(unittest.TestCase):
    def test_selector_boundaries_and_shift_error(self):
        raw=torch.arange(2*2*10*7,dtype=torch.float64).reshape(2,2,10,7).softmax(-1)
        aligned=route(raw,'aligned');wrong=route(raw,'shifted')
        self.assertTrue(torch.equal(aligned[:,:,:7],torch.eye(7,dtype=torch.float64).expand(2,2,7,7)))
        self.assertTrue(torch.equal(aligned[:,:,7:],raw[:,:,7:]))
        self.assertTrue(torch.equal(aligned[:,:,6,6],torch.ones(2,2,dtype=torch.float64)))
        self.assertFalse(torch.equal(aligned,wrong))
        self.assertTrue(torch.equal(route(raw,'native'),raw))
        with self.assertRaises(ValueError):route(raw,'bad')

    def test_known_logits_errors_and_absent_groups(self):
        rows=[[3,4]*5,[3]*10];z=torch.full((2,11,16),-10.,dtype=torch.float64)
        for i,row in enumerate(rows):
            for j,y in enumerate(row+[2]):z[i,j,y]=10.
        raw=torch.eye(11,dtype=torch.float64).expand(2,2,11,11)
        d=details(z,rows,raw);compare(own_details(z.tolist(),rows,raw.tolist()),d)
        self.assertEqual(set(d),{'all','repeated'});self.assertEqual(d['all']['first_error_counts'][-1],2)
        z[0,2,2]=30.;wrong=details(z,rows,raw);compare(own_details(z.tolist(),rows,raw.tolist()),wrong)
        self.assertEqual(wrong['all']['first_error_counts'][2],1)
        self.assertEqual(wrong['all']['positions'][2]['eos_top'],1)
        self.assertNotEqual(d['all']['first_error_counts'],wrong['all']['first_error_counts'])

    def test_equations_gradients_reload_and_actual_audit(self):
        for seed in SEEDS:
            for n in (6,8,10):
                torch.manual_seed(seed);model=make_model('rope').eval()
                rows=[list(range(3,3+n)) if n<=8 else [3,4]*5,[3]*n]
                source=torch.tensor(rows);prefix,labels=inputs(source)
                cp=DIRECTORY/f'preflight_{seed}_{n}.pt';torch.save(model.state_dict(),cp)
                loaded=make_model('rope').eval();loaded.load_state_dict(torch.load(cp,weights_only=True))
                for mode in MODES:
                    with self.subTest(seed=seed,length=n,mode=mode):
                        z,w=patched(model,source,prefix,mode);rz,rw=reference(model,source,prefix,mode)
                        ga=torch.autograd.grad(F.cross_entropy(z.flatten(0,1),labels.flatten()),tuple(model.parameters()))
                        gb=torch.autograd.grad(F.cross_entropy(rz.flatten(0,1),labels.flatten()),tuple(model.parameters()))
                        le=max(float((z-rz).detach().abs().max()),float((w-rw).detach().abs().max()))
                        ge=max(float((a-b).abs().max()) for a,b in zip(ga,gb));self.assertLessEqual(le,1e-9);self.assertLessEqual(ge,1e-9)
                        if mode=='native':self.assertLessEqual(float((z-logits(model,source,prefix)).detach().abs().max()),1e-9)
                        result,stored=evaluate(loaded,rows,mode);_,check=audit_cell(loaded,rows,mode,stored,result)
                        if n==10:self.assertEqual(set(result['metrics']),{'all','repeated'})
                        CASES.append(dict(seed=seed,length=n,mode=mode,logit_weight_error=le,gradient_error=ge,**check))
                a=patched(model,source,prefix,'aligned')[0];b=patched(model,source,prefix,'shifted')[0]
                self.assertGreater(float((a-b).detach().abs().max()),1e-6)

    def test_decision_negative_control(self):
        cells=[]
        for seed in SEEDS:
            for p in POOLS:
                for mode in MODES:
                    gs=('all','repeated') if p.startswith('uniform') or p.endswith('10') else ('all','distinct','repeated')
                    long=p in ('test_8','test_10');value=1. if mode=='aligned' or not long else 0.
                    cells.append(dict(seed=seed,pool=p,mode=mode,metrics={g:dict(oracle_content_exact=value,exact_accuracy=value) for g in gs}))
        self.assertEqual(decision(cells),own_decision(cells));self.assertTrue(decision(cells)['conditional_recovery'])
        cell=next(c for c in cells if (c['seed'],c['pool'],c['mode'])==(SEEDS[0],'test_8','aligned'))
        cell['metrics']['distinct']['oracle_content_exact']=0.
        self.assertEqual(decision(cells),own_decision(cells));self.assertFalse(decision(cells)['conditional_recovery'])


if __name__=='__main__':
    DIRECTORY.mkdir(exist_ok=True)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AlignmentTests))
    if result.wasSuccessful():
        assert len(CASES)==len(SEEDS)*3*len(MODES)
        write(DIRECTORY/'preflight.json',dict(passed=True,methods=result.testsRun,cases=CASES,expected_cases=len(SEEDS)*3*len(MODES),
            actual_cases=len(CASES),source_sha256=digest(Path(__file__)),evaluator_reload_and_audit_exercised=True,
            shifted_selector_error_detected=True,incorrect_logits_and_criterion_detected=True,absent_group_checked=True))
    sys.exit(not result.wasSuccessful())
