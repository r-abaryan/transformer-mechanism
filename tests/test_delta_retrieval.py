"""Reusable correctness and decision controls for compact retrieval readers."""
import copy
import hashlib
import sys
import unittest
import uuid
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import delta_retrieval as study


class DeltaRetrievalTests(unittest.TestCase):
    def test_equations_all_parameter_gradients_and_direction(self):
        torch.set_num_threads(1)
        for seed in (3271,3277,3299):
            rng=torch.Generator().manual_seed(seed+1)
            for n in (4,8,16):
                e=torch.stack((torch.randint(8,(3,n),generator=rng),torch.randint(6,(3,n),generator=rng)),dim=-1)
                q=torch.tensor([0,2,5]);y=torch.tensor([0,2,4])
                for arm,model in study.pair(seed).items():
                    other=copy.deepcopy(model)
                    a=model(e,q);b=study.reference(other,e,q)
                    self.assertLess((a-b).abs().max().item(),1e-10)
                    F.cross_entropy(a,y).backward();F.cross_entropy(b,y).backward()
                    for p,r in zip(model.parameters(),other.parameters()):
                        self.assertIsNotNone(p.grad);self.assertIsNotNone(r.grad)
                        self.assertLess((p.grad-r.grad).abs().max().item(),1e-10)
                    if n==8:
                        directions=[torch.randn(p.shape,generator=rng,dtype=p.dtype) for p in model.parameters()]
                        analytic=sum((p.grad*d).sum() for p,d in zip(model.parameters(),directions)).item()
                        variants=[]
                        for sign in (-1,1):
                            changed=copy.deepcopy(model)
                            with torch.no_grad():
                                for p,d in zip(changed.parameters(),directions):p.add_(sign*1e-6*d)
                            variants.append(F.cross_entropy(changed(e,q),y).item())
                        self.assertAlmostEqual(analytic,(variants[1]-variants[0])/2e-6,places=7)

    def test_wrong_updates_wrong_recency_and_batch_isolation(self):
        e=torch.tensor([[[0,1],[1,4],[0,3],[2,2]],[[1,2],[1,4],[1,0],[3,1]]]);q=torch.tensor([0,1])
        for model in study.pair(3271).values():
            actual=model(e,q);wrong=study.reference(model,e,q,wrong=True)
            self.assertGreater((actual-wrong).abs().max().item(),1e-4)
            self.assertTrue(torch.allclose(actual,torch.cat([model(a,b) for a,b in zip(e.split(1),q.split(1))]),atol=1e-12,rtol=1e-12))
            self.assertTrue(torch.equal(actual,model.read(model.prefill(e),q)))

    def test_labels_families_and_baselines(self):
        data=study.generate(2026100407)
        baseline=study.audit_data(data)
        self.assertEqual(sum(len(v) for v in data['pools'].values()),1920)
        for value in baseline.values():
            self.assertEqual(value['repeat']['first'],0)
            self.assertEqual(value['unique']['first'],value['unique']['n'])
        wrong=copy.deepcopy(data);wrong['pools']['train_4'][0]['label']=(wrong['pools']['train_4'][0]['label']+1)%6
        with self.assertRaisesRegex(AssertionError,'Wrong label'):study.audit_data(wrong)
        leak=copy.deepcopy(data);leak['pools']['validation_4'][0]=leak['pools']['train_4'][0]
        with self.assertRaises(AssertionError):study.audit_data(leak)

    def test_supplied_logits_absent_groups_and_initialized_audit_reload(self):
        rows=study.generate(2026100407)['pools']['train_4'][:6]
        y=torch.tensor([r['label'] for r in rows]);scores=torch.zeros(6,6,dtype=torch.float64)
        scores[torch.arange(6),y]=4
        m=study.metrics(scores,rows);self.assertEqual(m['unique']['accuracy'],1);self.assertNotIn('repeat',m)
        self.assertEqual(study.metrics(scores.roll(1,-1),rows)['all']['accuracy'],0)
        folder=ROOT/'results/test_runs'/('delta_preflight_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
        payload=dict(pools={'train_4':rows})
        for arm,model in study.pair(3271).items():
            cp=folder/f'{arm}.pt';torch.save(dict(model=model.state_dict(),step=0),cp)
            measured,_=study.evaluate(model,rows)
            run=dict(status='complete',phase='development',seed=3271,arm=arm,recipe='constant',
                     checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=study.digest(cp),
                     initial_sha256=study.state_hash(model.state_dict()),stream_sha256=hashlib.sha256().hexdigest(),
                     metrics={'train_4':measured})
            checked=study.audit(dict(runs=[run]),{'development':payload},expected_steps=0)
            self.assertTrue(checked['passed']);self.assertEqual(checked['predictions'],6)

    def test_selection_and_confirmation_negative_controls(self):
        metrics={p:{g:dict(accuracy=1.,loss=.01) for g in study.GROUPS} for p in ('train_4','train_8','validation_4','validation_8','test_4','test_8','test_16','test_32')}
        runs=[dict(arm=a,seed=s,recipe='constant',status='complete',metrics=copy.deepcopy(metrics)) for a in study.ARMS for s in study.DEV_SEEDS]
        self.assertEqual(study.selection(runs)['delta']['selected_recipe'],'constant')
        runs[-1]['metrics']['train_8']['repeat']['accuracy']=.98
        self.assertIsNone(study.selection(runs)['delta']['selected_recipe'])
        self.assertIsNone(study.selection([])['delta']['selected_recipe'])
        confirm=[dict(phase='confirmation',arm=a,seed=s,status='complete',metrics=copy.deepcopy(metrics)) for a in study.ARMS for s in study.CONFIRM_SEEDS]
        self.assertTrue(study.confirm_decision(confirm)['quality_preservation'])
        confirm[-1]['metrics']['test_32']['unique']['accuracy']=.97
        self.assertFalse(study.confirm_decision(confirm)['quality_preservation'])
        self.assertFalse(study.confirm_decision([])['softmax']['training_fit'])


if __name__=='__main__':unittest.main()
