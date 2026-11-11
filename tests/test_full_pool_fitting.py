"""Saved runtime, fault rejection and full-objective weighting checks."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import full_pool_fitting as study
import audit_full_pool_fitting as audit
import association_load


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_saved_runtime_and_faults(self):
        folder=ROOT/'results/test_runs'/('full_fit_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        model=association_load.pair(101)['delta'];opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        path=folder/'initial.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0),path)
        pools=association_load.generate(101,confirmation=True,base_counts=dict(train=2,validation=1,test=1))['pools']
        case=dict(seed=101,checkpoint=path.relative_to(ROOT).as_posix());out=folder/'study'
        rng=torch.get_rng_state().clone();study.execute(out,[case],pools,budget=4,ratio=4,interval=2,fixture=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(audit.audit(out,persist=False)['passed'])
        report=json.loads((out/'report.json').read_text());raw=(out/'report.json').read_bytes()
        self.assertLessEqual(report['runs'][0]['closure_evaluations'],4)
        for field,change in [('loss',lambda t:t['calls'][0].__setitem__('loss',t['calls'][0]['loss']+1)),
                             ('gradient',lambda t:t['calls'][0]['gradients']['write_logit'].add_(1)),
                             ('optimizer',lambda t:next(iter(t['steps'][0]['optimizer']['state'].values())).__setitem__('func_evals',999))]:
            p=ROOT/report['runs'][0]['trace'];original=p.read_bytes();trace=torch.load(p,weights_only=True);change(trace);torch.save(trace,p)
            wrong=copy.deepcopy(report);wrong['runs'][0]['trace_sha256']=study.prior.training.replay.digest(p);study.prior.training.replay.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError,msg=field):audit.audit(out,persist=False)
            p.write_bytes(original);(out/'report.json').write_bytes(raw)
        for field in ('labels','logits'):
            p=out/'scores.pt';original=p.read_bytes();scores=torch.load(p,weights_only=True);scores[0][field].flatten()[0]+=1;torch.save(scores,p)
            wrong=copy.deepcopy(report);wrong['scores_sha256']=study.prior.training.replay.digest(p);study.prior.training.replay.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError,msg=field):audit.audit(out,persist=False)
            p.write_bytes(original);(out/'report.json').write_bytes(raw)
        for defect in (lambda r:r['cells'].pop(),lambda r:r.__setitem__('answer_targets',r['answer_targets']+1),
                       lambda r:r['decision']['lbfgs'].__setitem__('all_seed_fit',not r['decision']['lbfgs']['all_seed_fit'])):
            wrong=copy.deepcopy(report);defect(wrong);study.prior.training.replay.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
            (out/'report.json').write_bytes(raw)
        # Actual Adam parameters and streams have semantic checks beyond hashes.
        p=ROOT/report['runs'][1]['trace'];original=p.read_bytes();trace=torch.load(p,weights_only=True);trace[0]['after']['value.weight'].add_(1);torch.save(trace,p)
        wrong=copy.deepcopy(report);wrong['runs'][1]['trace_sha256']=study.prior.training.replay.digest(p);study.prior.training.replay.write(out/'report.json',wrong)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        p.write_bytes(original);(out/'report.json').write_bytes(raw)

    def test_objective_weighting_labels_ties_and_fit_boundary(self):
        model=association_load.pair(103)['delta'];pools=association_load.generate(103,confirmation=True,base_counts=dict(train=1,validation=1,test=1))['pools']
        table=study.tables(pools);model.zero_grad(set_to_none=True);loss=study.objective(model,table)
        other={m:audit.reference.independent.inputs(pools[f'train_{m}'],'all_queries') for m in study.LOADS}
        value,grad=audit.independent_gradient(model,other);self.assertAlmostEqual(loss,value,places=10)
        for n,p in model.named_parameters():torch.testing.assert_close(p.grad,grad[n],atol=1e-10,rtol=1e-8)
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);repeat=torch.zeros_like(y,dtype=torch.bool)
        m=audit.reference.metrics(z,y,repeat);self.assertEqual(m['groups']['all']['accuracy'],.5);self.assertIsNone(m['groups']['repeat']['ce'])
        metrics={str(m):dict(groups=dict(all=dict(count=10000,accuracy=.99),unique=dict(count=10000 if m==32 else 0,accuracy=.99 if m==32 else None),repeat=dict(count=0 if m==32 else 10000,accuracy=None if m==32 else .99))) for m in study.LOADS}
        self.assertTrue(study.fit(metrics));metrics['32']['groups']['all']['accuracy']=.9899;metrics['32']['groups']['unique']['accuracy']=.9899
        self.assertFalse(study.fit(metrics));metrics.pop('16')
        with self.assertRaises(AssertionError):study.fit(metrics)


if __name__=='__main__':unittest.main()
