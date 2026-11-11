"""Saved-runtime audit, deliberate faults and direction-only optimizer control."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import fitting_direction as study
import audit_fitting_direction as audit
import association_load


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_native_control_and_direction_intervention(self):
        paths=[]
        for mode in ('native',)+study.ARMS:
            p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64))
            opt=study.prior.make_lbfgs(torch.nn.ParameterList([p])) if mode=='native' else study.DirectionLBFGS([p],mode)
            records=[]
            for _ in range(4):
                def closure():
                    opt.zero_grad(set_to_none=True);loss=(p.square()*torch.tensor([1.,9.],dtype=torch.float64)).sum()/2
                    loss.backward();return loss
                opt.step(closure);records.append(dict(parameters=p.detach().clone(),optimizer=copy.deepcopy(opt.state_dict())))
            paths.append(records)
        self.assertTrue(study.replay.equal(paths[0],paths[1]))
        self.assertTrue(study.replay.equal(paths[1][0],paths[2][0]))
        self.assertFalse(torch.equal(paths[1][1]['optimizer']['state'][0]['d'],paths[2][1]['optimizer']['state'][0]['d']))

    def test_saved_runtime_and_faults(self):
        folder=ROOT/'results/test_runs'/('fitting_direction_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        model=association_load.pair(107)['delta'];opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        path=folder/'initial.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0),path)
        pools=association_load.generate(107,confirmation=True,base_counts=dict(train=1,validation=1,test=1))['pools']
        case=dict(seed=107,checkpoint=path.relative_to(ROOT).as_posix());out=folder/'study'
        rng=torch.get_rng_state().clone();study.execute(out,[case],pools,budget=8,interval=4,fixture=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));result=audit.audit(out,persist=False)
        self.assertTrue(result['passed']);self.assertTrue(result['pair_controls'][0]['later_directions_differ'])
        report=json.loads((out/'report.json').read_text());raw=(out/'report.json').read_bytes()
        p=ROOT/report['runs'][1]['trace'];original=p.read_bytes()
        changes=[('gradient',lambda t:t['calls'][0]['gradients']['write_logit'].add_(1)),
                 ('direction',lambda t:t['steps'][0]['optimizer']['state'][0]['d'].add_(1)),
                 ('counter',lambda t:t['steps'][0]['optimizer']['state'][0].__setitem__('func_evals',999)),
                 ('loss',lambda t:t['calls'][0].__setitem__('loss',t['calls'][0]['loss']+1))]
        for name,change in changes:
            trace=torch.load(p,weights_only=True);change(trace);torch.save(trace,p)
            wrong=copy.deepcopy(report);wrong['runs'][1]['trace_sha256']=study.replay.digest(p);study.replay.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError,msg=name):audit.audit(out,persist=False)
            p.write_bytes(original);(out/'report.json').write_bytes(raw)
        p=out/'scores.pt';original=p.read_bytes()
        for name in ('labels','logits','missing_group'):
            cells=torch.load(p,weights_only=True)
            if name=='missing_group':cells.pop()
            else:cells[0][name].flatten()[0]+=1
            torch.save(cells,p);wrong=copy.deepcopy(report);wrong['scores_sha256']=study.replay.digest(p)
            study.replay.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError,msg=name):audit.audit(out,persist=False)
            p.write_bytes(original);(out/'report.json').write_bytes(raw)
        # Supplied logits include ties and an empty repeat group.
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);r=torch.zeros_like(y,dtype=torch.bool)
        metrics=audit.numerical.reference.metrics(z,y,r)
        self.assertEqual(metrics['groups']['all']['accuracy'],.5);self.assertIsNone(metrics['groups']['repeat']['ce'])


if __name__=='__main__':unittest.main()
