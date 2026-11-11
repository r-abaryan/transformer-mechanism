"""Native control and actual saved training/evaluator/audit with intentional faults."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import train_fitting_transport as study
import audit_fitting_transport_training as audit
import association_load


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_native_and_block_operator(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__();self.write_logit=torch.nn.Parameter(torch.tensor(.8,dtype=torch.float64))
                self.key=torch.nn.Embedding(1,1,dtype=torch.float64);self.value=torch.nn.Embedding(1,1,dtype=torch.float64)
                self.output=torch.nn.Linear(1,1,dtype=torch.float64)
                with torch.no_grad():
                    self.key.weight.fill_(.2);self.value.weight.fill_(.3);self.output.weight.fill_(.4);self.output.bias.fill_(.1)
        matrix=torch.diag(torch.arange(1,6,dtype=torch.float64));matrix[0,2]=matrix[2,0]=.7
        paths=[]
        for mode in ('native',)+study.ARMS:
            with torch.random.fork_rng(devices=[]):model=Quadratic()
            opt=study.previous.prior.make_lbfgs(model) if mode=='native' else study.make_optimizer(model,mode)
            records=[]
            for _ in range(4):
                before=torch.cat([p.detach().flatten() for p in model.parameters()]);g=matrix@before
                def closure():
                    opt.zero_grad(set_to_none=True);x=torch.cat([p.flatten() for p in model.parameters()]);loss=x@(matrix@x)/2
                    loss.backward();return loss
                opt.step(closure);state=next(iter(opt.state.values()))
                if mode=='block_diagonal':
                    expected=audit.dense_direction(g,state,list(dict(model.named_parameters())),dict(model.named_parameters()))
                    torch.testing.assert_close(state['d'],expected,atol=1e-10,rtol=1e-8)
                records.append(dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict())))
            paths.append(records)
        self.assertTrue(study.replay.equal(paths[0],paths[1]));self.assertTrue(study.replay.equal(paths[1][0],paths[2][0]))
        self.assertFalse(torch.equal(paths[1][1]['optimizer']['state'][0]['d'],paths[2][1]['optimizer']['state'][0]['d']))

    def test_saved_runtime_and_faults(self):
        folder=ROOT/'results/test_runs'/('fitting_transport_training_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        with torch.random.fork_rng(devices=[]):model=association_load.pair(151)['delta']
        opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        path=folder/'initial.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0),path)
        pools=association_load.generate(151,confirmation=True,base_counts=dict(train=1,validation=1,test=1))['pools']
        case=dict(seed=151,checkpoint=path.relative_to(ROOT).as_posix());out=folder/'study'
        rng=torch.get_rng_state().clone();study.execute(out,[case],pools,budget=8,interval=4,fixture=True)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));result=audit.audit(out,persist=False)
        self.assertTrue(result['passed']);self.assertTrue(result['pair_controls'][0]['later_directions_differ'])
        report=json.loads((out/'report.json').read_text());raw=(out/'report.json').read_bytes()
        path=ROOT/report['runs'][1]['trace'];original=path.read_bytes()
        changes=[('gradient',lambda t:t['calls'][0]['gradients']['write_logit'].add_(1)),
                 ('direction',lambda t:t['steps'][1]['optimizer']['state'][0]['d'].add_(1)),
                 ('counter',lambda t:t['steps'][0]['optimizer']['state'][0].__setitem__('func_evals',999)),
                 ('loss',lambda t:t['calls'][0].__setitem__('loss',t['calls'][0]['loss']+1))]
        for name,change in changes:
            trace=torch.load(path,weights_only=True);change(trace);torch.save(trace,path)
            bad=copy.deepcopy(report);bad['runs'][1]['trace_sha256']=study.replay.digest(path);study.replay.write(out/'report.json',bad)
            with self.assertRaises(AssertionError,msg=name):audit.audit(out,persist=False)
            path.write_bytes(original);(out/'report.json').write_bytes(raw)
        path=out/'scores.pt';original=path.read_bytes()
        for name in ('labels','logits','missing_group'):
            scores=torch.load(path,weights_only=True)
            if name=='missing_group':scores.pop()
            else:scores[0][name].flatten()[0]+=1
            torch.save(scores,path);bad=copy.deepcopy(report);bad['scores_sha256']=study.replay.digest(path);study.replay.write(out/'report.json',bad)
            with self.assertRaises(AssertionError,msg=name):audit.audit(out,persist=False)
            path.write_bytes(original);(out/'report.json').write_bytes(raw)
        bad=copy.deepcopy(report);bad['decision']['all_seed_transport_fitting_benefit']=not bad['decision']['all_seed_transport_fitting_benefit']
        study.replay.write(out/'report.json',bad)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        (out/'report.json').write_bytes(raw)
        z=torch.zeros(1,2,6,dtype=torch.float64);labels=torch.tensor([[0,1]]);repeat=torch.zeros_like(labels,dtype=torch.bool)
        metrics=audit.numerical.reference.metrics(z,labels,repeat)
        self.assertEqual(metrics['groups']['all']['accuracy'],.5);self.assertIsNone(metrics['groups']['repeat']['ce'])


if __name__=='__main__':unittest.main()
