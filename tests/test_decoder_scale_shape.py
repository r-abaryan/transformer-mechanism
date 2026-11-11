"""Actual saved scorer/audit, weight projection and scaled-classifier controls."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import decoder_scale_shape as s
import association_load


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_projection_transform_and_actual_forward(self):
        with torch.random.fork_rng(devices=[]):base=association_load.pair(137)['delta']
        bigger=copy.deepcopy(base)
        with torch.no_grad():bigger.output.weight.mul_(3);bigger.output.bias.mul_(3)
        states=dict(lbfgs=bigger.state_dict(),negative_gradient=base.state_dict())
        p=s.projection(states);self.assertAlmostEqual(p['scale'],3);self.assertLess(p['relative_residual'],1e-15)
        self.assertAlmostEqual(s.independent_projection(states)[0],3)
        rows=association_load.generate(137,confirmation=True,base_counts=dict(train=1,validation=1,test=1))['pools']['train_32']
        e,q,y,r=s.previous.reference.independent.inputs(rows,'all_queries')
        with torch.no_grad():
            z=s.previous.training.forward_queries(base,e,q);l=s.previous.training.forward_queries(bigger,e,q)
            actual=s.transform(z,l,3)
            torch.testing.assert_close(actual['scale'],l-l.mean(-1,keepdim=True),atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(actual['shape'],actual['base'],atol=1e-10,rtol=1e-10)
        tied=torch.zeros(1,2,6,dtype=torch.float64);labels=torch.tensor([[0,1]]);repeat=torch.zeros(1,2,dtype=torch.bool)
        variants=s.transform(tied,tied,3)
        for logits in variants.values():
            m=s.previous.reference.metrics(logits,labels,repeat)
            self.assertEqual(m['groups']['all']['accuracy'],.5);self.assertIsNone(m['groups']['repeat']['ce'])
        with self.assertRaises(AssertionError):s.transform(z,l,-1)
        wrong=copy.deepcopy(states)
        wrong['lbfgs']['output.weight'].neg_();wrong['lbfgs']['output.bias'].neg_()
        with self.assertRaises(AssertionError):s.projection(wrong)

    def test_saved_runtime_and_faults(self):
        folder=ROOT/'results/test_runs'/('decoder_scale_shape_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        with torch.random.fork_rng(devices=[]):
            base=association_load.pair(139)['delta'];bigger=copy.deepcopy(base)
        with torch.no_grad():
            bigger.output.weight.mul_(2.5);bigger.output.bias.mul_(2.5)
            bigger.key.weight[0,0].add_(.2);bigger.output.weight[0,0].add_(.1)
        cases=[];checkpoints={}
        for arm,m in [('lbfgs',bigger),('negative_gradient',base)]:
            path=folder/(arm+'.pt');torch.save(dict(model=m.state_dict()),path);checkpoints[arm]=path.relative_to(ROOT).as_posix()
        cases=[dict(seed=139,checkpoints=checkpoints)]
        pools=association_load.generate(139,confirmation=True,base_counts=dict(train=1,validation=1,test=1))['pools']
        rng=torch.get_rng_state().clone();source_folder=folder/'source'
        s.previous.execute(source_folder,cases,pools,fixture=True)
        source=torch.load(source_folder/'scores.pt',weights_only=True);out=folder/'study'
        s.execute(out,cases,pools,source,fixture=True)
        self.assertTrue(s.audit(out,persist=False)['passed']);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        report=json.loads((out/'report.json').read_text());raw=(out/'report.json').read_bytes()
        path=out/'scores.pt';original=path.read_bytes()
        for field in ('labels','logits','missing_cell'):
            scores=torch.load(path,weights_only=True)
            if field=='missing_cell':scores.pop()
            else:scores[0][field].flatten()[0]+=1
            torch.save(scores,path);bad=copy.deepcopy(report);bad['scores_sha256']=s.replay.digest(path);s.replay.write(out/'report.json',bad)
            with self.assertRaises(AssertionError,msg=field):s.audit(out,persist=False)
            path.write_bytes(original);(out/'report.json').write_bytes(raw)
        changes=[lambda d:d['projections'][0].__setitem__('scale',d['projections'][0]['scale']+1),
                 lambda d:d.__setitem__('evaluation_targets',d['evaluation_targets']+1),
                 lambda d:d['summary'].__setitem__('all_primary_scale_dominates',not d['summary']['all_primary_scale_dominates']),
                 lambda d:d['summary']['primary'].pop()]
        for change in changes:
            bad=copy.deepcopy(report);change(bad);s.replay.write(out/'report.json',bad)
            with self.assertRaises(AssertionError):s.audit(out,persist=False)
            (out/'report.json').write_bytes(raw)


if __name__=='__main__':unittest.main()
