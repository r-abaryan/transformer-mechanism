"""Saved intervention/audit, exact value-basis identity and inverse faults."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import fitting_representation as s
import association_load


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_factorization_and_basis_controls(self):
        with torch.random.fork_rng(devices=[]):
            first=association_load.pair(109)['delta'];second=association_load.pair(113)['delta']
        donors=dict(lbfgs=first,negative_gradient=second)
        pools=association_load.generate(109,confirmation=True,base_counts=dict(train=1,validation=1,test=1))['pools']
        e,q,y,r=s.reference.independent.inputs(pools['train_32'],'all_queries')
        for route,decoder in [(x,z) for x in s.ARMS for z in s.ARMS]:
            m=s.hybrid(donors,route,decoder);z=s.training.forward_queries(m,e,q)
            torch.testing.assert_close(z,s.factorized(m,e,q),atol=1e-10,rtol=1e-10)
            # Transform only the decoder donor's private value coordinates.
            rotated=copy.deepcopy(donors)
            permutation=torch.tensor([2,0,6,1,7,3,5,4])
            with torch.no_grad():
                v=rotated[decoder].value.weight;w=rotated[decoder].output.weight
                v.copy_(v[:,permutation]);w.copy_(w[:,permutation])
            other=s.hybrid(rotated,route,decoder)
            torch.testing.assert_close(z,s.training.forward_queries(other,e,q),atol=1e-10,rtol=1e-10)
            wrong=m.value.weight.detach().clone()
            with torch.no_grad():m.value.weight.zero_()
            self.assertGreater(float((z-s.training.forward_queries(m,e,q)).abs().max().detach()),1e-6)
        # Supplied tie logits, empty repeated-key group.
        metrics=s.reference.metrics(torch.zeros(1,2,6,dtype=torch.float64),torch.tensor([[0,1]]),torch.zeros(1,2,dtype=torch.bool))
        self.assertEqual(metrics['groups']['all']['accuracy'],.5);self.assertIsNone(metrics['groups']['repeat']['ce'])

    def test_saved_runtime_and_faults(self):
        folder=ROOT/'results/test_runs'/('fitting_representation_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        with torch.random.fork_rng(devices=[]):
            donors={a:association_load.pair(seed)['delta'] for a,seed in zip(s.ARMS,(127,131))}
        checkpoints={}
        for arm,m in donors.items():
            p=folder/(arm+'.pt');torch.save(dict(model=m.state_dict()),p);checkpoints[arm]=p.relative_to(ROOT).as_posix()
        case=dict(seed=127,checkpoints=checkpoints)
        pools=association_load.generate(127,confirmation=True,base_counts=dict(train=1,validation=1,test=1))['pools'];out=folder/'study'
        rng=torch.get_rng_state().clone();s.execute(out,[case],pools,fixture=True)
        self.assertTrue(s.audit(out,persist=False)['passed']);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        report=json.loads((out/'report.json').read_text());raw=(out/'report.json').read_bytes();path=out/'scores.pt';original=path.read_bytes()
        for field in ('labels','logits','missing_cell'):
            scores=torch.load(path,weights_only=True)
            if field=='missing_cell':scores.pop()
            else:scores[0][field].flatten()[0]+=1
            torch.save(scores,path);wrong=copy.deepcopy(report);wrong['scores_sha256']=s.replay.digest(path);s.replay.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError,msg=field):s.audit(out,persist=False)
            path.write_bytes(original);(out/'report.json').write_bytes(raw)
        # Same saved model shape, wrong donor parameter.
        p=ROOT/report['cells'][0]['checkpoint'];original=p.read_bytes();state=torch.load(p,weights_only=True)
        state['model']['key.weight'].add_(1);torch.save(state,p)
        with self.assertRaisesRegex(AssertionError,'Wrong parameter donor'):s.audit(out,persist=False)
        p.write_bytes(original)
        for change in [lambda d:d.__setitem__('evaluation_targets',d['evaluation_targets']+1),
                       lambda d:d['summary'].__setitem__('all_seed_routing_sufficient',not d['summary']['all_seed_routing_sufficient']),
                       lambda d:d['summary']['seeds'][0].__setitem__('decoder_sufficient',not d['summary']['seeds'][0]['decoder_sufficient'])]:
            wrong=copy.deepcopy(report);change(wrong);s.replay.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError):s.audit(out,persist=False)
            (out/'report.json').write_bytes(raw)


if __name__=='__main__':unittest.main()
