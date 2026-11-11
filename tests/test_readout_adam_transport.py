"""Actual saved replay, resume equivalence and momentum-direction inverse controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'tests'))
import readout_adam_transport as study
import audit_readout_adam_transport as audit
from test_query_feedback_training import payload


class AdamTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_initialized_and_warmed_saved_replay_and_inverse_controls(self):
        seed=1601;data=payload();folder=ROOT/'results/test_runs'/('adam_transport_'+uuid.uuid4().hex);folder.mkdir()
        model=study.training.model_source.pair(seed)['delta'];opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        path=folder/'initial.pt'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0,
            history_rng=torch.Generator().manual_seed(seed+10000).get_state(),query_rng=torch.Generator().manual_seed(seed+20000).get_state()),path)
        first=study.replay_cycle(path,data['pools'],seed,0);self.assertEqual(len(audit.audit_cycle(path,data['pools'],first)),4)
        # Reference uninterrupted native loop, including the otherwise unused draws.
        hr=torch.Generator().manual_seed(seed+10000);qr=torch.Generator().manual_seed(seed+20000)
        for step in range(1,9):
            load=(2,8,16,32)[(step-1)%4];rows=data['pools'][f'train_{load}'];e,_,_=study.training.parent.study.parent.tensors(rows)
            q,y=study.training.parent.query_tables(rows);ids=torch.randint(len(rows),(64,),generator=hr);torch.randint(load,(64,load),generator=qr)
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(study.training.forward_queries(model,e[ids],q[ids]).reshape(-1,6),y[ids].flatten());loss.backward();opt.step()
            if step==4:
                warmed=folder/'warm.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=4,history_rng=hr.get_state(),query_rng=qr.get_state()),warmed)
                self.assertTrue(study.training.replay.equal(model.state_dict(),first[-1]['after_model']))
        second=study.replay_cycle(warmed,data['pools'],seed,4);audit.audit_cycle(warmed,data['pools'],second)
        self.assertTrue(study.training.replay.equal(model.state_dict(),second[-1]['after_model']))
        self.assertTrue(study.training.replay.equal(opt.state_dict(),second[-1]['after_optimizer']))
        bad=copy.deepcopy(second);bad[0]['draws'][0,0]=(int(bad[0]['draws'][0,0])+1)%2
        with self.assertRaisesRegex(AssertionError,'stream'):audit.audit_cycle(warmed,data['pools'],bad)
        bad=copy.deepcopy(second[0]);pid=bad['after_optimizer']['param_groups'][0]['params'][-1]
        bad['after_optimizer']['state'][pid]['exp_avg'][0]+=.01
        with self.assertRaisesRegex(AssertionError,'Adam'):audit.scalar_adam(bad)
        bad=copy.deepcopy(second[0]);bad['after_model']['output.weight'][0,0]+=.01
        with self.assertRaisesRegex(AssertionError,'Adam'):audit.scalar_adam(bad)
        logs=[study.beta_record(t) for t in second];logs[0]['gradient']+=.01
        with self.assertRaisesRegex(AssertionError,'Historical'):audit.audit_cycle(warmed,data['pools'],second,logs)
        bad=copy.deepcopy(second[0]);bad['labels'][0,0]=(int(bad['labels'][0,0])+1)%6
        rows=[data['pools']['train_2'][int(i)] for i in bad['indices']]
        with self.assertRaisesRegex(AssertionError,'labels'):audit.matrix_gradients(bad,rows)

    def test_momentum_agreement_and_opposition(self):
        for sign in (-1,1):
            model=study.training.model_source.pair(1613)['delta'];opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
            before=copy.deepcopy(model.state_dict());gradients={}
            for name,p in model.named_parameters():
                g=(p.detach()-p.detach().mean(0,keepdim=True)) if name.startswith('output.') else torch.zeros_like(p)
                gradients[name]=g.clone();p.grad=g.clone()
                opt.state[p]=dict(step=torch.tensor(1.),exp_avg=sign*10*g.clone(),exp_avg_sq=torch.ones_like(p))
            before_opt=copy.deepcopy(opt.state_dict());opt.step()
            t=dict(before_model=before,after_model=copy.deepcopy(model.state_dict()),before_optimizer=before_opt,
                after_optimizer=copy.deepcopy(opt.state_dict()),gradients=gradients,names=list(dict(model.named_parameters())))
            actual=study.transport(before,t['after_model'],before_opt,t['after_optimizer'],gradients,t['names']);expected,_=audit.scalar_adam(t)
            audit.objective.prior.compare(actual,expected)
            self.assertGreater(actual['raw_g'],0);self.assertLess(actual['sgd_rho'],0)
            self.assertGreater(actual['adam_rho'],0) if sign==-1 else self.assertLess(actual['adam_rho'],0)
            wrong=copy.deepcopy(actual);wrong['stored_rho']*=-1
            with self.assertRaisesRegex(AssertionError,'scalar'):audit.objective.prior.compare(wrong,expected)

    def test_supplied_logits_missing_groups(self):
        z=torch.tensor([[[3,0,0,0,0,0],[0,3,0,0,0,0]]],dtype=torch.float64);y=torch.zeros(1,2,dtype=torch.long)
        cell=dict(logits=z,labels=y,repeat=torch.zeros_like(y,dtype=torch.bool))
        actual=study.pressure.summarize(cell);expected=audit.objective.reconstruct(cell);audit.objective.prior.compare(actual,expected)
        self.assertIsNone(actual['repeat']['mean_g']);wrong=copy.deepcopy(actual);wrong['all']['mean_g']+=.1
        with self.assertRaisesRegex(AssertionError,'scalar'):audit.objective.prior.compare(wrong,expected)


if __name__=='__main__':unittest.main()
