"""Saved training/evaluator/audit, scoped reset and inverse controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'tests'))
import memory_first_moment_reset as study
import audit_memory_first_moment_reset as audit
from test_query_feedback_training import payload

PROFILE={}


class FirstMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_saved_paired_training_complete_profile_evaluator_audit(self):
        data=payload();seed=1753;folder=ROOT/'results/test_runs'/('first_moment_'+uuid.uuid4().hex);folder.mkdir()
        initial=folder/'initial.pt';model=study.training.model_source.pair(seed)['delta'];opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0,
            history_rng=torch.Generator().manual_seed(seed+10000).get_state(),query_rng=torch.Generator().manual_seed(seed+20000).get_state()),initial)
        source=study.transport.replay_source.replay_cycle(initial,data['pools'],seed,0)[-1]
        path=folder/'start.pt';torch.save(study.start_payload(source),path);runs={}
        for arm in ('control','reset'):
            traces,_=study.train(path,data['pools'],seed,arm,7,folder);runs[arm]=traces
            audit.audit_run(path,data['pools'],traces,arm)
            checkpoint=folder/f'{seed}_{arm}_7.pt';cell=study.evaluate(checkpoint,data['pools']['train_32'])
            mask=cell['logits'].argmax(-1)!=cell['labels'];cell['fixed']=study.transport.fixed_metrics(cell['logits'],cell['labels'],mask)
            audit.audit_evaluation(checkpoint,data['pools']['train_32'],cell,cell,mask)
        audit.paired_scope(runs['control'],runs['reset'])
        first=runs['control'][0]
        PROFILE.update(device='cpu',dtype='float64',batch=64,context_writes=32,profiled_targets=first['labels'].numel(),
            full_training_step_seconds=first['step_seconds'],model_tensor_bytes=sum(v.numel()*v.element_size() for v in first['before_model'].values()),
            optimizer_tensor_bytes=sum(v.numel()*v.element_size() for s in first['before_optimizer']['state'].values() for v in s.values() if isinstance(v,torch.Tensor)),
            gpu_peak_allocated_bytes=0,gpu_peak_reserved_bytes=0,scope='One complete synthetic CPU forward/backward/Adam step; tensor payloads, not measured CPU peak memory or a throughput benchmark')
        bad=copy.deepcopy(runs['reset']);pid=bad[0]['before_optimizer']['param_groups'][0]['params'][0]
        bad[0]['before_optimizer']['state'][pid]['exp_avg_sq'].zero_()
        with self.assertRaisesRegex(AssertionError,'Reset scope'):audit.audit_run(path,data['pools'],bad,'reset')
        bad=copy.deepcopy(runs['control']);bad[0]['draws'][0,0]=(int(bad[0]['draws'][0,0])+1)%32
        with self.assertRaisesRegex(AssertionError,'stream'):audit.audit_run(path,data['pools'],bad,'control')
        bad=copy.deepcopy(runs['control']);bad[0]['gradients']['value.weight'][0,0]+=.1
        with self.assertRaisesRegex(AssertionError,'gradient'):audit.audit_run(path,data['pools'],bad,'control')
        bad=copy.deepcopy(runs['control']);bad[0]['after_model']['key.weight'][0,0]+=.1
        with self.assertRaisesRegex(AssertionError,'Adam'):audit.audit_run(path,data['pools'],bad,'control')
        bad=copy.deepcopy(cell);bad['labels'][0,0]=(int(bad['labels'][0,0])+1)%6
        with self.assertRaises(AssertionError):audit.audit_evaluation(checkpoint,data['pools']['train_32'],bad,bad,mask)

    def test_supplied_logits_missing_groups_and_fixed_targets(self):
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);repeat=torch.zeros_like(y,dtype=torch.bool)
        actual=study.training.reduce_scores(z,y,repeat);audit.reference.check_metrics(actual,audit.reference.metrics(z,y,repeat))
        self.assertIsNone(actual['groups']['repeat']['ce'])
        fixed=study.transport.fixed_metrics(z,y,z.argmax(-1)!=y);audit.prior.compare(fixed,audit.transport_audit.fixed(z,y,z.argmax(-1)!=y))


if __name__=='__main__':unittest.main()
