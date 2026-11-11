"""Actual saved-state transport/audit, fixed error masks and inverse controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'tests'))
import adam_memory_error_transport as study
import audit_adam_memory_error_transport as audit
from test_query_feedback_training import payload


class AdamMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_saved_fixture_actual_evaluator_audit_and_corruptions(self):
        data=payload();seed=1741;folder=ROOT/'results/test_runs'/('adam_memory_'+uuid.uuid4().hex);folder.mkdir()
        path=folder/'initial.pt';model=study.training.model_source.pair(seed)['delta'];opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0,
            history_rng=torch.Generator().manual_seed(seed+10000).get_state(),query_rng=torch.Generator().manual_seed(seed+20000).get_state()),path)
        trace=study.replay_source.replay_cycle(path,data['pools'],seed,0)[0]
        saved,_=study.evaluate_trace(trace,data['pools']['train_32']);audit.audit_cell(trace,data['pools']['train_32'],saved,saved['record'],batch=2)
        for field in ('labels','error_mask'):
            bad=copy.deepcopy(saved);t=bad['conditions']['before'][field]
            if field=='labels':t[0,0]=(int(t[0,0])+1)%6
            else:t[0,0]=~t[0,0]
            with self.assertRaises(AssertionError):audit.audit_cell(trace,data['pools']['train_32'],bad,bad['record'])
        bad=copy.deepcopy(saved);bad['conditions']['before']['gradients']['key.weight'][0,0]+=.01
        with self.assertRaisesRegex(AssertionError,'numerical'):audit.audit_cell(trace,data['pools']['train_32'],bad,bad['record'])
        bad=copy.deepcopy(saved);bad['parts']['stored']['write_logit']+=.01
        with self.assertRaisesRegex(AssertionError,'movement term'):audit.audit_cell(trace,data['pools']['train_32'],bad,bad['record'])
        bad=copy.deepcopy(trace);bad['after_model']['key.weight'][0,0]+=.01
        with self.assertRaisesRegex(AssertionError,'Adam'):audit.scalar_parts(bad)
        bad=copy.deepcopy(saved);bad['record']['probe']['analytic']+=1
        with self.assertRaisesRegex(AssertionError,'projection/sign'):audit.audit_cell(trace,data['pools']['train_32'],bad,bad['record'])

    def test_supplied_ties_absent_errors_and_fixed_membership(self):
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);mask=z.argmax(-1)!=y
        audit.prior.compare(study.fixed_metrics(z,y,mask),audit.fixed(z,y,mask))
        changed=z.clone();changed[0,1,1]=2.;changed[0,0,2]=2.
        result=study.fixed_metrics(changed,y,mask);audit.prior.compare(result,audit.fixed(changed,y,mask))
        self.assertEqual(result['originally_wrong_now_correct'],1);self.assertEqual(result['originally_correct_now_wrong'],1)
        absent=torch.zeros_like(mask);self.assertIsNone(study.fixed_metrics(z,y,absent)['error_ce'])

    def test_projection_terms_and_zero_resolution(self):
        state=study.training.model_source.pair(1747)['delta'].state_dict()
        gradient={n:torch.ones_like(v) for n,v in state.items()};parts={p:{n:torch.zeros_like(v) for n,v in state.items()} for p in ('actual','stored','current','sgd')}
        parts['actual']['write_logit'].fill_(.1);parts['stored']['write_logit'].fill_(.3);parts['current']['write_logit'].fill_(-.2)
        parts['sgd']['write_logit'].fill_(-.1)
        result=study.projections(gradient,parts);audit.prior.compare(result,audit.scalar_projections(gradient,parts))
        self.assertEqual(result['memory']['actual_direction'],'higher_error_ce');self.assertEqual(result['memory']['sgd_direction'],'lower_error_ce')
        self.assertEqual(result['keys']['actual_direction'],'unresolved')
        parts['current']['write_logit'].fill_(.2)
        with self.assertRaisesRegex(AssertionError,'terms'):study.projections(gradient,parts)


if __name__=='__main__':unittest.main()
