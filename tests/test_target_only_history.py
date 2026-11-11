"""Saved-model evaluator/audit entry and deliberate filtering/count/logit defects."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import target_only_history as study
import audit_target_only_history as audit


def rows():
    return [dict(events=[[3,1],[7,4],[3,1],[9,2]],query=3,label=1,load=3,group='repeat'),
            dict(events=[[2,0],[8,5],[2,0],[10,3]],query=8,label=5,load=3,group='unique')]


class TargetHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_saved_evaluator_and_actual_audit_with_corruptions(self):
        folder=ROOT/'results/test_runs'/('target_only_'+uuid.uuid4().hex);folder.mkdir()
        path=folder/'initial.pt';model=study.training.model_source.pair(1701)['delta']
        torch.save(dict(model=model.state_dict()),path);model=study.training.previous.load_model(path)
        data=rows(); original=study.training.evaluate(model,data);saved=study.evaluate(model,data,batch=1)
        record=dict(original=original['metrics'],filtered=saved['metrics'],
            transitions=study.transitions(original['logits'],saved['logits'],saved['labels'],saved['repeat']),
            retained_events=int(saved['target_writes'].sum()),target_writes=int(saved['target_writes'][0,0]),
            beta=float(model.write_logit.sigmoid().detach()))
        audit.audit_cell(path,data,original,saved,record)
        for field in ('target_writes','labels','logits'):
            bad=copy.deepcopy(saved);bad[field].flatten()[0]+=1
            with self.assertRaises(AssertionError):audit.audit_cell(path,data,original,bad,record)
        selected,qs,_,_,_=study.filtered_queries(data)
        self.assertEqual(selected[0],[[3,1],[3,1]]);self.assertEqual(qs[0].tolist(),[3,7,9])
        # Deliberately retain the competing write; independent clean formula rejects it.
        bad=copy.deepcopy(saved)
        with torch.no_grad():bad['logits'][0,0]=model(torch.tensor([data[0]['events']]),torch.tensor([3]))[0]
        with self.assertRaises(AssertionError):audit.audit_cell(path,data,original,bad,record)

    def test_closed_form_nonunit_clamped_keys_and_wrong_count(self):
        model=study.training.model_source.pair(1703)['delta']
        for magnitude in (1.,1e-14,0.):
            with torch.no_grad():model.key.weight[3].fill_(magnitude)
            for n in (1,2,16):
                with torch.no_grad():z=model(torch.tensor([[[3,1]]*n]),torch.tensor([3]))[0]
                expected=torch.tensor(audit.closed_logits(model,3,1,n),dtype=torch.float64)
                torch.testing.assert_close(z,expected,atol=1e-12,rtol=1e-12)
        with torch.no_grad():model.key.weight[3].fill_(1.)
        self.assertGreater(max(abs(a-b) for a,b in zip(audit.closed_logits(model,3,1,1),audit.closed_logits(model,3,1,2))),1e-5)

    def test_supplied_logits_ties_missing_groups_and_transitions(self):
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);repeat=torch.zeros(1,2,dtype=torch.bool)
        result=study.training.reduce_scores(z,y,repeat)
        audit.reference.check_metrics(result,audit.reference.metrics(z,y,repeat))
        self.assertEqual(result['groups']['all']['accuracy'],.5)
        self.assertIsNone(result['groups']['repeat']['accuracy'])
        altered=z.clone();altered[0,0,2]=2.;altered[0,1,1]=2.
        t=study.transitions(z,altered,y,repeat)['all']
        self.assertEqual(t,dict(count=2,correct_both=0,wrong_both=0,rescued=1,newly_wrong=1))


if __name__=='__main__':unittest.main()
