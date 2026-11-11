"""Saved evaluator/audit, exact certificates and inverse error controls."""
import copy
import json
import sys
import uuid
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import shared_state_feasibility as study
import audit_shared_state_feasibility as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_feasible_conflict_and_wrong_bounds(self):
        q=[[1.,0.],[0.,1.]]; w=[[1.,0.],[0.,1.]]; b=[0.,0.]
        s,dual,record=study.solve(q,w,b,[0,1])
        z,_=study.decimal_logits(q,s.tolist(),w,b)
        self.assertGreater(min(float(z[0][0]-z[0][1]),float(z[1][1]-z[1][0])),study.MARGIN)
        q=[[1.,0.],[1.,0.]]
        s,dual,record=study.solve(q,w,b,[0,1])
        self.assertTrue(study.dual_certificate(q,w,b,[0,1],dual)[0])
        self.assertTrue(audit.exact_upper(q,w,b,[0,1],dual)[0])
        altered=dual.copy(); altered[0]+=.125
        self.assertFalse(audit.exact_upper(q,w,b,[0,1],altered)[0])
        # A negative state entry is necessary for this otherwise feasible case.
        s,_,_=study.solve([[1.,0.]],[[0.,0.],[1.,0.]],[0.,0.],[0])
        self.assertLess(float(s[0,0]),0)

    def test_supplied_ties_missing_groups_and_criteria(self):
        z=torch.zeros(1,2,6,dtype=torch.float64); y=torch.tensor([[0,1]]); repeat=torch.zeros_like(y,dtype=torch.bool)
        metrics=study.metrics(z,y,repeat)
        self.assertEqual(metrics['all']['accuracy'],.5)
        self.assertEqual(metrics['repeat'],dict(count=0,accuracy=None,ce=None))
        expected=audit.existing.metrics(z,y,repeat)
        audit.compare(metrics,{**expected['groups'],'complete_maps':0})
        bad=copy.deepcopy(metrics); bad['repeat']['count']=1
        with self.assertRaises(AssertionError): audit.compare(bad,metrics)
        with self.assertRaises(AssertionError): audit.compare([], [dict(seed=1)])

    def test_saved_evaluator_and_actual_audit(self):
        root=study.ROOT/'results/test_runs'/('shared_state_'+uuid.uuid4().hex)
        root.mkdir()
        with self.subTest('saved fixture'):
            checkpoint=root/'initialized.pt'; data=root/'data.json'; out=root/'study'
            torch.manual_seed(29); model=association_load.LoadReader('delta')
            torch.save(dict(model=model.state_dict()),checkpoint)
            rows=[dict(events=[[0,2],[1,5]],query=0,label=2),dict(events=[[1,3],[0,1]],query=1,label=3)]
            study.write(data,dict(pools={'fixture':rows}))
            case=dict(seed=29,checkpoint=checkpoint.relative_to(study.ROOT).as_posix(),data=data.relative_to(study.ROOT).as_posix(),pool='fixture')
            study.execute(out,[case],fixture=True)
            self.assertTrue(audit.audit(out,persist=False)['passed'])
            original=torch.load(out/'witnesses.pt',weights_only=True)
            def changed(mutator):
                payload=copy.deepcopy(original); mutator(payload)
                torch.save(payload,out/'witnesses.pt')
                report=json.loads((out/'report.json').read_text()); report['witnesses_sha256']=study.digest(out/'witnesses.pt')
                study.write(out/'report.json',report)
                with self.assertRaises(AssertionError): audit.audit(out,persist=False)
            changed(lambda p:p[0]['labels'].__setitem__((0,0),0))
            changed(lambda p:p[0]['oracle_logits'].__setitem__((0,0,0),900.))
            changed(lambda p:p[0]['records'][0].__setitem__('outcome','certified_nonpositive'))
            changed(lambda p:p[0]['repeat'].__setitem__((0,0),True))
            changed(lambda p:p[0]['states'].__setitem__(0,torch.zeros(16,8,dtype=torch.float64)))
            torch.save(original,out/'witnesses.pt')
            report=json.loads((out/'report.json').read_text()); report['witnesses_sha256']=study.digest(out/'witnesses.pt')
            report['summary'][0]['positive']+=1; study.write(out/'report.json',report)
            with self.assertRaises(AssertionError): audit.audit(out,persist=False)


if __name__=='__main__':
    torch.set_num_threads(1); unittest.main()
