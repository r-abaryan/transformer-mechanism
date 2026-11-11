"""Actual saved evaluator/audit, finite differences and deliberate defects."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import write_credit_assignment as study
import audit_write_credit_assignment as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_initialized_saved_entry_and_deliberate_errors(self):
        folder=study.ROOT/'results/test_runs'/('write_credit_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(41);model=association_load.LoadReader('delta');checkpoint=folder/'initial.pt'
        torch.save(dict(model=model.state_dict()),checkpoint)
        rows=[dict(events=[[0,2],[1,5],[0,2]],query=0,label=2),dict(events=[[1,1],[0,3],[1,1]],query=1,label=1)]
        data=folder/'data.json';study.prior.write(data,dict(pools={'fixture':rows}))
        case=dict(seed=41,checkpoint=checkpoint.relative_to(study.ROOT).as_posix(),data=data.relative_to(study.ROOT).as_posix(),pool='fixture')
        out=folder/'study';study.execute(out,[case],fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        original=torch.load(out/'gradients.pt',weights_only=True);report=json.loads((out/'report.json').read_text())
        def defect(change):
            value=copy.deepcopy(original);change(value);torch.save(value,out/'gradients.pt')
            modified=copy.deepcopy(report);modified['gradients_sha256']=study.prior.digest(out/'gradients.pt');study.prior.write(out/'report.json',modified)
            with self.assertRaises((AssertionError,KeyError)):audit.audit(out,persist=False)
        defect(lambda x:x[0]['labels'].__setitem__((0,0),0))
        defect(lambda x:x[0]['logits'].__setitem__((0,0,0),99.))
        defect(lambda x:x[0]['gradients']['error']['states'].__setitem__((0,0,0,0),99.))
        defect(lambda x:x[0]['gradients']['total']['events'].__setitem__((0,0,0),99.))
        defect(lambda x:x[0]['masks']['error'].__setitem__((0,0),not bool(x[0]['masks']['error'][0,0])))
        defect(lambda x:x[0]['gradients'].pop('correct'))
        torch.save(original,out/'gradients.pt');report['gradients_sha256']=study.prior.digest(out/'gradients.pt')
        report['summary'][0]['queries']+=1;study.prior.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)

    def test_ties_absent_groups_and_transport_controls(self):
        model=association_load.LoadReader('delta')
        with torch.no_grad():model.output.weight.zero_();model.output.bias.zero_()
        rows=[dict(events=[[0,0],[0,0]],query=0,label=0)]
        result=study.measure(model,rows,0);self.assertEqual(result['metrics']['all']['accuracy'],1.)
        self.assertEqual(result['metrics']['repeat']['count'],1);self.assertEqual(result['metrics']['unique']['count'],0)
        records,summary=study.records(result)
        error=next(r for r in summary if r['group']=='error');self.assertIsNone(error['temporal_sum_ratio'])
        self.assertEqual(error['queries'],0)
        folder=study.ROOT/'results/test_runs'/('write_credit_ties_'+uuid.uuid4().hex);folder.mkdir()
        torch.save(dict(model=model.state_dict()),folder/'initial.pt');study.prior.write(folder/'data.json',dict(pools={'ties':rows}))
        case=dict(seed=0,checkpoint=(folder/'initial.pt').relative_to(study.ROOT).as_posix(),
                  data=(folder/'data.json').relative_to(study.ROOT).as_posix(),pool='ties')
        study.execute(folder/'study',[case],fixture=True);self.assertTrue(audit.audit(folder/'study',persist=False)['passed'])
        expected=audit.analytic(model,rows)
        for group in study.GROUPS:
            for field in expected['gradients'][group]:torch.testing.assert_close(result['gradients'][group][field],expected['gradients'][group][field])
        # Known scalar transport along the written key, orthogonal direction unchanged.
        k=torch.zeros(16,dtype=torch.float64);k[0]=1.;g=torch.zeros(16,8,dtype=torch.float64);g[0,0]=1.;g[1,0]=2.
        a=torch.eye(16,dtype=torch.float64)-.25*k[:,None]*k[None,:]
        for _ in range(4):g=a.T@g
        self.assertAlmostEqual(float(g[0,0]),.75**4);self.assertEqual(float(g[1,0]),2.)
        self.assertTrue(torch.equal((torch.eye(16,dtype=torch.float64)-0*k[:,None]*k[None,:]).T@g,g))

    def test_event_finite_difference(self):
        torch.manual_seed(43);model=association_load.LoadReader('delta')
        rows=[dict(events=[[0,2],[1,5],[0,2]],query=0,label=2)];result=study.measure(model,rows,0)
        e=torch.tensor([r['events'] for r in rows]);q,y,_=study.prior.labels(rows)
        with torch.no_grad():
            k=model.keys(e[...,0]);v=model.value(e[...,1]);beta=model.write_logit.sigmoid()
            def loss(values):
                s=torch.zeros(1,16,8,dtype=torch.float64)
                for t in range(3):s=s+beta*k[:,t,:,None]*(values[:,t]-(k[:,t,:,None]*s).sum(1))[:,None,:]
                z=model.output(model.keys(q)@s);return float(torch.nn.functional.cross_entropy(z.flatten(0,1),y.flatten())/4)
            plus=v.clone();minus=v.clone();plus[0,0,0]+=1e-5;minus[0,0,0]-=1e-5
            numerical=(loss(plus)-loss(minus))/(2e-5)
        self.assertLess(abs(numerical-float(result['gradients']['total']['events'][0,0,0])),1e-7)


if __name__=='__main__':unittest.main()
