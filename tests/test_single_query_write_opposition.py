"""Single-loss autograd, coefficient controls and actual saved audit defects."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import single_query_write_opposition as study
import audit_single_query_write_opposition as audit
import audit_write_credit_assignment as primary_audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_single_loss_autograd_and_zero_controls(self):
        torch.manual_seed(59);model=association_load.LoadReader('delta')
        rows=[dict(events=[[0,2],[1,5],[0,2]],query=0,label=2),dict(events=[[1,1],[0,3],[1,1]],query=1,label=1)]
        old=study.primary.measure(model,rows,59);f,r,s,x=study.analyze(model,rows,old)
        a,d,g,*_=audit.independent(model,rows,old)
        torch.testing.assert_close(f['coefficients'],a);torch.testing.assert_close(f['coefficients'][...,None]*f['read_gradient'][:,:,None],g)
        e=torch.tensor([r['events'] for r in rows]);q,y,_=study.primary.prior.labels(rows);k=model.keys(e[...,0]);v=model.value(e[...,1]);state=torch.zeros(2,16,8,dtype=torch.float64)
        beta=model.write_logit.sigmoid()
        for t in range(3):state=state+beta*k[:,t,:,None]*(v[:,t]-(k[:,t,:,None]*state).sum(1))[:,None,:]
        z=model.output(model.keys(q)@state)
        loss=torch.nn.functional.cross_entropy(z[0,0,None],y[0,0,None])/(4*y.numel())
        grad,=torch.autograd.grad(loss,v);torch.testing.assert_close(grad[0],g[0,0],atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.equal(grad[1],torch.zeros_like(grad[1])))
        coeff=torch.tensor([[[1.,-1.]]],dtype=torch.float64);event=coeff[...,None]
        masks=dict(correct=torch.ones(1,1,dtype=torch.bool),error=torch.zeros(1,1,dtype=torch.bool),total=torch.ones(1,1,dtype=torch.bool))
        r,s=study.query_statistics(event,coeff,torch.tensor([[0,0]]),masks,torch.tensor([[0]]),torch.tensor([[0]]),0)
        self.assertEqual(r[0]['within'],-2.);self.assertTrue(r[0]['opposite_same_value']);self.assertEqual(s[1]['queries'],0)
        r,s=study.query_statistics(event,coeff,torch.tensor([[0,1]]),masks,torch.tensor([[0]]),torch.tensor([[0]]),0)
        self.assertEqual(r[0]['within'],0.);self.assertFalse(r[0]['opposite_same_value'])
        with torch.no_grad():model.output.weight.zero_();model.output.bias.zero_()
        rows=[dict(events=[[0,0],[0,0]],query=0,label=0)]
        old=study.primary.measure(model,rows,0);f,r,s,x=study.analyze(model,rows,old)
        self.assertEqual(s[1]['queries'],0);self.assertEqual(s[0]['weighted_unresolved'],1)
        audit.independent(model,rows,old)

    def test_initialized_saved_entry_and_defects(self):
        folder=study.ROOT/'results/test_runs'/('single_query_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(61);model=association_load.LoadReader('delta')
        rows=[dict(events=[[0,2],[1,5],[0,2]],query=0,label=2),dict(events=[[1,1],[0,3],[1,1]],query=1,label=1)]
        torch.save(dict(model=model.state_dict()),folder/'initial.pt');study.primary.prior.write(folder/'data.json',dict(pools={'fixture':rows}))
        case=dict(seed=61,checkpoint=(folder/'initial.pt').relative_to(study.ROOT).as_posix(),data=(folder/'data.json').relative_to(study.ROOT).as_posix(),pool='fixture')
        source=folder/'primary';study.primary.execute(source,[case],fixture=True);primary_audit.audit(source)
        out=folder/'study';study.execute(source,out,fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        original=torch.load(out/'factors.pt',weights_only=True);report=json.loads((out/'report.json').read_text())
        def defect(mutator):
            factors=copy.deepcopy(original);mutator(factors);torch.save(factors,out/'factors.pt')
            changed=copy.deepcopy(report);changed['factors_sha256']=study.primary.prior.digest(out/'factors.pt');study.primary.prior.write(out/'report.json',changed)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        defect(lambda x:x[0]['coefficients'].__setitem__((0,0,0),99.))
        defect(lambda x:x[0]['labels'].__setitem__((0,0),0))
        defect(lambda x:x[0]['masks']['correct'].__setitem__((0,0),not bool(x[0]['masks']['correct'][0,0])))
        torch.save(original,out/'factors.pt');changed=copy.deepcopy(report);changed['factors_sha256']=study.primary.prior.digest(out/'factors.pt')
        changed['summary'][0]['queries']+=1;study.primary.prior.write(out/'report.json',changed)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        changed['summary']=copy.deepcopy(report['summary']);study.primary.prior.write(out/'report.json',changed)
        text=(out/'queries.csv').read_text();(out/'queries.csv').write_text(text.replace('negative','positive',1) if 'negative' in text else text.replace('unresolved','positive',1))
        changed['queries_sha256']=study.primary.prior.digest(out/'queries.csv');study.primary.prior.write(out/'report.json',changed)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)


if __name__=='__main__':unittest.main()
