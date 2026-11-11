"""Known LP alternatives, free signs, exact certificate and saved evaluator."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import numpy as np
import torch
import association_load
import shared_readout_feasibility as study
import audit_shared_readout_feasibility as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_primal_alternative_and_sparse_rows(self):
        x=np.zeros((2,8));x[:,0]=[-1.,1.];y=np.array([1,0])
        head,certificate,record,dual=study.solve(x,y)
        self.assertEqual(record['outcome'],'verified_feasible');self.assertLess(float(head['weight'][1,0]),0.)
        self.assertGreater(audit.independent_margin(x,head['weight'],head['bias'],y),study.MARGIN)
        audit.row_check(torch.from_numpy(x),torch.from_numpy(y))
        x=np.zeros((2,8));y=np.array([0,1]);head,certificate,record,dual=study.solve(x,y)
        self.assertEqual(record['outcome'],'certified_infeasible');self.assertTrue(audit.exact_certificate(x,y,certificate))
        wrong=copy.deepcopy(certificate);wrong[0]['weight']='-1';self.assertFalse(audit.exact_certificate(x,y,wrong))
        wrong=copy.deepcopy(certificate);wrong[0]['weight']='0';self.assertFalse(audit.exact_certificate(x,y,wrong))
        wrong=copy.deepcopy(certificate);wrong[0]['index']=999;self.assertFalse(audit.exact_certificate(x,y,wrong))
        d=np.zeros(11);d[0]=.5;d[5]=.5
        cert,info=study.reconstruct(x,y,d);self.assertTrue(audit.exact_certificate(x,y,cert))
        self.assertFalse(audit.exact_certificate(x,y,[]))
        with self.assertRaises(AssertionError):study.program(x,np.array([0,6]))

    def test_ties_missing_groups(self):
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);rep=torch.zeros_like(y,dtype=torch.bool)
        metric=study.prior.metrics(z,y,rep);self.assertEqual(metric['all']['accuracy'],.5)
        self.assertEqual(metric['repeat'],dict(count=0,accuracy=None,ce=None));audit.ref.compare(metric,audit.reduced(z,y,rep))

    def test_initialized_saved_evaluator_and_audit(self):
        folder=study.ROOT/'results/test_runs'/('shared_readout_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(67);model=association_load.LoadReader('delta');torch.save(dict(model=model.state_dict()),folder/'initial.pt')
        rows=[dict(events=[[0,2],[1,5],[0,2]],query=0,label=2),dict(events=[[1,1],[0,3],[1,1]],query=1,label=1)]
        study.prior.write(folder/'data.json',dict(pools={'fixture':rows}))
        case=dict(seed=67,checkpoint=(folder/'initial.pt').relative_to(study.ROOT).as_posix(),data=(folder/'data.json').relative_to(study.ROOT).as_posix(),pools=['fixture'])
        out=folder/'study';study.execute(out,[case],fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        original=torch.load(out/'witnesses.pt',weights_only=True);report=json.loads((out/'report.json').read_text())
        self.assertIsNotNone(original[0]['head'])
        def defect(mutator):
            bundles=copy.deepcopy(original);mutator(bundles);torch.save(bundles,out/'witnesses.pt')
            changed=copy.deepcopy(report);changed['witnesses_sha256']=study.prior.digest(out/'witnesses.pt');study.prior.write(out/'report.json',changed)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        defect(lambda x:x[0]['blocks'][0]['reads'].__setitem__((0,0,0),99.))
        defect(lambda x:x[0]['blocks'][0]['labels'].__setitem__((0,0),0))
        defect(lambda x:x[0]['head']['weight'].__setitem__((0,0),99.))
        defect(lambda x:x[0]['blocks'][0]['repeat'].__setitem__((0,0),False))
        torch.save(original,out/'witnesses.pt');report['witnesses_sha256']=study.prior.digest(out/'witnesses.pt')
        report['cells'][0]['metrics']['all']['count']+=1;study.prior.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)


if __name__=='__main__':unittest.main()
