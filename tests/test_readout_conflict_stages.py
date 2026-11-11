"""Saved stage runtime, missing groups, singular reads and deliberate defects."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import numpy as np
import association_load
import readout_conflict_stages as study
import audit_readout_conflict_stages as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self,zero=False):
        folder=study.ROOT/'results/test_runs'/('conflict_stages_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(67);model=association_load.LoadReader('delta')
        if zero:
            with torch.no_grad():model.value.weight.zero_()
        torch.save(dict(model=model.state_dict(),step=0),folder/'initial.pt')
        rows=[]
        for i in range(48):
            k=i%16;j=(k+1)%16;v=i%6;other=(v+1+i//16)%6
            rows.append(dict(events=[[k,v],[j,other],[k,v]],query=k,label=v))
        study.original.prior.write(folder/'data.json',dict(pools={'fixture':rows}))
        case=dict(seed=67,step=0,checkpoint=(folder/'initial.pt').relative_to(study.ROOT).as_posix(),data=(folder/'data.json').relative_to(study.ROOT).as_posix(),pools=['fixture'],cached=False)
        out=folder/'study';study.execute(out,[case],fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed']);return out

    def test_initialized_stage_and_defects(self):
        out=self.fixture();saved=torch.load(out/'stages.pt',weights_only=True);report=json.loads((out/'report.json').read_text())
        self.assertEqual(saved[0]['recovery']['record']['outcome'],'verified_feasible')
        def defect(change):
            wrong=copy.deepcopy(saved);change(wrong);torch.save(wrong,out/'stages.pt')
            amended=copy.deepcopy(report);amended['stages_sha256']=study.original.prior.digest(out/'stages.pt');study.original.prior.write(out/'report.json',amended)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        defect(lambda b:b[0]['source']['blocks'][0]['reads'].__setitem__((0,0,0),99.))
        defect(lambda b:b[0]['source']['blocks'][0]['labels'].__setitem__((0,0),5))
        defect(lambda b:b[0]['recovery']['coordinate']['transformed'].__setitem__((0,0),99.))
        defect(lambda b:b[0]['recovery']['candidate']['weight'].__setitem__((0,0),99.))
        defect(lambda b:b[0]['case'].__setitem__('step',1))
        torch.save(saved,out/'stages.pt');report['stages_sha256']=study.original.prior.digest(out/'stages.pt');report['fresh_native_queries']+=1;study.original.prior.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        # Restore valid evidence before corrupting only the CSV.
        report['fresh_native_queries']-=1;study.original.prior.write(out/'report.json',report)
        (out/'metrics.csv').write_text('bad\n',encoding='utf-8');report['metrics_sha256']=study.original.prior.digest(out/'metrics.csv');study.original.prior.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)

    def test_singular_missing_groups_and_certificate_control(self):
        out=self.fixture(zero=True);report=json.loads((out/'report.json').read_text());self.assertEqual(report['new_solver_calls'],0)
        self.assertEqual(report['summary'][0]['outcome'],'unresolved')
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);rep=torch.zeros_like(y,dtype=torch.bool)
        m=audit.proof.reduced(z,y,rep);self.assertEqual(m['all']['accuracy'],.5);self.assertIsNone(m['repeat']['accuracy'])
        x=np.zeros((2,8));labels=np.array([0,1]);cert=[dict(index=0,weight='1/2'),dict(index=5,weight='1/2')]
        self.assertTrue(audit.proof.exact_certificate(x,labels,cert));cert[0]['weight']='-1/2';self.assertFalse(audit.proof.exact_certificate(x,labels,cert))


if __name__=='__main__':unittest.main()
