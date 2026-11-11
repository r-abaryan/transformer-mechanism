"""Coordinate arithmetic, exact alternatives and the saved recovery entry."""
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
import shared_readout_recovery as study
import audit_shared_readout_recovery as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_coordinates_and_certificates(self):
        rng=np.random.default_rng(91);x=rng.normal(size=(24,8));coordinate=study.transform(x)
        self.assertTrue(audit.nonsingular(coordinate['matrix']))
        np.testing.assert_allclose(coordinate['transformed'],audit.independent_transform(x,coordinate),atol=1e-12,rtol=1e-12)
        wrong=coordinate['matrix'].clone();wrong[:,1]=wrong[:,0];self.assertFalse(audit.nonsingular(wrong))
        theta=rng.normal(size=46);head=study.map_head(theta,coordinate)
        u=np.vstack((np.zeros(9),theta[:-1].reshape(5,9)))
        np.testing.assert_allclose(x@head['weight'].numpy().T+head['bias'].numpy(),coordinate['transformed'].numpy()@u[:,:-1].T+u[:,-1],atol=1e-12,rtol=1e-12)
        y=(x[:,0]>0).astype(int);good,cert,record,dual=study.original.solve(coordinate['transformed'].numpy(),y)
        self.assertEqual(record['outcome'],'verified_feasible')
        # Two exactly equal original reads require different targets.
        x=np.vstack((x,x[0]));y=np.r_[y,1-y[0]];coordinate=study.transform(x)
        head,cert,record,dual=study.original.solve(coordinate['transformed'].numpy(),y)
        cert,info=study.original.reconstruct(x,y,dual);self.assertTrue(audit.proof.exact_certificate(x,y,cert))
        bad=copy.deepcopy(cert);bad[0]['weight']='-1';self.assertFalse(audit.proof.exact_certificate(x,y,bad))
        with self.assertRaises(AssertionError):study.transform(np.zeros((24,8)))

    def test_cancellation_and_absent_groups(self):
        x=np.tile([1.,1.,0.,0.,0.,0.,0.,0.],(2,1));matrix=torch.eye(8,dtype=torch.float64)
        matrix[0,0]=1e16;matrix[1,0]=-1e16;matrix[2,0]=1.
        x[0,2]=1.;coordinate=dict(matrix=matrix,mean=torch.zeros(8,dtype=torch.float64))
        self.assertEqual(audit.independent_transform(x,coordinate)[0,0],1.)
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);rep=torch.zeros_like(y,dtype=torch.bool)
        m=audit.proof.reduced(z,y,rep);self.assertEqual(m['all']['accuracy'],.5);self.assertIsNone(m['repeat']['ce'])

    def test_initialized_saved_recovery_and_audit(self):
        folder=study.ROOT/'results/test_runs'/('readout_recovery_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(67);model=association_load.LoadReader('delta');torch.save(dict(model=model.state_dict()),folder/'initial.pt')
        rows=[]
        for i in range(48):
            k=i%16;j=(k+1)%16;v=i%6;other=(v+1+i//16)%6
            rows.append(dict(events=[[k,v],[j,other],[k,v]],query=k,label=v))
        study.original.prior.write(folder/'data.json',dict(pools={'fixture':rows}))
        case=dict(seed=67,checkpoint=(folder/'initial.pt').relative_to(study.ROOT).as_posix(),data=(folder/'data.json').relative_to(study.ROOT).as_posix(),pools=['fixture'])
        origin=folder/'original';study.original.execute(origin,[case],fixture=True);audit.proof.audit(origin)
        out=folder/'recovery';study.execute(out,origin,fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        saved=torch.load(out/'witnesses.pt',weights_only=True);report=json.loads((out/'report.json').read_text())
        self.assertIsNotNone(saved[0]['coordinate'])
        def defect(change):
            wrong=copy.deepcopy(saved);change(wrong);torch.save(wrong,out/'witnesses.pt')
            amended=copy.deepcopy(report);amended['witnesses_sha256']=study.original.prior.digest(out/'witnesses.pt');study.original.prior.write(out/'report.json',amended)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        defect(lambda b:b[0]['coordinate']['transformed'].__setitem__((0,0),99.))
        defect(lambda b:b[0]['coordinate']['matrix'].__setitem__((0,0),99.))
        if saved[0]['candidate'] is not None: defect(lambda b:b[0]['candidate']['weight'].__setitem__((0,0),99.))
        if saved[0]['certificate']: defect(lambda b:b[0]['certificate'][0].__setitem__('weight','-1'))
        torch.save(saved,out/'witnesses.pt');report['witnesses_sha256']=study.original.prior.digest(out/'witnesses.pt');report['original_queries']+=1
        study.original.prior.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)


if __name__=='__main__':unittest.main()
