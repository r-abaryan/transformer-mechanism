"""Saved synthetic traces and deliberate faults for transport review."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import review_fitting_transport as review


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self, folder):
        shapes={'write_logit':(), 'key.weight':(1,2), 'value.weight':(1,2), 'output.weight':(1,2), 'output.bias':(1,)}
        def unpack(vector):
            result={};offset=0
            for name,shape in shapes.items():
                size=1
                for x in shape:size*=x
                result[name]=vector[offset:offset+size].clone().reshape(shape);offset+=size
            return result
        g=torch.arange(1,9,dtype=torch.float64)/10
        x=torch.arange(8,dtype=torch.float64)/100
        curvature=torch.eye(8,dtype=torch.float64)*2
        curvature[0,4]=curvature[4,0]=.5
        t=.1;d=-g;s=t*d;y=curvature@s;g1=g+y
        rho=1/y.dot(s);gamma=y.dot(s)/y.dot(y)
        v=torch.eye(8,dtype=torch.float64)-rho*torch.outer(s,y)
        h=v@(torch.eye(8,dtype=torch.float64)*gamma)@v.T+rho*torch.outer(s,s)
        d1=-(h@g1)
        states=[dict(d=d,t=t,H_diag=1.,old_dirs=[],old_stps=[],ro=[]),
                dict(d=d1,t=t,H_diag=gamma,old_dirs=[y],old_stps=[s],ro=[rho])]
        points=[x,x+s,x+s+t*d1]
        trace=dict(calls=[dict(gradients=unpack(z)) for z in [g,g1]],
                   steps=[dict(first=i,last=i+1,before=unpack(points[i]),after=unpack(points[i+1]),
                               optimizer=dict(state={0:states[i]})) for i in range(2)])
        path=folder/'trace.pt';torch.save(trace,path)
        run=dict(seed=17,arm='lbfgs',trace=path.relative_to(ROOT).as_posix(),closure_evaluations=2,optimizer_calls=2)
        return run,path

    def test_saved_producer_audit_and_faults(self):
        folder=ROOT/'results/test_runs'/('fitting_transport_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        run,path=self.fixture(folder);out=folder/'review'
        review.execute(out,[run],{run['trace']:review.replay.digest(path)},fixture=True)
        result=review.audit(out,persist=False);self.assertTrue(result['passed']);self.assertEqual(result['dense_directions'],2)
        original=(out/'report.json').read_bytes();report=json.loads(original)
        self.assertEqual(report['records'][0]['cross_norm_ratio'],0)
        self.assertGreater(report['records'][1]['cross_norm_ratio'],0)
        for change in [lambda r:r['records'].pop(), lambda r:r['records'][1].__setitem__('cross_norm_ratio',99),
                       lambda r:r['inventory'][0].__setitem__('routing_parameters',8)]:
            bad=copy.deepcopy(report);change(bad);review.replay.write(out/'report.json',bad)
            with self.assertRaises((AssertionError,StopIteration)):review.audit(out,persist=False)
            (out/'report.json').write_bytes(original)
        raw=path.read_bytes();bad=torch.load(path,weights_only=True)
        bad['steps'][1]['optimizer']['state'][0]['d'][0]+=1;torch.save(bad,path)
        lockpath=out/'lock.json';lockraw=lockpath.read_bytes();lock=json.loads(lockraw)
        lock['files'][run['trace']]=review.replay.digest(path);review.replay.write(lockpath,lock)
        bound=copy.deepcopy(report);bound['lock_sha256']=review.replay.digest(lockpath);review.replay.write(out/'report.json',bound)
        with self.assertRaises(AssertionError):review.audit(out,persist=False)
        path.write_bytes(raw);lockpath.write_bytes(lockraw);(out/'report.json').write_bytes(original)
        self.assertTrue(review.audit(out,persist=False)['passed'])

    def test_partition_and_dense_null(self):
        values=dict(write_logit=torch.zeros(()),**{'key.weight':torch.zeros(1,2),'value.weight':torch.zeros(1,2),
                                                   'output.weight':torch.zeros(1,2),'output.bias':torch.zeros(1)})
        _,mask=review.partition(values);g=torch.arange(1,9,dtype=torch.float64)
        state=dict(H_diag=2.,old_dirs=[],old_stps=[],ro=[])
        within,cross=review.components(g,state,mask)
        torch.testing.assert_close(within,-2*g);self.assertEqual(float(cross.norm()),0)
        with self.assertRaises(AssertionError):review.components(g,state,torch.ones(8,dtype=torch.bool))
        values['unexpected']=torch.zeros(1)
        with self.assertRaises(AssertionError):review.partition(values)


if __name__=='__main__':unittest.main()
