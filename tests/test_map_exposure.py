"""Actual saved continuation/audit and map/stream/evaluator fault controls."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import map_exposure as s
from audit_map_exposure import audit, audit_data


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_data_counterfactuals_ties_and_fit(self):
        original=s.data_source.generate(163,confirmation=True,base_counts=dict(train=2,validation=1,test=1))['pools']
        payload=s.build_data(original,167,fixture=True)
        checked=audit_data(payload)
        self.assertEqual(checked['changed_histories'],6);self.assertEqual(checked['added_map_families'],1)
        wrong=copy.deepcopy(payload);wrong['active']['half_new']['train_32']=wrong['original']['train_32']
        with self.assertRaises(AssertionError):audit_data(wrong)
        wrong=copy.deepcopy(payload);wrong['active']['half_new']['train_32'][0]['events'].reverse()
        with self.assertRaises(AssertionError):audit_data(wrong)
        from audit_full_pool_fitting import reference
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);repeat=torch.zeros_like(y,dtype=torch.bool)
        metrics=reference.metrics(z,y,repeat)
        self.assertEqual(metrics['groups']['all']['accuracy'],.5);self.assertIsNone(metrics['groups']['repeat']['ce'])
        fields={str(m):dict(groups=dict(all=dict(count=100,accuracy=.99),
            unique=dict(count=100 if m==32 else 0,accuracy=.99 if m==32 else None),
            repeat=dict(count=0 if m==32 else 100,accuracy=None if m==32 else .99))) for m in s.prior.LOADS}
        self.assertTrue(s.prior.fit(fields));fields.pop('16')
        with self.assertRaises(AssertionError):s.prior.fit(fields)

    def test_saved_training_audit_and_faults(self):
        folder=ROOT/'results/test_runs'/('map_exposure_'+uuid.uuid4().hex);folder.mkdir(parents=True)
        with torch.random.fork_rng(devices=[]):model=s.data_source.pair(173)['delta']
        opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        checkpoint=folder/'initial.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0),checkpoint)
        original=s.data_source.generate(173,confirmation=True,base_counts=dict(train=2,validation=1,test=1))['pools']
        payload=s.build_data(original,179,fixture=True)
        case=dict(seed=173,checkpoint=checkpoint.relative_to(ROOT).as_posix(),closures=2)
        native=folder/'native';native.mkdir();historical_scores=[]
        historical=s.prior.train_adam(case,payload['active']['reuse'],native,historical_scores,2,4,1)
        case['historical']=historical
        out=folder/'study';rng=torch.get_rng_state().clone()
        s.execute(out,[case],payload,ratio=4,interval=1,fixture=True)
        self.assertTrue(audit(out,persist=False)['passed']);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        report_path=out/'report.json';report_bytes=report_path.read_bytes();report=json.loads(report_bytes)
        self.assertEqual(report['optimizer_calls'],16)
        trace_path=ROOT/report['runs'][1]['trace'];trace_bytes=trace_path.read_bytes()
        for field in ['gradient','after','indices']:
            trace=torch.load(trace_path,weights_only=True)
            if field=='gradient':trace[0]['gradients']['write_logit'].add_(1)
            elif field=='after':trace[0]['after']['value.weight'].add_(1)
            else:trace[0]['indices'][0]=(trace[0]['indices'][0]+1)%12
            torch.save(trace,trace_path);wrong=copy.deepcopy(report)
            wrong['runs'][1]['trace_sha256']=s.replay.digest(trace_path);s.replay.write(report_path,wrong)
            with self.assertRaises(AssertionError,msg=field):audit(out,persist=False)
            trace_path.write_bytes(trace_bytes);report_path.write_bytes(report_bytes)
        score_path=out/'scores.pt';score_bytes=score_path.read_bytes()
        for field in ['labels','logits','missing_cell']:
            scores=torch.load(score_path,weights_only=True)
            if field=='missing_cell':scores.pop()
            else:scores[0][field].flatten()[0]+=1
            torch.save(scores,score_path);wrong=copy.deepcopy(report);wrong['scores_sha256']=s.replay.digest(score_path)
            s.replay.write(report_path,wrong)
            with self.assertRaises(AssertionError,msg=field):audit(out,persist=False)
            score_path.write_bytes(score_bytes);report_path.write_bytes(report_bytes)
        for change in [lambda r:r.__setitem__('answer_targets',r['answer_targets']+1),
                       lambda r:r['cells'][0]['metrics']['groups'].pop('repeat'),
                       lambda r:r['decision'].__setitem__('all_seed_repair',not r['decision']['all_seed_repair'])]:
            wrong=copy.deepcopy(report);change(wrong);s.replay.write(report_path,wrong)
            with self.assertRaises((AssertionError,KeyError)):audit(out,persist=False)
            report_path.write_bytes(report_bytes)


if __name__=='__main__':unittest.main()
