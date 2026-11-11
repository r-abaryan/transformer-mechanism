"""Saved evaluator/audit, known interference and deliberate evidence defects."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import write_interference_stages as study
import audit_write_interference_stages as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_saved_runtime_and_faults(self):
        folder=study.ROOT/'results/test_runs'/('write_stages_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(79);model=association_load.LoadReader('delta')
        torch.save(dict(model=model.state_dict(),step=0),folder/'initial.pt')
        rows=[dict(events=[[i%8,i%6],[(i+1)%8,(i+2)%6]],query=i%8,label=i%6) for i in range(24)]
        study.prior.write(folder/'data.json',dict(pools={'train_32':rows}))
        case=dict(seed=79,step=0,checkpoint=(folder/'initial.pt').relative_to(study.ROOT).as_posix(),data=(folder/'data.json').relative_to(study.ROOT).as_posix())
        out=folder/'study';study.execute(out,[case],fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        saved=torch.load(out/'traces.pt',weights_only=True);report=json.loads((out/'report.json').read_text())
        for corrupt in (lambda b:b[0]['labels'].__setitem__((0,0),5),
                        lambda b:b[0]['logits'].__setitem__((1,0,0,0),999.),
                        lambda b:b[0]['coefficients'].__setitem__((0,0,0),99.),
                        lambda b:b[0]['summary'][0].__setitem__('count',0),
                        lambda b:b[0]['fields']['position'].__setitem__((0,0),2),
                        lambda b:b[0]['endpoint_correct'].__setitem__((0,0),not bool(b[0]['endpoint_correct'][0,0]))):
            wrong=copy.deepcopy(saved);corrupt(wrong);torch.save(wrong,out/'traces.pt');report['traces_sha256']=study.prior.digest(out/'traces.pt');study.prior.write(out/'report.json',report)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        torch.save(saved,out/'traces.pt');report['traces_sha256']=study.prior.digest(out/'traces.pt');report['query_rows']+=1;study.prior.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)

    def test_known_retention_and_missing_cohorts(self):
        model=association_load.LoadReader('delta')
        with torch.no_grad():
            model.key.weight.zero_();model.key.weight[:,0]=1
            model.value.weight.zero_();model.value.weight[0,0]=2;model.value.weight[1,0]=-2
            model.output.weight.zero_();model.output.bias.zero_();model.output.weight[0,0]=1;model.output.weight[1,0]=-1
            model.write_logit.fill_(0)
        rows=[dict(events=[[0,0],[1,1]],query=0,label=0)]
        endpoint=torch.tensor([[False,True]])
        b=study.analyze(model,rows,endpoint,0,0)
        self.assertAlmostEqual(float(b['fields']['target_coefficient_ratio'][0,0]),.5)
        self.assertTrue(bool(b['fields']['post_correct'][0,0]));self.assertFalse(bool(b['fields']['final_correct'][0,0]))
        self.assertTrue(bool(b['fields']['later_harm'][0,0]))
        with torch.no_grad():model.key.weight[1].zero_();model.key.weight[1,1]=1
        orthogonal=study.analyze(model,rows,torch.ones_like(endpoint),0,0)
        self.assertAlmostEqual(float(orthogonal['fields']['target_coefficient_ratio'][0,0]),1.)
        self.assertTrue(bool(orthogonal['fields']['final_correct'][0,0]))
        empty=next(r for r in orthogonal['summary'] if r['group']=='endpoint_wrong' and r['position'] is None)
        self.assertEqual(empty['count'],0);self.assertIsNone(empty['final_margin_mean'])
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]])
        metric=audit.proof.reduced(z,y,torch.zeros_like(y,dtype=torch.bool))
        self.assertEqual(metric['all']['accuracy'],.5);self.assertIsNone(metric['repeat']['ce'])


if __name__=='__main__':unittest.main()
