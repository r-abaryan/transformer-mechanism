"""Initialized saved audit, local gradients and deliberate preservation errors."""
import copy
import json
import math
import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import shared_state_preservation as study
import audit_shared_state_preservation as audit
import audit_shared_state_feasibility as source_audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_known_damage_controls_and_local_gradient(self):
        model=association_load.LoadReader('delta')
        with torch.no_grad():
            model.key.weight.zero_();model.key.weight[0,0]=1.;model.key.weight[1,0]=.9;model.key.weight[1,1]=math.sqrt(.19)
            model.value.weight.zero_();model.value.weight[0,0]=2.;model.value.weight[1,0]=-2.
            model.output.weight.zero_();model.output.weight[0,0]=1.;model.output.weight[1,0]=-1.
            model.output.bias.fill_(-100.);model.output.bias[:2].zero_();model.write_logit.zero_()
        rows=[dict(events=[[0,0],[1,1]],query=0,label=0)]
        q,y,rs=study.prior.labels(rows);s=torch.zeros(1,16,8,dtype=torch.float64)
        s[0,0,0]=.1;s[0,1,0]=(-.1-.9*.1)/math.sqrt(.19)
        result=study.replay(model,rows,q,y,rs,s,0,'oracle')
        self.assertEqual(result['records'][0]['metrics']['complete_maps'],1)
        self.assertEqual(result['records'][1]['transitions']['other_harms'],1)
        self.assertEqual(result['records'][1]['transitions']['written_harms'],0)
        frozen=study.replay(model,rows,q,y,rs,s,0,'oracle',beta_override=0.)
        self.assertTrue(torch.equal(frozen['logits'][0],frozen['logits'][-1]))
        key=model.keys(torch.tensor([0])).detach();value=model.value(torch.tensor([0])).detach()
        variable=s.clone().requires_grad_(True);loss=.5*(value-(key[:,:,None]*variable).sum(1)).square().sum()
        grad,=torch.autograd.grad(loss,variable)
        expected=key[:,:,None]*((key[:,:,None]*s).sum(1)-value)[:,None,:]
        torch.testing.assert_close(grad,expected)
        with torch.no_grad():model.key.weight[1].zero_();model.key.weight[1,1]=1.
        orthogonal=study.replay(model,rows,q,y,rs,s,0,'oracle')
        self.assertEqual(orthogonal['records'][1]['transitions']['other_harms'],0)
        e=torch.tensor([rows[0]['events']]);zero=study.replay(model,rows,q,y,rs,torch.zeros_like(s),0,'native')
        torch.testing.assert_close(zero['final_state'],model.prefill(e)[0])

    def test_supplied_ties_and_missing_groups(self):
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);rs=torch.zeros_like(y,dtype=torch.bool)
        record=study.frame(z,y,rs,z.argmax(-1)==y,1,'native',0)
        self.assertEqual(record['metrics']['all']['accuracy'],.5)
        self.assertEqual(record['metrics']['repeat'],dict(count=0,accuracy=None,ce=None))
        audit.reference.compare(record['metrics'],audit.reduced(z,y,rs))

    def test_initialized_saved_evaluator_and_audit(self):
        folder=study.ROOT/'results/test_runs'/('shared_preservation_'+uuid.uuid4().hex);folder.mkdir()
        checkpoint=folder/'initial.pt';data=folder/'data.json';source=folder/'source';out=folder/'study'
        torch.manual_seed(31);model=association_load.LoadReader('delta');torch.save(dict(model=model.state_dict()),checkpoint)
        rows=[dict(events=[[0,2],[1,5],[0,2]],query=0,label=2),dict(events=[[1,1],[0,3],[1,1]],query=1,label=1)]
        study.prior.write(data,dict(pools={'fixture':rows}))
        case=dict(seed=31,checkpoint=checkpoint.relative_to(study.ROOT).as_posix(),data=data.relative_to(study.ROOT).as_posix(),pool='fixture')
        study.prior.execute(source,[case],fixture=True);source_audit.audit(source)
        study.execute(out,source,fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        original=torch.load(out/'traces.pt',weights_only=True);original_report=json.loads((out/'report.json').read_text())
        def changed(mutator):
            bundle=copy.deepcopy(original);mutator(bundle);torch.save(bundle,out/'traces.pt')
            report=copy.deepcopy(original_report);report['traces_sha256']=study.prior.digest(out/'traces.pt');study.prior.write(out/'report.json',report)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        changed(lambda x:x[0]['labels'].__setitem__((0,0),0))
        changed(lambda x:x[1]['logits'].__setitem__((1,0,0,0),999.))
        changed(lambda x:x[0]['write_loss_before'].__setitem__((0,0),999.))
        changed(lambda x:x[0]['first_incomplete_step'].__setitem__(0,999))
        changed(lambda x:x[0]['repeat'].__setitem__((0,0),False))
        changed(lambda x:x[0]['final_state'].__setitem__((0,0,0),999.))
        torch.save(original,out/'traces.pt');report=copy.deepcopy(original_report);report['traces_sha256']=study.prior.digest(out/'traces.pt')
        report['summary'][0]['final_correct']+=1;study.prior.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)


if __name__=='__main__':unittest.main()
