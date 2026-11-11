"""Actual resumed training/audit, cap scope, null controls and deliberate defects."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import readout_norm_cap_training as study
import audit_readout_norm_cap_training as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_saved_training_and_audit_defects(self):
        folder=study.ROOT/'results/test_runs'/('norm_cap_'+uuid.uuid4().hex);folder.mkdir()
        model=association_load.pair(89)['delta']
        with torch.no_grad():model.output.weight.mul_(.001);model.output.bias.mul_(.001)
        opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False)
        path=folder/'initial.pt';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=0),path)
        pools=association_load.generate(89,confirmation=True,base_counts=dict(train=2,validation=1,test=1))['pools']
        case=dict(seed=89,checkpoint=path.relative_to(study.ROOT).as_posix());out=folder/'study'
        study.execute(out,[case],pools,fixture=True,updates=4);self.assertTrue(audit.audit(out,persist=False)['passed'])
        report=json.loads((out/'report.json').read_text());run=report['runs'][1];self.assertGreater(run['projected_updates'],0)
        tracepath=study.ROOT/run['traces'];saved=torch.load(tracepath,weights_only=True)
        for defect in (lambda t:t[0]['gradients']['write_logit'].add_(1),lambda t:t[0]['raw_model']['value.weight'].add_(1),
                       lambda t:t[0]['after_model']['key.weight'].add_(1),lambda t:t[0]['indices'].__setitem__(0,(int(t[0]['indices'][0])+1)%12),
                       lambda t:t[0]['projection'].__setitem__('factor',.1)):
            wrong=copy.deepcopy(saved);defect(wrong);torch.save(wrong,tracepath);altered=copy.deepcopy(report);altered['runs'][1]['traces_sha256']=study.training.replay.digest(tracepath);study.training.replay.write(out/'report.json',altered)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        torch.save(saved,tracepath);report['runs'][1]['traces_sha256']=study.training.replay.digest(tracepath);study.training.replay.write(out/'report.json',report)
        scores=torch.load(out/'scores.pt',weights_only=True)
        for field in ('labels','logits'):
            wrong=copy.deepcopy(scores);wrong[0][field].flatten()[0]+=1;torch.save(wrong,out/'scores.pt');altered=copy.deepcopy(report);altered['scores_sha256']=study.training.replay.digest(out/'scores.pt');study.training.replay.write(out/'report.json',altered)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        torch.save(scores,out/'scores.pt');report['scores_sha256']=study.training.replay.digest(out/'scores.pt')
        checkpoint=study.ROOT/run['checkpoint'];original=torch.load(checkpoint,weights_only=True)
        wrong=copy.deepcopy(original);next(iter(wrong['optimizer']['state'].values()))['exp_avg'].add_(1)
        torch.save(wrong,checkpoint);bad=copy.deepcopy(report);bad['runs'][1]['checkpoint_sha256']=study.training.replay.digest(checkpoint);study.training.replay.write(out/'report.json',bad)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        torch.save(original,checkpoint);report['runs'][1]['checkpoint_sha256']=study.training.replay.digest(checkpoint)
        for change in (lambda r:r.__setitem__('evaluation_targets',r['evaluation_targets']+1),lambda r:r['cells'].pop()):
            bad=copy.deepcopy(report);change(bad);study.training.replay.write(out/'report.json',bad)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        report['decision']['training_benefit_all']=not report['decision']['training_benefit_all'];study.training.replay.write(out/'report.json',report)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)

    def test_projection_scope_null_ties_and_missing_groups(self):
        model=association_load.pair(97)['delta'];radius=study.scale.head_norm(model.state_dict())
        state=copy.deepcopy(model.state_dict());opt=torch.optim.Adam(model.parameters(),lr=.003,foreach=False);beforeopt=copy.deepcopy(opt.state_dict())
        null=study.cap_head(model,radius*2,'cap');self.assertEqual(null['factor'],1.)
        self.assertTrue(study.training.replay.equal(state,model.state_dict()))
        with torch.no_grad():
            for p in (model.output.weight,model.output.bias):
                mean=p.mean(0,keepdim=True);p.copy_(mean+2*(p-mean))
        e=torch.tensor([[[0,0],[1,1]]]);q=torch.tensor([[0,1]])
        before=study.training.forward_queries(model,e,q).detach();raw=copy.deepcopy(model.state_dict());projection=study.cap_head(model,radius,'cap')
        self.assertAlmostEqual(projection['factor'],.5)
        self.assertTrue(torch.equal(before.argmax(-1),study.training.forward_queries(model,e,q).argmax(-1)))
        for name in ('key.weight','value.weight','write_logit'):self.assertTrue(torch.equal(raw[name],model.state_dict()[name]))
        self.assertTrue(study.training.replay.equal(beforeopt,opt.state_dict()))
        wanted,_=audit.projection(raw,radius,'cap')
        for name,value in wanted.items():torch.testing.assert_close(value,model.state_dict()[name],atol=1e-11,rtol=0)
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);mask=torch.zeros_like(y,dtype=torch.bool)
        stats=study.fixed(z,y,mask);self.assertEqual(stats['harmed'],1);self.assertIsNone(stats['error_ce'])
        m=audit.reference.metrics(z,y,mask);self.assertEqual(m['groups']['all']['accuracy'],.5);self.assertIsNone(m['groups']['repeat']['ce'])


if __name__=='__main__':unittest.main()
