"""Full saved factorial runtime, intervention scope and intentional defects."""
import copy
import itertools
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import beta_key_transport as study
import audit_beta_key_transport as audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_saved_factorial_and_defects(self):
        folder=study.ROOT/'results/test_runs'/('beta_keys_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(81);model=association_load.LoadReader('delta')
        paths={}
        for t in (3000,12000):
            p=folder/f'{t}.pt';torch.save(dict(model=model.state_dict(),step=t),p);paths[str(t)]=p.relative_to(study.ROOT).as_posix()
        rows=[dict(events=[[i%8,i%6],[(i+1)%8,(i+2)%6]],query=i%8,label=i%6) for i in range(24)]
        study.prior.prior.write(folder/'data.json',dict(pools={'train_32':rows}))
        selected=[dict(seed=81,anchor=a,keys=k,beta=b,checkpoints=paths,data=(folder/'data.json').relative_to(study.ROOT).as_posix()) for a,k,b in itertools.product((3000,12000),repeat=3)]
        out=folder/'study';study.execute(out,selected,fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        saved=torch.load(out/'traces.pt',weights_only=True);report=json.loads((out/'report.json').read_text())
        for b in saved:torch.testing.assert_close(b['logits'],saved[0]['logits'],atol=0,rtol=0)
        for row in report['contrasts']:
            for k,v in row.items():
                if k.startswith(('ratio_','accuracy_','ce_')) and v is not None:self.assertEqual(v,0.)
        for change in (lambda b:b[0]['labels'].__setitem__((0,0),5),lambda b:b[0]['logits'].__setitem__((1,0,0,0),999.),
                       lambda b:b[0]['coefficients'].__setitem__((0,0,0),99.),lambda b:b[0]['case'].__setitem__('keys',12000)):
            wrong=copy.deepcopy(saved);change(wrong);torch.save(wrong,out/'traces.pt')
            altered=copy.deepcopy(report);altered['traces_sha256']=study.prior.prior.digest(out/'traces.pt');study.prior.prior.write(out/'report.json',altered)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        torch.save(saved,out/'traces.pt');report['traces_sha256']=study.prior.prior.digest(out/'traces.pt')
        for field in ('count','ratio_interaction'):
            wrong=copy.deepcopy(report);wrong['contrasts'][0][field]+=1;study.prior.prior.write(out/'report.json',wrong)
            with self.assertRaises(AssertionError):audit.audit(out,persist=False)

    def test_scope_geometry_and_ties(self):
        torch.manual_seed(83);a=association_load.LoadReader('delta');b=copy.deepcopy(a)
        with torch.no_grad():
            b.key.weight.copy_(b.key.weight.roll(1,0));b.write_logit.fill_(-1);b.value.weight.add_(7);b.output.bias.add_(3)
        case=dict(anchor=3000,keys=12000,beta=12000);models={3000:a,12000:b};result=study.hybrid(models,case)
        self.assertTrue(torch.equal(result.value.weight,a.value.weight));self.assertTrue(torch.equal(result.output.bias,a.output.bias))
        self.assertTrue(torch.equal(result.key.weight,b.key.weight));self.assertTrue(torch.equal(result.write_logit,b.write_logit))
        with torch.no_grad():result.value.weight.zero_();result.output.weight.zero_();result.output.bias.zero_()
        rows=[dict(events=[[0,0],[1,1]],query=0,label=0)];mask=torch.tensor([[True,False]])
        x=study.prior.analyze(result,rows,mask,83,3000)
        self.assertEqual(int(x['fields']['final_correct'].sum()),1)
        empty=next(r for r in x['summary'] if r['group']=='correct_in_complete' and r['position'] is None)
        self.assertEqual(empty['count'],0);self.assertIsNone(empty['final_margin_mean'])
        # A common coordinate rotation of every tied key preserves all reads.
        rotated=copy.deepcopy(a)
        with torch.no_grad():rotated.key.weight.copy_(a.key.weight.roll(1,1))
        one=study.prior.analyze(a,rows,mask,83,3000);two=study.prior.analyze(rotated,rows,mask,83,3000)
        torch.testing.assert_close(one['logits'],two['logits'],atol=1e-9,rtol=1e-9)


if __name__=='__main__':unittest.main()
