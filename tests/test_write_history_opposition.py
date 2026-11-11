"""Known partitions and saved-artifact entry with intentional defects."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
import association_load
import write_history_opposition as study
import audit_write_history_opposition as audit
import audit_write_credit_assignment as primary_audit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def check(self,g,ids):
        g=torch.tensor(g,dtype=torch.float64);ids=torch.tensor(ids);mask=torch.ones(len(g),dtype=torch.bool)
        a,rows,pos=study.reduction(g,ids,mask,0,'total');b,other,q=audit.independent(g,ids,mask,0,'total')
        audit.ref.compare(a,b);audit.ref.compare(rows,other);torch.testing.assert_close(pos,q);return a

    def test_known_partitions_and_different_values(self):
        a=self.check([[[1.],[-1.]]],[[0,0]])
        self.assertEqual(a['within'],-2.);self.assertEqual(a['between_temporal'],0.);self.assertEqual(a['between_same_position'],0.)
        a=self.check([[[1.],[0.]],[[0.],[-1.]]],[[0,0],[0,0]])
        self.assertEqual(a['within'],0.);self.assertEqual(a['between_temporal'],-2.)
        a=self.check([[[1.]], [[-1.]]],[[0],[0]])
        self.assertEqual(a['between_same_position'],-2.);self.assertEqual(a['temporal_cross'],0.)
        a=self.check([[[1.],[-1.]]],[[0,1]])
        self.assertEqual(a['within'],0.);self.assertEqual(a['joint_energy'],2.)
        a=self.check([[[1.,0.],[0.,1.]]],[[0,0]])
        self.assertEqual(a['within'],0.)
        a=self.check([[[1.],[1.]],[[1.],[1.]]],[[0,0],[0,0]])
        self.assertEqual((a['within'],a['between_temporal'],a['between_same_position']),(4.,4.,4.))
        a=self.check([[[0.],[0.]]],[[0,0]])
        self.assertIsNone(a['within_over_event']);self.assertEqual(a['supervised_unresolved'],1)

    def test_actual_saved_entry_and_deliberate_errors(self):
        folder=study.ROOT/'results/test_runs'/('history_opposition_'+uuid.uuid4().hex);folder.mkdir()
        torch.manual_seed(47);model=association_load.LoadReader('delta')
        with torch.no_grad():model.output.weight.zero_();model.output.bias.zero_()
        torch.save(dict(model=model.state_dict()),folder/'initial.pt')
        rows=[dict(events=[[0,0],[1,1],[0,0]],query=0,label=0),dict(events=[[1,2],[0,0],[1,2]],query=1,label=2)]
        study.primary.prior.write(folder/'data.json',dict(pools={'fixture':rows}))
        case=dict(seed=47,checkpoint=(folder/'initial.pt').relative_to(study.ROOT).as_posix(),data=(folder/'data.json').relative_to(study.ROOT).as_posix(),pool='fixture')
        source=folder/'primary';study.primary.execute(source,[case],fixture=True);primary_audit.audit(source)
        out=folder/'reduction';study.execute(source,out,fixture=True);self.assertTrue(audit.audit(out,persist=False)['passed'])
        original=json.loads((out/'report.json').read_text())
        changed=copy.deepcopy(original);changed['summary'][0]['within']=1.;study.primary.prior.write(out/'report.json',changed)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        changed=copy.deepcopy(original);changed['summary'].pop();study.primary.prior.write(out/'report.json',changed)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        study.primary.prior.write(out/'report.json',original)
        text=(out/'histories.csv').read_text();(out/'histories.csv').write_text(text.replace('unresolved','negative',1))
        changed=copy.deepcopy(original);changed['histories_sha256']=study.primary.prior.digest(out/'histories.csv');study.primary.prior.write(out/'report.json',changed)
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)
        (out/'histories.csv').write_text(text);study.primary.prior.write(out/'report.json',original)
        tensors=torch.load(source/'gradients.pt',weights_only=True);tensors[0]['gradients'].pop('error');torch.save(tensors,source/'gradients.pt')
        with self.assertRaises(AssertionError):audit.audit(out,persist=False)


if __name__=='__main__':unittest.main()
