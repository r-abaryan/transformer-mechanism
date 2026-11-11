"""Four-way score transitions, initialized evaluator/audit reload and inverse controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(ROOT/'experiments'))
import feedback_loss_decomposition as study
import audit_feedback_loss_decomposition as audit
from test_query_feedback_training import payload


class LossDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_four_transitions_and_wrong_contribution(self):
        old=torch.tensor([[2,0,0,0,0,0],[0,2,0,0,0,0],[0,2,0,0,0,0],[2,0,0,0,0,0]],dtype=torch.float64)[None]
        new=torch.tensor([[4,0,0,0,0,0],[0,4,0,0,0,0],[4,0,0,0,0,0],[0,4,0,0,0,0]],dtype=torch.float64)[None]
        y=torch.zeros(1,4,dtype=torch.long);repeat=torch.zeros_like(y,dtype=torch.bool)
        a=dict(logits=old,labels=y,repeat=repeat);b=dict(logits=new,labels=y,repeat=repeat)
        actual=study.decompose(a,b);expected,_=audit.reconstruct(a,b);audit.compare(actual,expected)
        self.assertEqual([v['count'] for v in actual['all']['transitions'].values()],[1,1,1,1])
        self.assertIsNone(actual['repeat']['ce_change'])
        self.assertAlmostEqual(sum(v['ce_contribution'] for v in actual['all']['transitions'].values()),actual['all']['new_ce']-actual['all']['old_ce'])
        for cell in (a,b):cell['logits']+=1000
        audit.compare(study.decompose(a,b),expected)
        wrong=copy.deepcopy(actual);wrong['all']['transitions']['wrong_both']['ce_contribution']+=.01
        with self.assertRaisesRegex(AssertionError,'scalar'):audit.compare(wrong,expected)
        wrong=copy.deepcopy(actual);wrong['all']['transitions']['rescued']['count']=0
        with self.assertRaisesRegex(AssertionError,'Count'):audit.compare(wrong,expected)
        bad=copy.deepcopy(b);bad['labels'][0,0]=1
        with self.assertRaisesRegex(AssertionError,'Unaligned'):study.decompose(a,bad)

    def test_actual_initialized_saved_evaluator_and_audit_reload(self):
        rows=payload()['pools']['train_32'];folder=ROOT/'results/test_runs'/('loss_decomposition_'+uuid.uuid4().hex);folder.mkdir()
        model=study.parent.parent.model_source.pair(1207)['delta'];cells=[]
        for step,scale in [(0,1),(1,2)]:
            if scale==2:
                with torch.no_grad():model.output.weight.mul_(2);model.output.bias.mul_(2)
            path=folder/f'{step}.pt';torch.save(dict(model=model.state_dict(),step=step),path)
            loaded=study.parent.parent.previous.load_model(path)
            cell=study.parent.parent.evaluate(loaded,rows);cell['checkpoint']=path.relative_to(ROOT).as_posix()
            self.assertLess(audit.previous.audit_cell(path,rows,cell)['max_logit_error'],1e-9);cells.append(cell)
        a,b=cells;record=dict(groups=study.decompose(a,b),top_loss_increases=study.examples(a,b,rows))
        self.assertLess(audit.audit_pair(a,b,rows,record),1e-9)
        self.assertEqual(record['groups']['all']['transitions']['rescued']['count'],0)
        self.assertEqual(record['groups']['all']['transitions']['newly_wrong']['count'],0)
        self.assertIsNone(record['groups']['repeat']['old_ce'])
        wrong=copy.deepcopy(record);wrong['top_loss_increases'][0]['key']+=1
        with self.assertRaisesRegex(AssertionError,'identity'):audit.audit_pair(a,b,rows,wrong)
        bad=copy.deepcopy(rows);bad[0]['label']=5
        with self.assertRaisesRegex(AssertionError,'label'):audit.audit_pair(a,b,bad,record)

    def test_ties_counts_and_wrong_probability(self):
        z=torch.zeros(1,2,6,dtype=torch.float64);y=torch.tensor([[0,1]]);repeat=torch.ones_like(y,dtype=torch.bool)
        cell=dict(logits=z,labels=y,repeat=repeat)
        actual=study.decompose(cell,cell);expected,_=audit.reconstruct(cell,cell);audit.compare(actual,expected)
        self.assertEqual(actual['all']['old_accuracy'],.5)
        self.assertEqual(actual['all']['transitions']['wrong_both']['count'],1)
        self.assertEqual(actual['all']['ce_change'],0.)
        wrong=copy.deepcopy(actual);wrong['all']['transitions']['correct_both']['old_true_probability']=.5
        with self.assertRaisesRegex(AssertionError,'scalar'):audit.compare(wrong,expected)
        self.assertEqual(2*(768+192+384)*sum((2,8,16,32)),155904)
        self.assertEqual(2*3*4,24);self.assertEqual(24*3*4,288)


if __name__=='__main__':unittest.main()
