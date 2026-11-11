"""Scale intervention, real initialized reload/evaluation and inverse controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'tests'))
import readout_scale_control as study
import audit_readout_scale_control as audit
from test_query_feedback_training import payload


class ScaleControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_norm_and_offset_invariance(self):
        model=study.training.model_source.pair(1409)['delta'];state=copy.deepcopy(model.state_dict())
        record=study.scaling(state,state);self.assertEqual(record['scale'],1.)
        for name in ('output.weight','output.bias'):state[name]+=7
        self.assertAlmostEqual(study.head_norm(state),record['final_norm'])
        self.assertAlmostEqual(audit.norm(state),record['final_norm'])
        state['output.weight'].zero_();state['output.bias'].zero_()
        with self.assertRaisesRegex(AssertionError,'norm'):study.head_norm(state)
        with self.assertRaisesRegex(AssertionError,'norm'):audit.norm(state)
        for s in (0,-1,float('nan')):
            with self.assertRaises(AssertionError):study.intervene(model,s)

    def test_supplied_scores_absent_groups_and_wrong_controls(self):
        z=torch.tensor([[[4,0,0,0,0,0],[0,4,0,0,0,0],[0,0,0,0,0,0]]],dtype=torch.float64)
        y=torch.zeros(1,3,dtype=torch.long);r=torch.zeros_like(y,dtype=torch.bool)
        a=dict(logits=z/2,labels=y,repeat=r);b=dict(logits=z,labels=y,repeat=r);c=dict(logits=z/2,labels=y,repeat=r)
        summary=study.summarize(a,b,c);audit.prior.compare(summary,audit.reconstruct(a,b,c))
        self.assertIsNone(summary['repeat']['control_ce']);self.assertEqual(summary['all']['prediction_changes'],0)
        self.assertEqual(summary['all']['transitions']['correct_both']['loss_increased'],1)
        self.assertEqual(summary['all']['transitions']['wrong_both']['loss_decreased'],1)
        wrong=copy.deepcopy(summary);wrong['all']['control_ce']+=.1
        with self.assertRaisesRegex(AssertionError,'scalar'):audit.prior.compare(wrong,audit.reconstruct(a,b,c))
        bad=copy.deepcopy(c);bad['labels'][0,0]=1
        with self.assertRaisesRegex(AssertionError,'labels'):study.summarize(a,b,bad)
        bad=copy.deepcopy(c);bad['logits']=-z
        with self.assertRaisesRegex(AssertionError,'winner'):study.summarize(a,b,bad)

    def test_actual_saved_evaluator_matrix_audit_and_memory_control(self):
        rows=payload()['pools']['train_32'];folder=ROOT/'results/test_runs'/('scale_control_'+uuid.uuid4().hex);folder.mkdir()
        model=study.training.model_source.pair(1423)['delta'];old=copy.deepcopy(model.state_dict())
        with torch.no_grad():model.output.weight.mul_(2);model.output.bias.mul_(2)
        record=study.scaling(old,model.state_dict());self.assertAlmostEqual(record['scale'],.5)
        control=study.intervene(model,record['scale'])
        audit.verify_head(old,model.state_dict(),control.state_dict(),record)
        cells=[]
        for name,state in (('old',old),('final',model.state_dict()),('control',control.state_dict())):
            path=folder/(name+'.pt');torch.save(dict(model=state,step=0),path)
            loaded=study.training.previous.load_model(path);cell=study.training.evaluate(loaded,rows)
            self.assertLess(audit.prior.previous.audit_cell(path,rows,cell)['max_logit_error'],1e-9);cells.append(cell)
        a,b,c=cells;summary=dict(groups=study.summarize(a,b,c),complete_maps=c['metrics']['complete_maps'])
        self.assertLess(audit.audit_cell(a,b,c,rows,summary),1e-9)
        events,_,_,_=study.training.replay.pressure.prepare(rows,'all_queries')
        self.assertTrue(torch.equal(model.prefill(events)[0],control.prefill(events)[0]))
        bad=copy.deepcopy(control.state_dict());bad['key.weight'][0,0]+=.01
        with self.assertRaisesRegex(AssertionError,'Memory'):audit.verify_head(old,model.state_dict(),bad,record)
        bad=copy.deepcopy(control.state_dict());bad['output.bias']+=.1
        with self.assertRaisesRegex(AssertionError,'head'):audit.verify_head(old,model.state_dict(),bad,record)
        wrong=dict(record);wrong['scale']*=2
        with self.assertRaisesRegex(AssertionError,'scalar'):audit.verify_head(old,model.state_dict(),control.state_dict(),wrong)
        # Scaling W alone can change a winner when b remains fixed.
        weight=torch.tensor([4.,0.]);bias=torch.tensor([0.,3.])
        self.assertNotEqual(int((weight+bias).argmax()),int((weight*.5+bias).argmax()))


if __name__=='__main__':unittest.main()
