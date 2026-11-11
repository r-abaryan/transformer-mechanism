"""Checks for the single zero-recency intervention, without training."""
import copy
import sys
import unittest
import uuid
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import recency_ablation as diagnostic
import delta_retrieval as parent


class RecencyTests(unittest.TestCase):
    def test_zero_bias_equations_and_inverse_error(self):
        e=torch.tensor([[[0,1],[1,4],[0,3],[2,2]],[[1,2],[1,4],[1,0],[3,1]]]);q=torch.tensor([0,1])
        for seed in (3299,3301,3307):
            model=parent.pair(seed)['softmax'];original=parent.state_hash(model.state_dict())
            changed=diagnostic.without_recency(model)
            with torch.no_grad():
                expected=parent.reference(model,e,q,wrong=True)
                diagnostic.check_scores(changed(e,q),expected)
                # The inverse leaves recency enabled and must fail the same check.
                with self.assertRaisesRegex(AssertionError,'Intervention equations disagree'):
                    diagnostic.check_scores(model(e,q),expected)
            self.assertEqual(parent.state_hash(model.state_dict()),original)
            for name,tensor in changed.state_dict().items():
                if name!='log_recency':self.assertTrue(torch.equal(tensor,model.state_dict()[name]))

    def test_permutation_ambiguity_and_initialized_reload_entry(self):
        e=torch.tensor([[[0,1],[1,4],[0,3]],[[0,3],[1,4],[0,1]]]);q=torch.tensor([0,0])
        self.assertNotEqual(parent.label(e[0].tolist(),0),parent.label(e[1].tolist(),0))
        folder=ROOT/'results/test_runs'/('recency_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
        cp=folder/'initial.pt';model=parent.pair(3299)['softmax'];torch.save(model.state_dict(),cp)
        loaded=parent.Reader('softmax');loaded.load_state_dict(torch.load(cp,weights_only=True))
        rows=[dict(events=ev.tolist(),query=0,label=parent.label(ev.tolist(),0),group='repeat') for ev in e]
        metrics,scores,_=diagnostic.evaluate(loaded,rows,'no_recency')
        self.assertLess((scores[0]-scores[1]).abs().max().item(),1e-12)
        self.assertLessEqual(metrics['repeat']['accuracy'],.5)
        self.assertNotIn('unique',metrics)
        native,original,_=diagnostic.evaluate(loaded,rows,'native')
        self.assertGreater((original[0]-original[1]).abs().max().item(),1e-4)

    def test_scalar_labels_and_criterion_errors(self):
        rows=[dict(events=[[0,1],[1,4]],query=0,label=1,group='unique')]
        scores=torch.zeros(1,6,dtype=torch.float64);scores[0,1]=4
        native=parent.metrics(scores,rows)
        diagnostic.check_metrics(native,diagnostic.reduce_scores(scores,rows))
        bad=copy.deepcopy(native);bad['unique']['correct']=0
        with self.assertRaises(AssertionError):diagnostic.check_metrics(bad,diagnostic.reduce_scores(scores,rows))
        wrong=copy.deepcopy(rows);wrong[0]['label']=2
        with self.assertRaises(AssertionError):diagnostic.reduce_scores(scores,wrong)
        with self.assertRaisesRegex(AssertionError,'Missing or duplicate'):diagnostic.decision([])
        metrics={g:dict(accuracy=1.) for g in parent.GROUPS}
        cells=[dict(seed=s,pool=p,mode=m,metrics=copy.deepcopy(metrics)) for s in diagnostic.SEEDS for p in diagnostic.POOLS for m in diagnostic.MODES]
        self.assertTrue(diagnostic.decision(cells)['long_unique_recovery'])
        cell=next(c for c in cells if c['mode']=='no_recency' and c['pool']=='test_32')
        cell['metrics']['unique']['accuracy']=.94
        self.assertFalse(diagnostic.decision(cells)['long_unique_recovery'])
        cell['metrics']['repeat']['accuracy']=.5
        self.assertTrue(diagnostic.decision(cells)['repeat_cost_detected'])


if __name__=='__main__':unittest.main()
