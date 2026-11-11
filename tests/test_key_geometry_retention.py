"""Geometry, all-query state reuse, actual reload audit and inverse controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import key_geometry_retention as study


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def rows(self):return [dict(events=[[k,k%6] for k in range(32)],query=7,label=1,group='unique',load=32)]

    def test_key_contract_and_padding_inverse(self):
        model=study.study.LoadReader('delta')
        k=study.key_table(model,'native');p=study.key_table(model,'padded');o=study.key_table(model,'orthogonal')
        torch.testing.assert_close(k@k.T,p@p.T,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(o@o.T,torch.eye(32,dtype=o.dtype),atol=0,rtol=0)
        wrong=p.clone();wrong[0,16]=1
        with self.assertRaises(AssertionError):torch.testing.assert_close(k@k.T,wrong@wrong.T,atol=1e-12,rtol=1e-12)
        with self.assertRaisesRegex(AssertionError,'Undeclared'):study.key_table(model,'other')

    def test_actual_reload_audit_and_corrupted_scores_metrics(self):
        folder=ROOT/'results/test_runs'/('key_geometry_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
        model=study.study.LoadReader('delta');cp=folder/'model.pt';torch.save(dict(model=model.state_dict(),step=0),cp)
        rows=self.rows()
        for variant in study.VARIANTS:
            scores,_=study.evaluate(model,rows,variant);metrics=study.summarize(scores,rows)
            study.audit_cell(cp,rows,variant,scores,metrics)
            bad=scores.clone();bad[0,0,0]+=.01
            with self.assertRaisesRegex(AssertionError,'Repetition equations'):study.audit_cell(cp,rows,variant,bad,metrics)
            badmetric=copy.deepcopy(metrics);badmetric['all_queries']['loss']+=.01
            with self.assertRaises(AssertionError):study.audit_cell(cp,rows,variant,scores,badmetric)
            badmetric=copy.deepcopy(metrics);badmetric['complete_maps']['correct']+=1
            with self.assertRaisesRegex(AssertionError,'Wrong complete-map'):study.audit_cell(cp,rows,variant,scores,badmetric)

    def test_all_query_labels_context_gate_and_errors(self):
        rows=self.rows();expanded=study.query_rows(rows)
        self.assertEqual([r['query'] for r in expanded],list(range(32)))
        self.assertEqual([r['label'] for r in expanded],[k%6 for k in range(32)])
        scores=torch.eye(6,dtype=torch.float64)[torch.tensor([[k%6 for k in range(32)]])]
        self.assertEqual(study.summarize(scores,rows)['complete_maps']['correct'],1)
        scores[0,0]=torch.eye(6,dtype=torch.float64)[1]
        self.assertEqual(study.summarize(scores,rows)['complete_maps']['correct'],0)
        self.assertEqual(study.summarize(scores,rows,True)['all_queries']['correct'],31)
        wrong=copy.deepcopy(rows);wrong[0]['label']=2
        with self.assertRaisesRegex(AssertionError,'Wrong query label'):study.query_rows(wrong)
        cells=[dict(seed=1,variant=v) for v in study.VARIANTS];study.check_coverage(cells,(1,))
        with self.assertRaisesRegex(AssertionError,'Missing or duplicate'):study.check_coverage(cells[:-1],(1,))

    def test_shared_state_direct_identity_and_original_reader(self):
        model=study.study.LoadReader('delta');rows=self.rows();e,q,_=study.study.parent.tensors(rows)
        with torch.no_grad():
            scores,_=study.evaluate(model,rows,'native');padded,_=study.evaluate(model,rows,'padded')
            torch.testing.assert_close(scores[:,7],model(e,q),atol=1e-12,rtol=1e-12)
            torch.testing.assert_close(scores,padded,atol=1e-12,rtol=1e-12)
            orth,state=study.evaluate(model,rows,'orthogonal')
            labels=torch.tensor([[k%6 for k in range(32)]])
            expected=model.output(model.write_logit.sigmoid()*model.value(labels))
            torch.testing.assert_close(orth,expected,atol=1e-12,rtol=1e-12)
            self.assertLess(state['direct_read_error'],1e-12)
            # Reversing event order changes no orthogonal query read.
            reversed_rows=[dict(rows[0],events=list(reversed(rows[0]['events'])))]
            reversed_scores,_=study.evaluate(model,reversed_rows,'orthogonal')
            torch.testing.assert_close(orth,reversed_scores,atol=1e-12,rtol=1e-12)


if __name__=='__main__':unittest.main()
