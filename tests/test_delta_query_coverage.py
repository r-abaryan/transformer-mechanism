"""Query labels, one-target objective and actual checkpoint evaluator/audit controls."""
import copy
import sys
import unittest
import uuid
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import delta_query_coverage as study


def rows_for(load):
    events=[[k,k%6] for _ in range(32//load) for k in range(load)]
    return [dict(events=events,query=1,label=1,load=load,group='unique' if load==32 else 'repeat')]


class QueryCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_query_labels_and_sampling_controls(self):
        rows=rows_for(8);idx=torch.zeros(8,dtype=torch.long);slots=torch.arange(8)
        q,y=study.training_queries(rows,idx,slots,'resampled')
        self.assertEqual(q.tolist(),list(range(8)));self.assertEqual(y.tolist(),[k%6 for k in range(8)])
        q,y=study.training_queries(rows,idx,slots,'fixed')
        self.assertEqual(q.tolist(),[1]*8);self.assertEqual(y.tolist(),[1]*8)
        with self.assertRaises(AssertionError):study.training_queries(rows,idx,slots+1,'resampled')
        bad=copy.deepcopy(rows);bad[0]['label']=2
        with self.assertRaises(AssertionError):study.query_tables(bad)
        bad=copy.deepcopy(rows);bad[0]['events'][-1][1]=3
        with self.assertRaises(AssertionError):study.query_tables(bad)

    def test_uniform_one_query_objective_and_gradient(self):
        rows=rows_for(8);model=study.study.pair(3461)['delta'];e,_,_=study.study.parent.tensors(rows)
        queries,labels=study.query_tables(rows);scores=model.read(model.prefill(e),queries[:,0])
        separate=[F.cross_entropy(model(e,queries[:,j]),labels[:,j]) for j in range(8)]
        expected=torch.stack(separate).mean()
        all_scores=torch.cat([model(e,queries[:,j]) for j in range(8)])
        actual=F.cross_entropy(all_scores,labels.flatten())
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        params=tuple(model.parameters());g1=torch.autograd.grad(expected,params);g2=torch.autograd.grad(actual,params)
        for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
        wrong=F.cross_entropy(all_scores,torch.ones(8,dtype=torch.long))
        self.assertGreater(abs(float(wrong.detach()-actual.detach())),1e-5)
        self.assertEqual(scores.shape,(1,6))

    def test_actual_initialized_checkpoint_evaluator_and_audit_all_loads(self):
        folder=ROOT/'results/test_runs'/('query_coverage_'+uuid.uuid4().hex);folder.mkdir(parents=True,exist_ok=False)
        model=study.study.pair(3461)['delta'];cp=folder/'model.pt';torch.save(dict(model=model.state_dict(),step=0),cp)
        for load in study.study.LOADS:
            rows=rows_for(load)
            assigned,scores=study.previous.metrics(model,rows)
            study.audit_cell(cp,rows,False,assigned,scores)
            metrics,scores=study.evaluate_all(model,rows);study.audit_cell(cp,rows,True,metrics,scores)
            self.assertEqual(metrics['all_queries']['count'],load);self.assertEqual(metrics['complete_maps']['count'],1)
            bad=copy.deepcopy(metrics);bad['all_queries']['loss']+=.01
            with self.assertRaises(AssertionError):study.audit_cell(cp,rows,True,bad,scores)
            bad=copy.deepcopy(metrics);bad['complete_maps']['correct']=1-bad['complete_maps']['correct']
            with self.assertRaises(AssertionError):study.audit_cell(cp,rows,True,bad,scores)
            wrong=scores.clone();wrong[...,0]+=.1
            with self.assertRaises(AssertionError):study.audit_cell(cp,rows,True,metrics,wrong)

    def test_development_gate_requires_all_loads_and_exact_pairs(self):
        runs=[]
        for seed in study.DEV_SEEDS:
            for arm in study.ARMS:
                metrics={f'train_{m}':dict(accuracy=1.) for m in study.study.LOADS}
                metrics.update({f'all_train_{m}':dict(all_queries=dict(accuracy=1.)) for m in study.study.LOADS})
                runs.append(dict(seed=seed,arm=arm,metrics=metrics))
        self.assertTrue(study.development_gate(runs)['confirmation_eligible'])
        fixed=next(r for r in runs if r['arm']=='fixed');fixed['metrics']['all_train_32']['all_queries']['accuracy']=.5
        self.assertTrue(study.development_gate(runs)['confirmation_eligible'])
        sampled=next(r for r in runs if r['arm']=='resampled');sampled['metrics']['all_train_8']['all_queries']['accuracy']=.98
        self.assertFalse(study.development_gate(runs)['confirmation_eligible'])
        sampled['metrics']['all_train_8']['all_queries']['accuracy']=1.;fixed['metrics']['train_2']['accuracy']=.98
        self.assertFalse(study.development_gate(runs)['confirmation_eligible'])
        with self.assertRaises(AssertionError):study.development_gate(runs[:-1])

    def test_confirmation_decision_separates_gain_fit_and_map_reliability(self):
        self.assertFalse(study.decision([])['confirmation_executed'])
        runs=[]
        for seed in study.CONFIRM_SEEDS:
            for arm in study.ARMS:
                metrics={f'train_{m}':dict(accuracy=1.) for m in study.study.LOADS}
                metrics.update({f'all_train_{m}':dict(all_queries=dict(accuracy=1.)) for m in study.study.LOADS})
                metrics.update({f'all_test_{m}':dict(all_queries=dict(accuracy=.5 if arm=='fixed' else .7),complete_maps=dict(accuracy=.96)) for m in study.study.LOADS})
                runs.append(dict(seed=seed,arm=arm,metrics=metrics))
        self.assertTrue(study.decision(runs)['retention_supported'])
        sampled=next(r for r in runs if r['arm']=='resampled');sampled['metrics']['all_test_32']['complete_maps']['accuracy']=.1
        result=study.decision(runs);self.assertTrue(result['benefit']);self.assertFalse(result['retention_supported'])
        sampled['metrics']['all_train_16']['all_queries']['accuracy']=.98
        self.assertFalse(study.decision(runs)['benefit'])
        sampled['metrics']['all_train_16']['all_queries']['accuracy']=1.;sampled['metrics']['all_test_32']['all_queries']['accuracy']=.55
        self.assertFalse(study.decision(runs)['benefit'])
        with self.assertRaises(AssertionError):study.decision(runs[:-1])


if __name__=='__main__':unittest.main()
