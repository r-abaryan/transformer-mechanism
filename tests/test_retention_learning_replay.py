"""Actual training loop, saved probes and deliberately corrupted traces."""
import copy
import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import retention_learning_replay as study
import audit_retention_learning_replay as audit
import audit_objective_beta_pressure as objective_audit


def payload():
    pools={}
    for split in ('train','validation'):
        for m in study.pressure.LOADS:
            events=[[k,k%6] for _ in range(32//m) for k in range(m)]
            pools[f'{split}_{m}']=[dict(events=events,query=0,label=0,load=m,
                group='unique' if m==32 else 'repeat'),dict(events=list(reversed(events)),query=1,
                label=1,load=m,group='unique' if m==32 else 'repeat')]
    return dict(pools=pools)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_actual_loop_unchanged_and_audited(self):
        data=payload()
        for arm in study.parent.ARMS:
            base=ROOT/'results/test_runs'/('retention_replay_'+uuid.uuid4().hex)
            plain,observed=base/'plain',base/'observed'
            plain.mkdir(parents=True);observed.mkdir()
            baseline=dict(runs=[]);instrumented=dict(runs=[])
            with patch.object(study.parent,'OUT',plain),patch.object(study.parent,'STEPS',8):
                study.parent.train_run('development',1193,arm,data,baseline,{})
            study.replay_run(observed,1193,arm,data,instrumented,{},probes=(0,4,8),steps=8)
            a=torch.load(ROOT/baseline['runs'][0]['checkpoint'],weights_only=True)
            b=torch.load(ROOT/instrumented['runs'][0]['checkpoint'],weights_only=True)
            self.assertTrue(study.equal(a,b))
            for field in ('metrics','initial_sha256','history_stream_sha256','actual_queries_labels_sha256','exposure'):
                self.assertEqual(baseline['runs'][0][field],instrumented['runs'][0][field])
            checked=audit.audit_trajectory(observed,1193,arm,data,(0,4,8),8)
            self.assertEqual(checked['cells'],24)
            records=json.loads((observed/f'trajectory_1193_{arm}.json').read_text())
            wrong=copy.deepcopy(records['steps']);wrong[3]['alpha_after']+=.001
            with self.assertRaisesRegex(AssertionError,'Adam recurrence'):
                audit.check_steps(wrong,records['steps'][0]['alpha_before'])
            altered=copy.deepcopy(b);altered['optimizer']['state'][0]['exp_avg']+=.01
            self.assertFalse(study.equal(a,altered))

    def test_evaluator_supplied_logits_groups_and_controls(self):
        logits=torch.tensor([[[0.,4.,0.,0.,0.,0.]]],dtype=torch.float64)
        labels=torch.tensor([[1]]);repeat=torch.ones(1,1,dtype=torch.bool)
        native=study.pressure.reduce_logits(logits,labels,repeat)
        independent=objective_audit.metrics(logits,labels,repeat)
        self.assertEqual(native['all']['accuracy'],1.)
        self.assertIsNone(native['unique']['ce'])
        objective_audit.close(native['all']['ce'],independent['all']['ce'])
        with self.assertRaises(AssertionError):objective_audit.close(native['all']['ce']+.1,independent['all']['ce'])
        rows=payload()['pools']['train_32']
        bad=copy.deepcopy(rows);bad[0]['label']=5
        with self.assertRaisesRegex(AssertionError,'label'):objective_audit.check_rows(bad)
        self.assertEqual(2*2*7*4*2,224)
        self.assertEqual(2*2*7*768*(4+2+8+16+32),1333248)


if __name__=='__main__':unittest.main()
