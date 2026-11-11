"""Ordered criteria, inverse controls and actual saved evaluation/audit fixture."""
import copy
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'tests'))
import recover_memory_first_moment_audit as recovery
import test_memory_first_moment_reset as fixture


class RecoveryTests(unittest.TestCase):
    def test_actual_saved_entry_and_nested_criteria_controls(self):
        before=recovery.original.prior.compare;recovery.original.prior.compare=recovery.compare
        try:
            fixture.FirstMomentTests.setUpClass()
            fixture.FirstMomentTests('test_saved_paired_training_complete_profile_evaluator_audit').test_saved_paired_training_complete_profile_evaluator_audit()
            cells=[]
            for seed in (3461,3463):
                for arm in ('control','reset'):
                    for load in (2,8,16,32):
                        cells.append(dict(seed=seed,arm=arm,pool=f'train_{load}',fixed=dict(error_ce=1. if arm=='control' else .9),
                            metrics=dict(groups=dict(all=dict(ce=.2 if arm=='control' else .19,accuracy=.99)))))
            a=recovery.study.criteria([],cells,cells);b=recovery.original.decision(cells,cells);recovery.compare(a,b)
            bad=copy.deepcopy(a);bad['seeds'].reverse()
            with self.assertRaises(AssertionError):recovery.compare(bad,b)
            bad=copy.deepcopy(a);bad['seeds'].pop()
            with self.assertRaisesRegex(AssertionError,'List coverage'):recovery.compare(bad,b)
            bad=copy.deepcopy(a);bad['seeds'][0]['immediate_benefit']=False
            with self.assertRaises(AssertionError):recovery.compare(bad,b)
        finally:recovery.original.prior.compare=before


if __name__=='__main__':unittest.main()
