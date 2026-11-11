"""Regression: native CE subtraction crosses the discrete comparison boundary."""
import copy
import decimal
import sys
import unittest
from pathlib import Path

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
import recover_readout_scale_audit as recovery


class ThresholdTests(unittest.TestCase):
    def test_exact_boundary_and_rejection(self):
        final=[-36.68534058935156,-4.796937465061345,71.98473391389865,-8.025606774742686,-10.091605809227007,-17.162443108964364]
        control=[-13.889568200259808,-2.2557815692483967,25.756332380566665,-3.4336909711248396,-4.187425545207597,-6.767065928174323]
        a=recovery.precise_change(final,control,2,60);b=recovery.precise_change(final,control,2,90)
        self.assertLess(abs(a-b),decimal.Decimal('1e-50'))
        self.assertGreater(a,-decimal.Decimal('1e-12'))
        z=torch.tensor([[final]],dtype=torch.float64);y=torch.tensor([[2]])
        cell=dict(logits=z,labels=y,repeat=torch.ones_like(y,dtype=torch.bool))
        controlled=dict(cell,logits=torch.tensor([[control]],dtype=torch.float64))
        raw=recovery.study.summarize(cell,cell,controlled)
        expected,boundary=recovery.reconstruct(cell,cell,controlled)
        self.assertEqual(raw['all']['transitions']['correct_both']['loss_increased'],1)
        self.assertEqual(expected['all']['transitions']['correct_both']['loss_increased'],0)
        self.assertEqual(len(boundary),1)
        with self.assertRaisesRegex(AssertionError,'Count'):recovery.original.prior.compare(raw,expected)
        changes,error=recovery.reconcile(raw,expected);self.assertEqual(len(changes),2);self.assertLess(error,1e-9)
        wrong=copy.deepcopy(expected);wrong['all']['control_ce']+=.01
        with self.assertRaisesRegex(AssertionError,'scalar'):recovery.reconcile(wrong,expected)


if __name__=='__main__':unittest.main()
