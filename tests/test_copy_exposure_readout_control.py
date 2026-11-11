"""Repeatable arithmetic boundaries; no frozen evidence is overwritten."""
import copy
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from copy_exposure_readout_control import STAGES,score_pair,scalar_audit,preflight
from copy_exposure_stage_trace import trace
from copy_position_sites import make_model


class ReadoutControlTests(unittest.TestCase):
    def test_saved_preflight(self):
        torch.set_num_threads(1)
        self.assertTrue(preflight()['passed'])

    def test_single_factor_and_terminal_rival(self):
        torch.manual_seed(151);a=trace(make_model('rope').eval(),[[3,4,5]])
        b=copy.deepcopy(a)
        b['readout'][2]+=.02
        for s in STAGES:
            axis=b['readout'][2]-b['readout'][b['competitor']]
            b['margins'][s]=(b['stages'][s]*axis).sum(-1)
        terms=score_pair(a,b);scalar_audit(a,b,terms)
        self.assertTrue(all(torch.count_nonzero(t['representation'])==0 for t in terms.values()))
        state_only=copy.deepcopy(a)
        for s in STAGES:
            state_only['stages'][s]+=.03
            axis=state_only['readout'][2]-state_only['readout'][state_only['competitor']]
            state_only['margins'][s]=(state_only['stages'][s]*axis).sum(-1)
        state_terms=score_pair(a,state_only);scalar_audit(a,state_only,state_terms)
        self.assertTrue(all(torch.count_nonzero(t['classifier'])==0 for t in state_terms.values()))
        # Give the intervention a different terminal rival; audit the separate term.
        old=int(a['competitor'][0,-1]);rival=3 if old!=3 else 4
        b['competitor'][0,-1]=rival
        for s in STAGES:
            axis=b['readout'][2]-b['readout'][b['competitor']]
            b['margins'][s]=(b['stages'][s]*axis).sum(-1)
        terms=score_pair(a,b);scalar_audit(a,b,terms)
        self.assertEqual(float(terms['final']['rival_change'][0,0]),0.)
        b['competitor'][0,-1]=2
        with self.assertRaises(AssertionError):score_pair(a,b)


if __name__=='__main__':unittest.main()
