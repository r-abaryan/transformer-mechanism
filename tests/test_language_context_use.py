"""Fresh fixtures only; never invoke archived recording entry points."""
import copy
import sys
import uuid
import unittest
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments'))
from language_context_use import make_inputs, transform, evaluate, metrics, CONDITIONS
from audit_language_context_use import (verify_inputs, independent_input, independent_metrics,
    verify_metrics, independent_logits, audit_checkpoint)
from wikitext2_training import model_new, configure, forward_device


class ContextUseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def fixture(self):
        g = torch.Generator().manual_seed(413)
        stream = torch.randint(13, (193,), generator=g)
        return stream, make_inputs(stream)

    def test_labels_permutations_and_error_control(self):
        stream, inputs = self.fixture()
        self.assertEqual(verify_inputs(inputs, stream), 3)
        bad = copy.deepcopy(inputs); bad['targets'][0] = (bad['targets'][0]+1)%13
        with self.assertRaises(AssertionError): verify_inputs(bad, stream)
        for condition in CONDITIONS:
            native, offset = transform(inputs, condition, 0, 3)
            other, position = independent_input(inputs, condition, 0, 3)
            self.assertTrue(torch.equal(native, other)); self.assertEqual(offset+4, position)
            self.assertTrue(torch.equal(native[:,-1], inputs['ids'][:,-1]))

    def test_actual_saved_checkpoint_evaluator_and_audit(self):
        _, inputs = self.fixture()
        torch.manual_seed(617)
        model = model_new('mqa', 13).double().eval()
        modes = [('cpu', torch.float64)] + ([('cuda', torch.float32)] if torch.cuda.is_available() else [])
        for device, dtype in modes:
            deployed = copy.deepcopy(model).to(device=device, dtype=dtype)
            cells = {c:evaluate(deployed, inputs, c) for c in CONDITIONS}
            records = {c:metrics(cells[c], cells['original']) for c in CONDITIONS}
            # Retain disposable reload fixture; do not touch historical outputs.
            directory = Path(__file__).parents[1]/'results/test_runs'/('context_use_'+uuid.uuid4().hex)
            directory.mkdir()
            checkpoint = directory/'initial.pt'; torch.save(dict(model=deployed.state_dict()), checkpoint)
            result = audit_checkpoint(checkpoint, 13, inputs, cells, records)
            self.assertLess(result['max_token_ce_error'], 1e-10 if device == 'cpu' else 1e-4)
            broken = copy.deepcopy(cells); broken['recent_only']['losses'][0] += .1
            with self.assertRaises(AssertionError): audit_checkpoint(checkpoint, 13, inputs, broken, records)

    def test_native_forward_and_position_control(self):
        _, inputs = self.fixture()
        torch.manual_seed(619); model = model_new('mqa', 13).double().eval()
        cells = {c:evaluate(model, inputs, c) for c in CONDITIONS}
        with torch.inference_mode():
            expected = forward_device(model, inputs['ids'])[:,-1]
            actual = cells['original']['fixtures'][0]['logits']
            self.assertLess(float((expected[:2]-actual).abs().max()), 1e-10)
            suffix = inputs['ids'][:,32:]
            # RoPE common shifts preserve relative positions in this pure-RoPE model.
            a = independent_logits(model, suffix, 4)
            b = independent_logits(model, suffix, 36)
            self.assertLess(float((a-b).abs().max()), 1e-10)
            self.assertGreater(float((expected[:2]-cells['recent_shuffle']['fixtures'][0]['logits']).abs().max()), 1e-6)

    def test_supplied_scores_missing_groups_and_bad_metrics(self):
        target = torch.tensor([2,3,4])
        scores = torch.tensor([[0.,0.,3.,0.,0.],[0.,0.,0.,2.,0.],[0.,0.,3.,0.,0.]])
        loss = scores.logsumexp(1)-scores[torch.arange(3),target]
        cell = dict(targets=target, predictions=scores.argmax(1), losses=loss)
        result = metrics(cell, cell)
        verify_metrics(result, independent_metrics(cell, cell))
        self.assertEqual(result['unknown']['count'], 0); self.assertIsNone(result['row_end']['ce'])
        result['all']['rescues'] = 1
        with self.assertRaises(AssertionError): verify_metrics(result, independent_metrics(cell, cell))


if __name__ == '__main__':
    unittest.main()
