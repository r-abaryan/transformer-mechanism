"""Order, query-present traces, actual checkpoint audit and deliberate errors."""
import copy
import sys
import unittest
import uuid
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import query_write_order as study


class QueryWriteOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def rows(self):
        return [dict(events=[[k, k % 6] for k in range(32)], query=7, label=1, group='unique', load=32)]

    def test_order_contract_labels_and_coverage(self):
        rows = self.rows()
        for variant in study.VARIANTS:
            changed, positions = study.rearrange(rows, variant)
            expected = 7 if variant == 'native' else study.POSITIONS[variant]
            self.assertEqual(positions, [expected]); self.assertEqual(changed[0]['events'][expected], [7, 1])
            self.assertEqual([k for k, _ in changed[0]['events'] if k != 7], [k for k in range(32) if k != 7])
        wrong = copy.deepcopy(rows); wrong[0]['label'] = 2
        with self.assertRaisesRegex(AssertionError, 'Wrong query label'): study.rearrange(wrong, 'first')
        cells = [dict(seed=s, arm=a, variant=v) for s in (1,) for a in ('softmax','delta') for v in study.VARIANTS]
        study.check_coverage(cells, (1,))
        with self.assertRaisesRegex(AssertionError, 'Missing or duplicate'): study.check_coverage(cells[:-1], (1,))

    def test_actual_reload_audit_scalar_and_trace_errors(self):
        folder = ROOT / 'results/test_runs' / ('query_write_order_' + uuid.uuid4().hex)
        folder.mkdir(parents=True, exist_ok=False); rows = self.rows()
        for arm, model in study.study.pair(3323).items():
            cp = folder / f'{arm}.pt'; torch.save(dict(model=model.state_dict(), step=0), cp)
            for variant in study.VARIANTS:
                trace, positions = study.evaluate(model, rows, variant)
                summary = study.summarize(trace, positions, rows)
                study.audit_cell(cp, arm, rows, variant, trace, positions, summary)
                bad = copy.deepcopy(summary); bad['final']['loss'] += .01
                with self.assertRaises(AssertionError): study.audit_cell(cp, arm, rows, variant, trace, positions, bad)
                badtrace = trace.clone(); badtrace[0, 0, 0] += .1
                with self.assertRaisesRegex(AssertionError, 'Repetition equations'): study.audit_cell(cp, arm, rows, variant, badtrace, positions, summary)

    def test_suffix_harm_and_wrong_freeze_control(self):
        model = study.study.LoadReader('delta')
        with torch.no_grad():
            model.key.weight.zero_(); model.key.weight[:, 2] = 1
            model.key.weight[0].zero_(); model.key.weight[0, 0] = 1
            model.key.weight[1].zero_(); model.key.weight[1, :2] = torch.tensor([.6,.8])
            model.value.weight.zero_(); model.value.weight[0, 0] = 1
            model.output.weight.zero_(); model.output.bias.zero_()
            model.output.weight[0, 0] = 1; model.output.bias[1] = .8
        rows = [dict(events=[[0,0]]+[[k,1] for k in range(1,32)], query=0, label=0, group='unique', load=32)]
        trace, positions = study.evaluate(model, rows, 'native')
        ref, _ = study.reference_trace(model, rows, 'native')
        study.prior.check_scores(trace.reshape(-1,6), ref.reshape(-1,6))
        counts = study.summarize(trace, positions, rows)['immediate_to_final']
        self.assertEqual(counts['harmed'], 1)
        frozen = trace[:, :1].expand_as(trace)
        with self.assertRaisesRegex(AssertionError, 'Repetition equations'): study.prior.check_scores(frozen.reshape(-1,6), ref.reshape(-1,6))

    def test_softmax_final_invariance_and_no_absent_query_metrics(self):
        model = study.study.LoadReader('softmax'); rows = self.rows()
        final = None
        for variant in study.VARIANTS:
            trace, positions = study.evaluate(model, rows, variant)
            if final is None: final = trace[:, -1]
            torch.testing.assert_close(trace[:, -1], final, atol=1e-12, rtol=1e-12)
            curve = study.summarize(trace, positions, rows)['curve']
            self.assertEqual(len(curve), 32-positions[0])
            self.assertEqual(curve[0]['count'], 1)


if __name__ == '__main__':
    unittest.main()
