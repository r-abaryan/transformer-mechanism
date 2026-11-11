"""Independent CPU continuation reference and arithmetic/measurement audit."""
import itertools
import json
import statistics
from pathlib import Path

import torch
from mqa_training import ROOT,model_new
from audit_mqa_training import independent
from position_distractors import digest


def main():
    torch.set_num_threads(1)
    target=ROOT/'results/mqa_decoding.json';r=json.loads(target.read_text());assert r['phase']=='complete'
    prior_path=ROOT/'results/mqa_training.json';prior=json.loads(prior_path.read_text());all_reports=[];errors=[]
    for index in (0,1,2):
        rp=ROOT/f'results/mqa_decoding_worker{index}.json';assert digest(rp)==r['workers'][str(index)]
        report=json.loads(rp.read_text());assert report['phase']=='complete' and report['worker']==index
        assert report['prior_sha256']==digest(prior_path)
        assert digest(ROOT/'results/mqa_training_audit.json')==report['prior_audit_sha256']
        for p,h in report['provenance'].items():assert digest(ROOT/p)==h
        tp=ROOT/f'results/mqa_decoding_worker{index}_tensors.pt';assert digest(tp)==report['tensors_sha256']
        saved=torch.load(tp,weights_only=True);assert len(saved)==len(report['cells'])==12
        assert {(c['batch'],c['prefix'],c['group']) for c in report['cells']}==set(itertools.product((1,16),(14,128,512),('mha','mqa')))
        for cell,t in zip(report['cells'],saved):
            batch,prefix,group=[cell[k] for k in ('batch','prefix','group')]
            assert all(cell[k]==t[k] for k in ('batch','prefix','group'))
            expected=torch.randint(19,(batch,prefix+8),generator=torch.Generator().manual_seed(20270109+prefix));assert torch.equal(t['ids'],expected)
            run=next(x for x in prior['runs'] if x['seed']==1411 and x['group']==group)
            assert cell['checkpoint']==run['checkpoint'] and cell['checkpoint_sha256']==run['checkpoint_sha256']
            cp=ROOT/run['checkpoint'];assert digest(cp)==run['checkpoint_sha256']
            model=model_new(group).eval();model.load_state_dict(torch.load(cp,weights_only=True)['model']);model.float().double()
            with torch.inference_mode():reference=independent(model,t['ids'])[:,prefix:]
            error=(reference-t['reference']).abs().max().item();assert error<=1e-10;errors.append(error)
            assert torch.isfinite(t['output']).all()
            delta=t['output'].double()-t['reference']
            metrics=dict(max_absolute_error=float(delta.abs().max()),relative_l2_error=float(delta.norm()/t['reference'].norm().clamp_min(1e-12)),
                         argmax_differences=int((t['output'].argmax(-1)!=t['reference'].argmax(-1)).sum()))
            assert metrics==cell['correctness']
            passed=metrics['max_absolute_error']<=1e-4 and metrics['relative_l2_error']<=1e-4
            assert (cell['status']=='passed')==passed
            groups=2 if group=='mha' else 1
            assert cell['cache_shapes']==[[batch,groups,prefix+8,8]]*4
            assert cell['cache_bytes']==batch*2*groups*(prefix+8)*16*4
            if passed:
                for metric in ('event','wall'):
                    assert len(cell[metric+'_ms'])==10 and all(x>0 for x in cell[metric+'_ms'])
                    assert statistics.median(cell[metric+'_ms'])==cell[metric+'_median_ms']
                assert cell['peak_allocated']>=cell['baseline_allocated'] and cell['peak_reserved']>=cell['baseline_reserved']
            del model
        all_reports.append(report)
    for shape in r['summary']:
        rows=[]
        for index,report in enumerate(all_reports):
            b,m=[next(c for c in report['cells'] if c['batch']==shape['batch'] and c['prefix']==shape['prefix'] and c['group']==g) for g in ('mha','mqa')]
            assert b['cache_bytes']==2*m['cache_bytes']
            if b['status']==m['status']=='passed':
                rows.append(dict(worker=index,event_ratio=m['event_median_ms']/b['event_median_ms'],wall_ratio=m['wall_median_ms']/b['wall_median_ms']))
        assert rows==shape['process_ratios']
        assert shape['speed_claim_pass']==(len(rows)==3 and all(x['event_ratio']<=.9 and x['wall_ratio']<=.9 for x in rows))
    result=dict(passed=True,cells=len(errors),max_independent_reference_error=max(errors),
                numerical_passed_cells=sum(c['status']=='passed' for p in all_reports for c in p['cells']),
                speed_passed_shapes=sum(s['speed_claim_pass'] for s in r['summary']),report_sha256=digest(target),source_sha256=digest(Path(__file__)))
    (ROOT/'results/mqa_decoding_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
