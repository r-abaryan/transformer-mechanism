"""Independent causal-row analytical VJP audit of saved GPU evidence."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/native_backward.json'; tp=ROOT/'results/native_backward_tensors.pt'
    report=json.loads(rp.read_text()); assert report['complete']
    for p,h in report['hashes'].items(): assert digest(ROOT/p)==h
    assert digest(tp)==report['tensors_sha256']
    saved=torch.load(tp,weights_only=True,map_location='cpu')
    assert len(report['cells'])==12
    assert {(r['n'],r['method']) for r in report['cells']}==set(itertools.product((512,1024,2048,4096),('dense','efficient','cudnn')))
    summaries=[]; reference_errors=[]
    for length,part in saved.items():
        n=int(length); q,k,v=[x.double() for x in part['inputs']]; g=part['upstream'].double()
        torch.manual_seed(919)
        for original in part['inputs']: assert torch.equal(original,torch.randn(n,64).half())
        assert torch.equal(part['upstream'],torch.randn(n,64,generator=torch.Generator().manual_seed(929)).half())
        dq=torch.zeros_like(q); dk=torch.zeros_like(k); dv=torch.zeros_like(v); out=torch.empty_like(v)
        for i in range(n):
            a=(q[i]@k[:i+1].T/8).softmax(-1)
            out[i]=a@v[:i+1]
            da=v[:i+1]@g[i]
            ds=a*(da-(a*da).sum())
            dq[i]=ds@k[:i+1]/8
            dk[:i+1]+=ds[:,None]*q[i][None,:]/8
            dv[:i+1]+=a[:,None]*g[i][None,:]
        refs=[out,dq,dk,dv]
        for a,b in zip(refs,[part['reference'],*part['reference_gradients']]):
            err=(a-b).abs().max().item(); assert err<1e-10; reference_errors.append(err)
        rows=[r for r in report['cells'] if r['n']==n]
        assert set(part['outputs'])=={r['method'] for r in rows if r['status']!='runtime_failure'}
        for row in rows:
            if row['status']=='runtime_failure': continue
            actual=part['outputs'][row['method']]
            all_errors=[]
            for index,(a,b,original) in enumerate(zip([actual['output'],*actual['gradients']],refs,[part['reference'],*part['reference_gradients']])):
                assert torch.isfinite(a).all()
                delta=a.double()-b
                err=dict(max_absolute=delta.abs().max().item(),relative_l2=(delta.norm()/b.norm().clamp_min(1e-12)).item())
                recorded=row['output_error'] if index==0 else row['gradient_errors'][index-1]
                for key in err: assert abs(err[key]-recorded[key])<1e-10
                all_errors.append(err)
            passes=all_errors[0]['max_absolute']<=.005 and all(e['max_absolute']<=.02 and e['relative_l2']<=.005 for e in all_errors[1:])
            assert (row['status']=='passed')==passes
            assert len(row['event_ms'])==len(row['wall_ms'])==10
            assert min(row['event_ms'])>0 and min(row['wall_ms'])>0
            assert row['peak_reserved']>=row['peak_allocated']>=row['baseline_allocated']
            summaries.append(dict(n=n,method=row['method'],status=row['status'],event_median_ms=statistics.median(row['event_ms']),
                event_range_ms=[min(row['event_ms']),max(row['event_ms'])],wall_median_ms=statistics.median(row['wall_ms']),
                total_allocated_mib=row['peak_allocated']/2**20,extra_allocated_mib=(row['peak_allocated']-row['baseline_allocated'])/2**20,
                reserved_mib=row['peak_reserved']/2**20,errors=all_errors))
    result=dict(audit_passed=True,passed_cells=sum(r['status']=='passed' for r in report['cells']),
                failed_cells=[r for r in report['cells'] if r['status']!='passed'],
                max_reference_disagreement=max(reference_errors),source_sha256=digest(Path(__file__)),report_sha256=digest(rp),summary=summaries)
    (ROOT/'results/native_backward_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
