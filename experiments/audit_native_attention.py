"""CPU-side audit of saved GPU outputs, precision and benchmark records."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import torch

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/native_attention.json'; r=json.loads(rp.read_text()); assert r['complete']
    assert digest(ROOT/'experiments/gpu_attention.py')==r['dense_source_sha256']
    assert digest(ROOT/'experiments/native_attention.py')==r['source_sha256']
    assert digest(ROOT/'docs/native-attention-protocol.md')==r['protocol_sha256']
    dp=ROOT/'results/native_attention_tensors.pt'; assert digest(dp)==r['tensors_sha256']; saved=torch.load(dp,weights_only=True)
    assert {(x['n'],x['method']) for x in r['cells']}==set(itertools.product((512,1024,2048,4096),('dense','efficient','cudnn')))
    errors=[]; summary=[]
    for n,part in saved.items():
        q,k,v=[t.double() for t in part['inputs']]; length=int(n); torch.manual_seed(919)
        for original in part['inputs']: assert torch.equal(original,torch.randn(length,64).half())
        # Row-at-a-time float64 reference avoids reusing the saved dense softmax matrix.
        ref=torch.stack([((q[i]@k[:i+1].T/8).softmax(-1)@v[:i+1]) for i in range(length)])
        assert (ref-part['reference']).abs().max()<1e-10
        for method,out in part['outputs'].items():
            row=next(x for x in r['cells'] if x['n']==length and x['method']==method)
            error=(out.double()-ref).abs().max().item(); assert error<=.005 and torch.isfinite(out).all(); errors.append(error)
            assert abs((out.double()-part['reference']).abs().max().item()-row['max_absolute_error'])<1e-12
            assert row['status']=='passed' and len(row['event_ms'])==len(row['wall_ms'])==10
            assert min(row['event_ms'])>0 and min(row['wall_ms'])>0
            assert row['peak_allocated']>=row['baseline_allocated'] and row['peak_reserved']>=row['peak_allocated']
            summary.append(dict(n=length,method=method,event_median_ms=statistics.median(row['event_ms']),
                event_range_ms=[min(row['event_ms']),max(row['event_ms'])],wall_median_ms=statistics.median(row['wall_ms']),
                allocated_mib=row['peak_allocated']/2**20,extra_allocated_mib=(row['peak_allocated']-row['baseline_allocated'])/2**20,
                reserved_mib=row['peak_reserved']/2**20,max_error=error))
    result=dict(passed=True,successful_cells=len(errors),failed_cells=[x for x in r['cells'] if x['status']!='passed'],max_error=max(errors),
                report_sha256=digest(rp),source_sha256=digest(Path(__file__)),summary=summary)
    (ROOT/'results/native_attention_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
