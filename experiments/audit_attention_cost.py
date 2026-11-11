"""Audit persisted matrix and cross-check sample rows with earlier scalar code."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import torch
from attention_spikes import streaming_row
from attention_cost import ROOT, digest


def main():
    torch.set_num_threads(1)
    rp=ROOT/'results/attention_cost.json'; r=json.loads(rp.read_text()); assert r['complete']
    assert digest(ROOT/'experiments/attention_cost.py')==r['source_sha256']
    assert digest(ROOT/'docs/attention-cost-protocol.md')==r['protocol_sha256']
    assert {(x['n'],x['seed'],x['method']) for x in r['runs']}==set(itertools.product((256,512,1024,2048),(901,907),('dense','tiled')))
    errors=[]; scalar=[]; summary=[]
    for n in (256,512,1024,2048):
        for seed in (901,907):
            pair=[x for x in r['runs'] if x['n']==n and x['seed']==seed]
            torch.manual_seed(seed); q,k,v=[torch.randn(n,64,dtype=torch.float64) for _ in range(3)]
            sha=hashlib.sha256(json.dumps([t.tolist() for t in (q,k,v)]).encode()).hexdigest()
            outputs=[]
            for row in pair:
                assert sha==row['input_sha256'] and len(row['times_seconds'])==3 and min(row['times_seconds'])>0
                assert row['after']['peak_working_set']>=row['before']['peak_working_set']
                assert row['peak_growth']==row['after']['peak_working_set']-row['before']['peak_working_set']
                path=ROOT/row['output']; assert digest(path)==row['output_sha256']; out=torch.load(path,weights_only=True)
                assert out.shape==(n,64) and torch.isfinite(out).all(); outputs.append(out)
                for i in (0,127,n-1):
                    scores=(q[i]@k[:i+1].T/8).tolist(); ref=torch.tensor(streaming_row(scores,v[:i+1].tolist()),dtype=torch.float64)
                    error=(ref-out[i]).abs().max().item(); assert error<1e-10; scalar.append(error)
            error=(outputs[0]-outputs[1]).abs().max().item(); assert error<1e-10; errors.append(error)
        for method in ('dense','tiled'):
            rows=[x for x in r['runs'] if x['n']==n and x['method']==method]; times=[t*1000 for x in rows for t in x['times_seconds']]
            summary.append(dict(n=n,method=method,median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                peak_mib=[x['after']['peak_working_set']/2**20 for x in rows],peak_growth_mib=[x['peak_growth']/2**20 for x in rows],
                single_score_buffer_mib=(8*n*n if method=='dense' else 8*128*128)/2**20))
    assert max(errors)==r['max_output_error']
    result=dict(passed=True,workers=16,timings=48,max_paired_error=max(errors),max_scalar_row_error=max(scalar),
                report_sha256=digest(rp),audit_sha256=digest(Path(__file__)),scalar_source_sha256=digest(ROOT/'experiments/attention_spikes.py'),summary=summary)
    (ROOT/'results/attention_cost_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
