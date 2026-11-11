"""Independent scalar witness and resource-count verification."""
import json
import math
from cross_attention import ROOT
from position_distractors import digest
from recurrent_retrieval import write


def main():
    source=ROOT/'results/linear_attention.json'
    report=json.loads(source.read_text())
    assert report['passed'] and len(report['cases'])==72
    assert all(digest(ROOT/p)==h for p,h in report['provenance'].items())
    assert all(digest(ROOT/p)==h for p,h in report['historical'].items())
    reads=[]
    for q in (-2.,0.,1.,2.):
        feature=math.exp(q) if q<0 else q+1
        values=[sum(feature*(k+1)*v for k,v in zip((0.,1.),memory))/sum(feature*(k+1) for k in (0.,1.)) for memory in ((0.,1.),(2.,0.))]
        assert abs(values[0]-values[1])<=1e-15
        assert abs(values[0]-2/3)<=1e-15
        reads.append(dict(q=q,reads=values))
    for c in report['counts']:
        state=[[0.]*c['dv'] for _ in range(c['m'])]
        assert sum(len(row) for row in state)+c['m']==c['state_scalars']
        assert sum(c['m']+c['dv'] for _ in range(c['n']))==c['kv_scalars']
    target=ROOT/'results/linear_attention_audit.json'
    assert not target.exists()
    write(target,dict(passed=True,source_sha256=digest(source),audit_sha256=digest(ROOT/'experiments/audit_linear_attention.py'),scalar_float64_reads=reads,counts_checked=len(report['counts']),note='Main numerical matrix is float64. Its illustrative query_reads field used default float32 scalar tensors; this independent scalar check supplies double-precision reads. Main source/report preserved.'))
    print('Passed: 72 recorded cases, source/history hashes, four independent collision reads, six resource counts.')


if __name__=='__main__':main()
