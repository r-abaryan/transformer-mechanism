"""Decimal and exact-integer checks independent of torch SVD/softmax."""
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def det(a):
    n=len(a)
    return sum((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))*math.prod(a[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))


def main():
    p=ROOT/'results/head_width.json'; r=json.loads(p.read_text()); assert r['passed']
    for f,h in r['hashes'].items(): assert digest(ROOT/f)==h
    errors=[]
    def check(a,b):
        error=abs(a-float(b)); assert error<1e-10; errors.append(error)
    with localcontext() as ctx:
        ctx.prec=50
        def softmax(row):
            e=[Decimal(str(x)).exp() for x in row]; return [x/sum(e) for x in e]
        a=[softmax([i*j for j in range(4)]) for i in range(4)]
        assert det(a)>0
        for ar,br in zip(r['width1']['attention'],a):
            for x,y in zip(ar,br): check(x,y)
        check(r['width1']['determinant'],det(a)); check(r['width1']['vandermonde_determinant'],det(a))
        b=[[int(i==j) if i<3 else -1 for j in range(3)] for i in range(4)]
        gram=[[sum(row[i]*row[j] for row in b) for j in range(3)] for i in range(3)]
        assert gram==[[2,1,1],[1,2,1],[1,1,2]] and det(gram)==4
        for i in range(4):
            target=softmax([int(i==j) for j in range(4)])
            for j in range(4):
                check(r['target']['attention'][i][j],target[j]); check(r['width3']['attention'][i][j],target[j])
            for j in range(3): check(r['target']['log_odds'][i][j],b[i][j])
        for actual,expected in zip(r['target']['spectrum']['singular_values'],[2,1,1]): check(actual,expected)
        for entry,expected in zip(r['target']['approximations'],[Decimal(2).sqrt(),Decimal(1)]):
            low=[[Decimal(str(x)) for x in row] for row in entry['log_odds']]
            error=sum((Decimal(b[i][j])-low[i][j])**2 for i in range(4) for j in range(3)).sqrt()
            check(entry['frobenius_error'],error); check(entry['frobenius_error'],expected)
        causal=[[Decimal(1)/Decimal(i+1) if j<=i else Decimal(0) for j in range(4)] for i in range(4)]
        for ar,br in zip(r['causal']['attention'],causal):
            for x,y in zip(ar,br): check(x,y)
        check(r['causal']['determinant'],det(causal))
        rational=[[Fraction(1,i+1) if j<=i else Fraction(0) for j in range(4)] for i in range(4)]
        assert det(rational)==Fraction(1,24)
    result=dict(passed=True,precision=50,max_disagreement=max(errors),checked_scalars=len(errors),report_sha256=digest(p),source_sha256=digest(Path(__file__)))
    (ROOT/'results/head_width_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
