"""Decimal output construction and rational identifiability checks."""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=ROOT/'results/output_capacity.json'; r=json.loads(path.read_text()); assert r['passed']
    for p,h in r['hashes'].items(): assert digest(ROOT/p)==h
    w=r['reused_s05']; a,b=[[Fraction(str(x)) for x in w[k]] for k in ['attention_a','attention_b']]
    assert sum(a)==sum(b)==1 and sum((x-y)*i for i,(x,y) in enumerate(zip(a,b)))==0
    assert a!=b
    scalar=r['scalar']; unique=r['identifiable']
    assert scalar['values']==scalar['k']==[[0.],[1.],[2.],[3.]] and scalar['augmented_rank']==2
    assert unique['values']==[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]]
    assert unique['value_rank']==3 and unique['augmented_rank']==4
    # [V,1] is upper triangular with unit diagonal: determinant 1.
    # [scalar V,1] has two independent columns (its first two rows suffice).
    errors=[]; output_diffs=[]; attention_diffs=[]; recovery_diffs=[]; controls=[]
    def check(x,y):
        err=abs(x-float(y)); assert err<1e-10; errors.append(err)
    with localcontext() as ctx:
        ctx.prec=50
        for i in range(4):
            e=[Decimal(int(i==j)).exp() for j in range(4)]; target=[x/sum(e) for x in e]
            q=Decimal(str(scalar['q'][i][0])); e=[(q*j).exp() for j in range(4)]; other=[x/sum(e) for x in e]
            actual=sum(x*j for j,x in enumerate(other)); desired=sum(x*j for j,x in enumerate(target))
            output_diffs.append(abs(actual-desired)); assert abs(actual-desired)<Decimal('1e-10')
            check(scalar['target_output'][i][0],desired); check(scalar['actual_output'][i][0],actual)
            for j in range(4):
                check(r['target_attention'][i][j],target[j]); check(scalar['attention'][i][j],other[j])
                attention_diffs.append(abs(target[j]-other[j]))
                recovered=target[j] if j<3 else 1-sum(target[:3])
                check(unique['recovered_attention'][i][j],recovered); recovery_diffs.append(abs(recovered-target[j]))
                if j<3:
                    check(unique['output'][i][j],target[j]); controls.append(abs(other[j]-target[j]))
        check(scalar['max_output_error'],max(output_diffs)); check(scalar['max_attention_difference'],max(attention_diffs))
        check(unique['max_recovery_error'],max(recovery_diffs)); check(unique['wrong_attention_output_error'],max(controls))
        assert max(attention_diffs)>Decimal('.01') and max(controls)>Decimal('.01')
    result=dict(passed=True,checked_scalars=len(errors),precision=50,max_disagreement=max(errors),report_sha256=digest(path),source_sha256=digest(Path(__file__)))
    (ROOT/'results/output_capacity_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
