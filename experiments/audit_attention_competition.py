"""Independent high-precision arithmetic audit, without the experiment's softmax."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=ROOT/'results/attention_competition.json'; r=json.loads(path.read_text())
    for p,h in r['hashes'].items(): assert digest(ROOT/p)==h
    assert r['passed']
    counts=(1,4,16,64,256,1024)
    assert len(r['equal'])==24 and {(x['m'],x['gap']) for x in r['equal']}=={(m,g) for m in counts for g in (0,2,4,8)}
    assert len(r['maintained'])==6 and {x['m'] for x in r['maintained']}==set(counts)
    assert [x['gaps'] for x in r['unequal']]==[[2.,2.,2.,2.],[.5,.5,3.5,3.5],[-1.,1.,3.,5.]]
    expected_values={'same':[1.]*16,'zero':[0.]*16,'opposite':[-1.]*16,'cancel':[0.]*8+[2.]*8}
    assert len(r['values'])==4 and {x['name'] for x in r['values']}==set(expected_values)
    errors=[]
    def check(actual,expected):
        err=abs(actual-float(expected)); assert err<=1e-12; errors.append(err)
    def weights(scores):
        exps=[Decimal(str(x)).exp() for x in scores]; total=sum(exps)
        return [x/total for x in exps]
    with localcontext() as ctx:
        ctx.prec=50
        for row in r['equal']+r['maintained']:
            p=weights([row['gap']]+[0]*row['m'])[0]; check(row['target_mass'],p)
            if 'formula' in row: check(row['formula'],p)
            else: check(row['target_mass'],Decimal('.9'))
        for row in r['unequal']:
            p=weights([0]+[-x for x in row['gaps']])[0]
            check(row['target_mass'],p); check(row['formula'],p)
            assert row['bounds'][0]-1e-12<=float(p)<=row['bounds'][1]+1e-12
        w=weights([2]+[0]*16)
        for row in r['values']:
            assert row['values']==[1.]+expected_values[row['name']]
            out=sum(a*Decimal(str(v)) for a,v in zip(w,row['values']))
            check(row['target_mass'],w[0]); check(row['output'],out); check(row['error'],abs(out-1)); check(row['identity'],abs(out-1))
            check(row['bound'],(1-w[0])*max(abs(Decimal(str(v))-1) for v in row['values'][1:]))
    result=dict(passed=True,precision=50,checked_scalars=len(errors),max_absolute_disagreement=max(errors),report_sha256=digest(path),source_sha256=digest(Path(__file__)))
    (ROOT/'results/attention_competition_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
