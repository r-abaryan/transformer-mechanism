"""Scalar high-precision finite differences, independent of torch and formulas."""
from decimal import Decimal, localcontext
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dec(x):return Decimal(str(x))


def norm(x,gamma,method,epsilon):
    d=Decimal(len(x));mu=sum(x)/d if method=='ln' else Decimal(0)
    z=[t-mu for t in x];radius=(sum(t*t for t in z)/d+epsilon).sqrt()
    return [g*t/radius for g,t in zip(gamma,z)]


def difference(a,b):return max(abs(dec(x)-y) for x,y in zip(a,b))


def main():
    rp=ROOT/'results/rmsnorm_math.json';ap=ROOT/'results/rmsnorm_math_audit.json'
    assert not ap.exists(),'Preserve completed audit.'
    r=json.loads(rp.read_text());assert r['phase']=='complete' and r['passed']
    for p,h in r['provenance'].items():assert digest(ROOT/p)==h
    expected=set()
    for d,name,epsilon,method,gain in itertools.product((4,16),('ordinary','centered','constant','near_zero','zero'),(0.,1e-5),('ln','rms'),('unit','varying')):
        if epsilon==0 and (name=='zero' or (name=='constant' and method=='ln')):continue
        expected.add((d,name,epsilon,method,gain))
    actual={(c['dimension'],c['vector'],c['epsilon'],c['method'],c['gain']) for c in r['cases']}
    assert actual==expected and len(r['cases'])==len(expected)==68
    errors=[];scalars=0;counterexamples=dict(rms_shift=False,ln_epsilon_scale=False,rms_epsilon_scale=False)
    with localcontext() as context:
        context.prec=70;h=Decimal('1e-22')
        for c in r['cases']:
            d=c['dimension'];name=c['vector'];method=c['method'];epsilon=dec(c['epsilon'])
            indices=list(range(1,d+1))
            vectors={'ordinary':[Decimal(i) for i in indices],
                     'centered':[Decimal(i)-Decimal(d+1)/2 for i in indices],
                     'constant':[Decimal(2)]*d,'near_zero':[Decimal(i if i%2 else -i)*Decimal('1e-6') for i in indices],
                     'zero':[Decimal(0)]*d}
            x=[dec(t) for t in c['input']];gamma=[dec(t) for t in c['gamma']]
            assert max(abs(a-b) for a,b in zip(x,vectors[name]))<=Decimal('1e-20')
            gains=[Decimal(1)]*d if c['gain']=='unit' else [Decimal('.5')+Decimal(i)/d for i in indices]
            assert gamma==gains
            y=norm(x,gamma,method,epsilon);assert difference(c['output'],y)<=Decimal('1e-10');scalars+=d
            columns=[]
            for j in range(d):
                a=x.copy();b=x.copy();a[j]+=h;b[j]-=h
                columns.append([(p-q)/(2*h) for p,q in zip(norm(a,gamma,method,epsilon),norm(b,gamma,method,epsilon))])
            scale=max(Decimal(1),max(abs(t) for col in columns for t in col))
            for field,threshold in (('jacobian',Decimal('1e-10')),('autograd_jacobian',Decimal('1e-10')),('finite_difference_jacobian',Decimal('1e-7'))):
                error=max(abs(dec(c[field][i][j])-columns[j][i])/scale for i in range(d) for j in range(d))
                assert error<=threshold
                if field=='jacobian':errors.append(float(error))
                scalars+=d*d
            transformed={k:norm(v,gamma,method,epsilon) for k,v in (('shift',[t+3 for t in x]),('positive_scale',[t*2 for t in x]),('negative_scale',[-t for t in x]))}
            for k,v in transformed.items():assert difference(c['transformed_outputs'][k],v)<=Decimal('1e-8' if k=='shift' else '1e-10');scalars+=d
            shift=max(abs(a-b) for a,b in zip(transformed['shift'],y));positive=max(abs(a-b) for a,b in zip(transformed['positive_scale'],y))
            if method=='ln':assert shift<=Decimal('1e-50')
            elif shift>Decimal('1e-3'):counterexamples['rms_shift']=True
            if epsilon==0:assert positive<=Decimal('1e-50')
            elif positive>Decimal('1e-3'):counterexamples[method+'_epsilon_scale']=True
            assert max(abs(a+b) for a,b in zip(transformed['negative_scale'],y))<=Decimal('1e-50')
            radius=(sum((t-sum(x)/d if method=='ln' else t)**2 for t in x)/d+epsilon).sqrt()
            z=[t-sum(x)/d if method=='ln' else t for t in x]
            radial=[sum(columns[j][i]*z[j] for j in range(d)) for i in range(d)]
            assert max(abs(a-g*t*epsilon/radius**3)/scale for a,g,t in zip(radial,gamma,z))<=Decimal('1e-30')
            if method=='ln':assert max(abs(sum(columns[j][i] for j in range(d)))/scale for i in range(d))<=Decimal('1e-30')
            if c['gain']=='unit':
                eigen=([Decimal(0),epsilon/radius**3]+[1/radius]*(d-2)) if method=='ln' else [epsilon/radius**3]+[1/radius]*(d-1)
                for field in ('expected','measured'):assert difference(c['spectrum'][field],sorted(eigen))/scale<=Decimal('1e-10')
        assert all(counterexamples.values())
        w=r['centered_witness'];x=[dec(t) for t in w['input']];epsilon=dec(w['epsilon']);radius=(sum(t*t for t in x)/len(x)+epsilon).sqrt()
        assert sum(x)==0 and w['output_error']==0
        assert difference(w['ln_mean_direction'],[Decimal(0)]*len(x))<=Decimal('1e-10')
        assert difference(w['rms_mean_direction'],[1/radius]*len(x))<=Decimal('1e-10')
        assert all(abs(dec(t)-1/(len(x)*radius))<=Decimal('1e-10') for row in w['jacobian_difference'] for t in row)
        w=r['residual_witness'];x=[dec(t) for t in w['input']];shift=dec(w['shift']);epsilon=dec(w['epsilon']);gain=[Decimal(1)]*len(x)
        for method in ('ln','rms'):
            a=norm(x,gain,method,epsilon);b=norm([t+shift for t in x],gain,method,epsilon)
            values=[shift+q-p for p,q in zip(a,b)];assert difference(w['differences'][method],values)<=Decimal('1e-10')
            if method=='ln':assert all(abs(v-shift)<=Decimal('1e-50') for v in values)
    result=dict(passed=True,cases=68,scalar_comparisons=scalars,max_decimal_jacobian_relative_error=max(errors),
                counterexamples=counterexamples,report_sha256=digest(rp),source_sha256=digest(Path(__file__)))
    ap.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
