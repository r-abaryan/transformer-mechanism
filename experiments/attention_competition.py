"""Controlled softmax competition and value-content counterexamples."""
import hashlib
import json
import math
from pathlib import Path
from attention_spikes import softmax

ROOT=Path(__file__).resolve().parents[1]
COUNTS=(1,4,16,64,256,1024)


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=ROOT/'results/attention_competition.json'; assert not path.exists()
    report=dict(hashes={p:digest(ROOT/p) for p in ['experiments/attention_competition.py','experiments/attention_spikes.py','docs/attention-competition-protocol.md']},equal=[],maintained=[],unequal=[],values=[])
    for gap in (0,2,4,8):
        previous=1.
        for m in COUNTS:
            p=softmax([gap]+[0.]*m)[0]; expected=1/(1+m*math.exp(-gap))
            assert abs(p-expected)<=1e-12 and p<previous
            previous=p
            report['equal'].append(dict(m=m,gap=gap,target_mass=p,formula=expected))
    for m in COUNTS:
        gap=math.log(m)+math.log(9); p=softmax([gap]+[0.]*m)[0]
        assert abs(p-.9)<=1e-12
        report['maintained'].append(dict(m=m,gap=gap,target_mass=p))
    for gaps in ([2.,2.,2.,2.],[.5,.5,3.5,3.5],[-1.,1.,3.,5.]):
        p=softmax([0.]+[-x for x in gaps])[0]
        exact=1/(1+sum(math.exp(-x) for x in gaps)); mean=sum(gaps)/len(gaps)
        lower=1/(1+len(gaps)*math.exp(-min(gaps))); upper=1/(1+len(gaps)*math.exp(-max(gaps)))
        assert abs(p-exact)<=1e-12 and lower-1e-12<=p<=upper+1e-12
        report['unequal'].append(dict(gaps=gaps,target_mass=p,formula=exact,mean_gap=mean,mean_gap_shortcut=1/(1+len(gaps)*math.exp(-mean)),bounds=[lower,upper]))
    assert abs(report['unequal'][0]['target_mass']-report['unequal'][1]['target_mass'])>.1
    weights=softmax([2.]+[0.]*16); p=weights[0]
    for name,distractors in [('same',[1.]*16),('zero',[0.]*16),('opposite',[-1.]*16),('cancel',[0.]*8+[2.]*8)]:
        values=[1.]+distractors; out=sum(a*v for a,v in zip(weights,values))
        mean=sum(a*v for a,v in zip(weights[1:],distractors))/(1-p)
        error=abs(out-1); identity=(1-p)*abs(mean-1); bound=(1-p)*max(abs(v-1) for v in distractors)
        assert abs(error-identity)<=1e-12 and error<=bound+1e-12
        report['values'].append(dict(name=name,values=values,target_mass=p,output=out,error=error,identity=identity,bound=bound))
    report['passed']=True
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('Passed: 24 equal-score, 6 fixed-mass, 3 unequal-score, 4 value fixtures.')


if __name__=='__main__': main()
