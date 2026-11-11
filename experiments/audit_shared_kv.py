"""Decimal attention and exact-rational rank checks, independent of torch."""
from decimal import Decimal,localcontext
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attention(q,k,v,causal=False):
    weights,outputs = [],[]
    scale = Decimal(len(q[0])).sqrt()
    for row_i,row in enumerate(q):
        logits = [sum(Decimal(str(x))*Decimal(str(y)) for x,y in zip(row,kr))/scale for kr in k]
        e = [s.exp() if not causal or j<=row_i else Decimal(0) for j,s in enumerate(logits)]
        a = [p/sum(e) for p in e]
        weights.append(a)
        outputs.append([sum(a[j]*Decimal(str(v[j][d])) for j in range(len(k))) for d in range(len(v[0]))])
    return weights,outputs


def main():
    path = ROOT/'results/shared_kv.json';r=json.loads(path.read_text());assert r['passed']
    for p,h in r['hashes'].items():assert digest(ROOT/p)==h
    assert digest(ROOT/'results/head_width.json')==r['prior_head_width_sha256']
    prior_audit=json.loads((ROOT/'results/head_width_audit.json').read_text())
    assert prior_audit['passed'] and prior_audit['report_sha256']==r['prior_head_width_sha256']
    assert digest(ROOT/'results/head_width_audit.json')==r['prior_head_width_audit_sha256']
    errors=[]
    def check(actual,expected):
        error=abs(float(actual)-float(expected));assert error<=1e-10;errors.append(error)
    def matrix(actual,expected):
        assert len(actual)==len(expected)
        for ar,er in zip(actual,expected):
            assert len(ar)==len(er)
            for x,y in zip(ar,er):check(x,y)
    assert [c['groups'] for c in r['storage']]==[4,2,1]
    for c in r['storage']:
        g=c['groups'];assert c['cache_bytes']==2*2*128*g*5*8
        assert c['projection_parameters']==160+40*g
        assert c['k_shape']==[2,g,128,2] and c['v_shape']==[2,g,128,3]
    assert {(c['groups'],c['causal']) for c in r['equivalence']}=={(g,m) for g in (4,2,1) for m in (False,True)}
    with localcontext() as context:
        context.prec=50
        for c in r['equivalence']:
            q=[[[float((i%7)-3)/3 for i in range(h*10+n*2,h*10+n*2+2)] for n in range(5)] for h in range(4)]
            assert c['q']==q
            expected_k=[[[float((i%5)-2)/2 for i in range(g*10+n*2,g*10+n*2+2)] for n in range(5)] for g in range(c['groups'])]
            expected_v=[[[float((i%11)-5)/4 for i in range(g*15+n*3,g*15+n*3+3)] for n in range(5)] for g in range(c['groups'])]
            assert c['k']==expected_k and c['v']==expected_v
            for h in range(4):
                group=h//(4//c['groups'])
                a,y=attention(c['q'][h],c['k'][group],c['v'][group],c['causal'])
                matrix(c['attention'][h],a);matrix(c['dense_attention'][h],a)
                matrix(c['output'][h],y);matrix(c['dense_output'][h],y)
            error=max(abs(x-y) for name in ('attention','output') for head,dense_head in zip(c[name],c['dense_'+name]) for row,dense_row in zip(head,dense_head) for x,y in zip(row,dense_row))
            check(c['max_error'],error)
            different=max(abs(x-y) for row,other in zip(c['attention'][0],c['attention'][1]) for x,y in zip(row,other))
            check(c['head01_attention_difference'],different)
            if c['groups']==1 and not c['causal']:assert different>1e-3
        t=r['routing'];assert t['scores']==[[1.,0.,0.],[0.,1.,0.]]
        target=[]
        for row in t['scores']:
            e=[Decimal(str(s)).exp() for s in row];target.append([x/sum(e) for x in e])
        matrix(t['attention'],target)
        odds=[[(row[j]/row[2]).ln() for j in range(2)] for row in target]
        matrix(t['joint_log_odds'],odds)
        exact=[[Fraction(str(row[j]))-Fraction(str(row[2])) for j in range(2)] for row in t['scores']]
        assert exact[0][0]*exact[1][1]-exact[0][1]*exact[1][0]==1
        assert t['spectrum']['rank']==2
        for s in t['spectrum']['singular_values']:check(s,1)
        assert t['shared_width2_k']==[[1.,0.],[0.,1.],[0.,0.]]
        matrix(t['shared_width2_q'],[[Decimal(2).sqrt(),0],[0,Decimal(2).sqrt()]])
        a,_=attention(t['shared_width2_q'],t['shared_width2_k'],[[0],[1],[2]])
        matrix(t['recovered_attention'],a);matrix(t['recovered_attention'],target)
        check(t['max_error'],max(abs(float(x)-float(y)) for row,tar in zip(a,target) for x,y in zip(row,tar)))
        p=r['pooling'];assert p['q']==[[1.]] and p['k']==[[[1.],[0.],[0.]],[[0.],[1.],[0.]]]
        assert p['v']==[[[0.],[1.],[2.]],[[3.],[4.],[5.]]]
        before=[attention(p['q'],k,v)[1][0][0] for k,v in zip(p['k'],p['v'])]
        mk=[[sum(Decimal(str(k[j][0])) for k in p['k'])/2] for j in range(3)]
        mv=[[sum(Decimal(str(v[j][0])) for v in p['v'])/2] for j in range(3)]
        after=attention(p['q'],mk,mv)[1][0][0]
        for x,y in zip(p['original_outputs'],before):check(x,y)
        for x in p['pooled_outputs']:check(x,after)
        check(p['max_output_error'],max(abs(x-after) for x in before));assert p['max_output_error']>1e-3
        for x in p['identical_head_output']:check(x,before[0])
        check(p['identical_control_error'],0)
    result=dict(passed=True,decimal_precision=50,checked_scalars=len(errors),max_disagreement=max(errors),
                exact_joint_log_odds_determinant=1,report_sha256=digest(path),source_sha256=digest(Path(__file__)))
    (ROOT/'results/shared_kv_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    main()
