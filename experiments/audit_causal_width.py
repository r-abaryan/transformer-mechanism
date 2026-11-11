"""Independent decimal checks on a fixed causal pattern and value outputs."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=ROOT/'results/causal_width.json'; r=json.loads(path.read_text()); assert r['passed']
    for p,h in r['hashes'].items(): assert digest(ROOT/p)==h
    assert r['k']==[[1,0],[0,1],[0,0],[0,0]]
    s=r['scalar']; u=r['identifiable']; assert s['values']==s['k']==[[0],[1],[2],[3]]
    assert u['values']==[[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
    errors=[]; output_errors=[]; differences=[]; negative=[]; recoveries=[]
    def check(a,b):
        err=abs(a-float(b)); assert err<1e-10; errors.append(err)
    with localcontext() as ctx:
        ctx.prec=50
        def softmax(row):
            exp=[x.exp() for x in row]; return [x/sum(exp) for x in exp]
        expected=[[0],[0,0],[1,0,0],[0,1,0,0]]; rows=[]
        for i in range(4):
            a=softmax([Decimal(x) for x in expected[i]])+[Decimal(0)]*(3-i); rows.append(a)
            query=Decimal(str(s['q'][i][0])); b=softmax([query*j for j in range(i+1)])+[Decimal(0)]*(3-i)
            out=sum(b[j]*j for j in range(4)); target=sum(a[j]*j for j in range(4))
            output_errors.append(abs(out-target)); check(s['output'][i][0],out); check(s['target_output'][i][0],target)
            for j in range(4):
                score=sum(Decimal(str(r['q'][i][h]))*Decimal(r['k'][j][h]) for h in range(2))/Decimal(2).sqrt()
                check(r['scores'][i][j],score)
                if j<=i: check(r['scores'][i][j],expected[i][j])
                check(r['attention'][i][j],a[j]); check(s['attention'][i][j],b[j]); differences.append(abs(a[j]-b[j]))
                recover=a[j] if j<3 else 1-sum(a[:3]); check(u['recovered'][i][j],recover); recoveries.append(abs(recover-a[j]))
                if j<3: check(u['output'][i][j],a[j]); negative.append(abs(a[j]-b[j]))
        for i in range(2):
            for j in range(2):
                ratio=(rows[i+2][j]/rows[i+2][2]).ln()
                check(r['log_odds'][i][j],ratio); check(r['log_odds'][i][j],int(i==j))
        # I_2 has determinant 1; positive causal diagonal certifies full rank.
        assert all(rows[i][i]>0 for i in range(4)) and r['attention_rank']==4
        for value in r['log_odds_singular_values']: check(value,1)
        check(s['max_output_error'],max(output_errors)); check(s['max_attention_difference'],max(differences))
        check(u['negative_control_error'],max(negative)); check(u['max_recovery_error'],max(recoveries))
        assert max(output_errors)<Decimal('1e-10') and max(differences)>Decimal('.01') and max(negative)>Decimal('.01')
    result=dict(passed=True,precision=50,checked_scalars=len(errors),max_disagreement=max(errors),report_sha256=digest(path),source_sha256=digest(Path(__file__)))
    (ROOT/'results/causal_width_audit.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))


if __name__=='__main__': main()
