"""Check archived measurements, displayed tables, protected files and local links."""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((OUT/'report.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
lock=json.loads((OUT/'lock.json').read_text())
assert audit['passed'] and audit['report_sha256']==digest(OUT/'report.json')
assert report['lock_sha256']==digest(OUT/'lock.json') and report['csv_sha256']==digest(OUT/'directions.csv')
protected=dict(lock['files'])
prior=json.loads((ROOT/'results/decoder_scale_shape/lock.json').read_text())
for name,value in prior['files'].items():
    if name in protected:assert protected[name]==value
    protected[name]=value
for name,value in protected.items():assert digest(ROOT/name)==value,name
pre=json.loads((ROOT/'results/fitting_transport_review_preflight.json').read_text())
assert pre['passed'] and pre['methods']==2
assert pre['sources']=={n:digest(ROOT/n) for n in ['experiments/review_fitting_transport.py','tests/test_fitting_transport.py']}
with (OUT/'directions.csv').open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
assert len(rows)==len(report['records'])==306
for row,saved in zip(rows,report['records']):
    assert all(float(row[k])==v if isinstance(v,(int,float)) else row[k]==v for k,v in saved.items())
assert sum(x['calls'] for x in report['inventory'])==766
assert sum(x['steps'] for x in report['inventory'])==306
assert all((x['routing_parameters'],x['decoder_parameters'])==(513,102) for x in report['inventory'])
doc=ROOT/'docs/assessments/fitting-transport-review.md';text=doc.read_text(encoding='utf-8')
tables=[b for b in text.split('\n\n') if b.startswith('| Seed |')]
parsed=[[list(map(str.strip,line.strip('|').split('|'))) for line in t.splitlines()[2:]] for t in tables]
assert len(parsed)==2
expected=[[str(x['seed']),'L-BFGS' if x['arm']=='lbfgs' else 'Negative gradient',str(x['calls']),str(x['steps'])] for x in report['inventory']]
assert parsed[0]==expected
expected=[];up=0;later_total=0
for seed in [3473,3479,3481]:
    later=[x for x in report['records'] if x['seed']==seed and x['arm']=='lbfgs' and x['call']>1]
    assert len(later)==62 and all(x['cross_norm_ratio']>0 for x in later)
    down=sum(x['cross_projection']<0 for x in later);ascent=sum(x['cross_projection']>0 for x in later)
    assert down+ascent==62
    expected.append([str(seed),f"{statistics.median(x['cross_norm_ratio'] for x in later):.4f}",
                     f'{down} / 62',f'{ascent} / 62',f"{statistics.median(x['within_cosine'] for x in later):.4f}"])
    up+=ascent;later_total+=len(later)
assert parsed[1]==expected and (up,later_total)==(163,186)
lb=[x for x in report['records'] if x['arm']=='lbfgs'];ng=[x for x in report['records'] if x['arm']=='negative_gradient']
assert len(lb)==189 and len(ng)==117
assert all(x['native_projection']<0 and x['within_projection']<0 for x in lb)
assert all(x['cross_norm_ratio']==0 for x in ng)
assert all(x['cross_norm_ratio']==0 for x in lb if x['call']==1)
assert audit['dense_directions']==189 and audit['archived_calls']==306 and audit['protected_files']==47
assert '1.11e-15' in text and f"{report['seconds']:.2f} seconds" in text
assert all(report[k]==0 for k in ['optimizer_calls','gradient_calls','model_forwards'])
docs=['docs/assessments/fitting-transport-review.md','docs/protocols/fitting-transport-review.md',
      'docs/assessments/research-checkpoint.md','docs/assessments/learning-mechanism-agenda.md',
      'docs/assessments/retention-learning-strategy.md','docs/assessments/research-history.md',
      'docs/assessments/failure-register.md','docs/assessments/README.md','docs/topics/attention/README.md',
      'docs/guides/README.md','docs/results/README.md','docs/protocols/README.md','docs/reference/sources.md','tests/README.md']
links=0
for name in docs:
    path=ROOT/name
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^[a-zA-Z]+:',target) or target.startswith('#'):continue
        assert (path.parent/unquote(target.split('#')[0])).resolve().exists(),(name,target)
        links+=1
result=dict(passed=True,csv_rows=306,measured_table_rows=9,lbfgs_directions=189,negative_gradient_controls=117,
            primary_seed_count=3,transport_calls=186,cross_opposes_descent=163,protected_files=len(protected),
            formal_audit_protected_files=47,local_links=links,additional_model_evaluations=0,additional_training=0,
            report_sha256=digest(OUT/'report.json'),audit_sha256=digest(OUT/'audit.json'),
            files={name:digest(ROOT/name) for name in docs},verifier_sha256=digest(Path(__file__)))
destination=OUT/'document_verification.json';assert not destination.exists()
destination.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
