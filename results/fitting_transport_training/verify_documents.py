"""Independent table/CSV concordance and protected-artifact checks."""
import csv
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((OUT/'report.json').read_text());a=json.loads((OUT/'audit.json').read_text());lock=json.loads((OUT/'lock.json').read_text())
assert a['passed'] and a['report_sha256']==digest(OUT/'report.json')
protected=dict(lock['files'])
for folder in ['decoder_scale_shape','fitting_transport_review']:
    for name,h in json.loads((ROOT/'results'/folder/'lock.json').read_text())['files'].items():
        if name in protected:assert protected[name]==h
        protected[name]=h
for name,h in protected.items():assert digest(ROOT/name)==h,name
pre=json.loads((ROOT/'results/fitting_transport_training_preflight.json').read_text())
profile=json.loads((ROOT/'results/fitting_transport_training_profiles.json').read_text())
assert pre['passed'] and pre['sources']==profile['sources']
assert all(digest(ROOT/n)==h for n,h in pre['sources'].items())
assert r['lock_sha256']==digest(OUT/'lock.json') and r['scores_sha256']==digest(OUT/'scores.pt')
assert r['pretest_sha256']==digest(OUT/'pretest_decision.json') and lock['pools_sha256']==digest(OUT/'pools.pt')
cells={(c['seed'],c['arm'],c['step'],c['pool']):c['metrics'] for c in r['cells']}
assert len(cells)==len(r['cells'])==a['cells']
with (OUT/'evaluation.csv').open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
assert len(rows)==len(cells)*3;seen=set()
for row in rows:
    key=(int(row['seed']),row['arm'],int(row['step']),row['pool']);group=row['group']
    assert (key,group) not in seen;seen.add((key,group))
    m=cells[key]['groups'][group];maps=cells[key]['complete_maps']
    assert int(row['count'])==m['count']
    for name in ['ce','accuracy']:assert (None if row[name]=='' else float(row[name]))==m[name]
    for name,k in [('maps_count','count'),('maps_correct','correct'),('maps_accuracy','accuracy')]:assert float(row[name])==maps[k]
assert seen=={(k,g) for k in cells for g in ['all','unique','repeat']}
text=(ROOT/'docs/topics/attention/fitting-transport-results.md').read_text(encoding='utf-8')
tables=[b for b in text.split('\n\n') if b.startswith('| Seed |')]
parsed=[[list(map(str.strip,line.strip('|').split('|'))) for line in t.splitlines()[2:]] for t in tables]
assert len(parsed)==4
label={'lbfgs':'Full','block_diagonal':'Block'};train=[];held={p:[] for p in ['validation','test']};errors={}
for run in r['runs']:
    step=run['snapshots'][-1]['step'];seed=run['seed'];arm=run['arm']
    ms=[cells[seed,arm,step,f'train_{m}'] for m in [2,8,16,32]]
    e=sum(round(m['groups']['all']['count']*(1-m['groups']['all']['accuracy'])) for m in ms)
    ce=math.fsum(m['groups']['all']['ce'] for m in ms)/4;maps=ms[-1]['complete_maps']['correct'];errors[seed,arm]=e
    assert all(g['accuracy']>=.99 for m in ms for g in m['groups'].values() if g['count'])
    train.append([str(seed),label[arm],str(e),f'{ce:.8g}',f"{100*ms[-1]['groups']['all']['accuracy']:.4f}%",str(maps)])
    for split in held:
        m=cells[seed,arm,step,split+'_32'];g=m['groups']['all'];maps=m['complete_maps']
        held[split].append([str(seed),label[arm],f"{100*g['accuracy']:.4f}%",f"{g['ce']:.4f}",f"{maps['correct']} / {maps['count']}"])
assert parsed[:3]==[train,held['validation'],held['test']]
budget=[]
for seed in [3473,3479,3481]:
    full,block=[next(x for x in r['runs'] if (x['seed'],x['arm'])==(seed,arm)) for arm in ['lbfgs','block_diagonal']]
    budget.append([str(seed),f"{full['closure_evaluations']} / {block['closure_evaluations']}",f"{full['optimizer_calls']} / {block['optimizer_calls']}"])
    assert full['closure_evaluations']<=128 and block['closure_evaluations']<=128
    for split,load in itertools.product(['validation','test'],[16,32]):
        x,y=[cells[seed,z['arm'],z['snapshots'][-1]['step'],f'{split}_{load}']['groups']['all']['ce'] for z in [full,block]]
        assert y>x
assert parsed[3]==budget
assert [errors[s,'lbfgs'] for s in [3473,3479,3481]]==[0,0,6]
assert [errors[s,'block_diagonal'] for s in [3473,3479,3481]]==[0,0,24]
primary=all(errors[s,'block_diagonal']>errors[s,'lbfgs'] for s in [3473,3479,3481])
assert r['decision']['all_seed_transport_fitting_benefit']==primary==False
for pair in r['decision']['pairs']:
    assert pair['removal_increases_errors']==(errors[pair['seed'],'block_diagonal']>errors[pair['seed'],'lbfgs'])
assert all(c['historical_native_exact'] and c['first_step_identical'] and c['later_directions_differ'] for c in a['pair_controls'])
for name in ['optimizer_calls','answer_targets','histories','input_events']:assert r[name]==sum(x[name] for x in r['runs'])
assert r['gradient_evaluations']==sum(x['closure_evaluations'] for x in r['runs'])==765
assert r['optimizer_calls']==369 and r['answer_targets']==765*44544
assert r['histories']==765*3072 and r['input_events']==765*3072*32
assert r['evaluation_targets']==sum(m['groups']['all']['count'] for m in cells.values())
moved=[s for c in a['checks'] for s in c['steps'] if s['moved']]
assert f"{sum(s['strong_wolfe'] for s in moved)}/{len(moved)} moved steps" in text
assert f"{r['seconds']:.2f} seconds" in text
docs=['docs/topics/attention/fitting-transport-results.md','docs/topics/attention/fitting-transport-guide.md',
      'docs/protocols/fitting-transport-training.md','docs/assessments/research-checkpoint.md',
      'docs/assessments/learning-mechanism-agenda.md','docs/assessments/retention-learning-strategy.md',
      'docs/assessments/failure-register.md','docs/assessments/research-history.md','docs/assessments/README.md',
      'docs/topics/attention/README.md','docs/guides/README.md','docs/results/README.md','docs/protocols/README.md',
      'docs/reference/sources.md','tests/README.md']
links=0
for name in docs:
    path=ROOT/name
    for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
        target=target.strip('<>')
        if re.match(r'^[a-zA-Z]+:',target) or target.startswith('#'):continue
        assert (path.parent/unquote(target.split('#')[0])).resolve().exists(),(name,target)
        links+=1
result=dict(passed=True,score_cells=len(cells),csv_rows=len(rows),measured_table_rows=sum(map(len,parsed)),
            primary_rule_passes=primary,protected_files=len(protected),formal_audit_protected_files=a['protected_files'],local_links=links,
            additional_training=0,additional_model_evaluations=0,report_sha256=digest(OUT/'report.json'),audit_sha256=digest(OUT/'audit.json'),
            files={n:digest(ROOT/n) for n in docs},csv_sha256=digest(OUT/'evaluation.csv'),verifier_sha256=digest(Path(__file__)))
path=OUT/'document_verification.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='files'}))
